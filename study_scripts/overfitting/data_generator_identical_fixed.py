import os
import shutil
import argparse
import traceback
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import discoverse
from discoverse.envs import make_env
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase, PyavImageEncoder, recoder_single_arm
from discoverse.utils import SimpleStateMachine, step_func, get_body_tmat


@dataclass
class FixedVariables:
    seed: int
    object_initial_qpos: list
    material_choice: Any
    texture_choice: Any
    randomizer_internal_choices: Dict[str, Any]


@dataclass
class SceneStateSnapshot:
    qpos: list
    qvel: list
    act: list
    ctrl: list
    mocap_pos: list
    mocap_quat: list
    userdata: list
    time: float


def _to_serializable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64, np.integer)):
        return int(value)
    if isinstance(value, (np.bool_, )):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _safe_json_dump(path: str, payload: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)


class UniversalRuntimeTaskExecutor:
    def __init__(
        self,
        task: UniversalTaskBase,
        viewer,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        robot_name: str,
        task_name: str,
        output_root: str,
        sync: bool = False,
    ):
        self.task = task
        self.viewer = viewer
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.renderer = mujoco.Renderer(mj_model)
        self.robot_name = robot_name
        self.task_name = task_name
        self.output_root = output_root
        self.sync = sync
        self.sim_timestep = mj_model.opt.timestep
        self.viewer_fps = 60
        self.resolved_states = task.task_config.get_resolved_states()
        self.total_states = len(self.resolved_states)
        self.n_arm_joints = len(task.robot_interface.arm_joints)
        self.gripper_ctrl_idx = self.n_arm_joints
        self.joint_pos_sensor_idx = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            for sensor_name in task.robot_interface.joint_pos_sensors
        ]
        self.mujoco_ctrl_dim = mj_model.nu
        self.move_speed = 1.5
        self.max_time = 20.0
        self.task.randomizer.set_viewer(viewer)
        self.task.randomizer.set_renderer(self.renderer)
        self.record_frq = self.task.task_config.record_fps

        # 저장 카메라 고정
        self.camera_cfgs = {
            'global_cam': {'name': 'global_cam', 'width': 640, 'height': 480},
            'wrist_cam': {'name': 'wrist_cam', 'width': 640, 'height': 480},
        }
        self.camera_encoders = {}
        self.fixed_variables: Optional[FixedVariables] = None
        self.scene_snapshot: Optional[SceneStateSnapshot] = None
        self.current_sample_idx = 0
        self.reset_runtime_state()

    def set_global_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)

    def reset_runtime_state(self):
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = self.total_states
        self.target_control = np.zeros(self.mujoco_ctrl_dim)
        self.action = np.zeros(self.mujoco_ctrl_dim)
        self.joint_move_ratio = np.ones(self.mujoco_ctrl_dim)
        self.running = True
        self.success = False
        self.viewer_closed = False
        self.current_delay = 0.0
        self.delay_start_sim_time = None

    def get_current_qpos(self):
        return self.mj_data.qpos.copy()

    def _capture_randomizer_snapshot(self) -> Dict[str, Any]:
        snapshot = {}
        randomizer = getattr(self.task, 'randomizer', None)
        if randomizer is None:
            return snapshot

        skip_keys = {'viewer', 'renderer', 'mj_model', 'mj_data', 'task', 'scene', 'objects'}
        if hasattr(randomizer, 'export_current_choices') and callable(randomizer.export_current_choices):
            try:
                return _to_serializable(randomizer.export_current_choices())
            except Exception:
                pass

        if hasattr(randomizer, '__dict__'):
            for key, value in randomizer.__dict__.items():
                if key.startswith('_') or key in skip_keys:
                    continue
                if isinstance(value, (int, float, str, bool, list, tuple, dict, np.ndarray, np.floating, np.integer, np.bool_)):
                    snapshot[key] = _to_serializable(value)
        return snapshot

    def _extract_named_choice(self, candidates, default_value):
        randomizer = getattr(self.task, 'randomizer', None)
        for owner in [randomizer, self.task]:
            if owner is None:
                continue
            for name in candidates:
                if hasattr(owner, name):
                    value = getattr(owner, name)
                    if not callable(value):
                        return _to_serializable(value)
        return default_value

    def _capture_scene_snapshot(self) -> SceneStateSnapshot:
        return SceneStateSnapshot(
            qpos=self.mj_data.qpos.copy().tolist(),
            qvel=self.mj_data.qvel.copy().tolist(),
            act=self.mj_data.act.copy().tolist() if self.mj_data.act is not None else [],
            ctrl=self.mj_data.ctrl.copy().tolist(),
            mocap_pos=self.mj_data.mocap_pos.copy().tolist() if self.mj_model.nmocap > 0 else [],
            mocap_quat=self.mj_data.mocap_quat.copy().tolist() if self.mj_model.nmocap > 0 else [],
            userdata=self.mj_data.userdata.copy().tolist() if self.mj_model.nuserdata > 0 else [],
            time=float(self.mj_data.time),
        )

    def build_fixed_scene_once(self, seed: int = 1000):
        """
        랜덤화는 딱 1번만 수행하고, 그 결과 장면 상태를 스냅샷으로 저장합니다.
        이후 5개 데이터는 모두 이 스냅샷을 그대로 복원해서 완전히 동일하게 생성합니다.
        """
        self.set_global_seed(seed)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')

        if hasattr(self.task, 'randomize_scene'):
            self.task.randomize_scene()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.fixed_variables = FixedVariables(
            seed=seed,
            object_initial_qpos=self.get_current_qpos().tolist(),
            material_choice=self._extract_named_choice(
                ['material_choice', 'current_material', 'material_name', 'material'],
                f'fixed_seed_{seed}_material',
            ),
            texture_choice=self._extract_named_choice(
                ['texture_choice', 'current_texture', 'texture_name', 'texture'],
                f'fixed_seed_{seed}_texture',
            ),
            randomizer_internal_choices=self._capture_randomizer_snapshot(),
        )
        self.scene_snapshot = self._capture_scene_snapshot()

    def restore_fixed_scene(self):
        if self.scene_snapshot is None:
            raise RuntimeError('scene_snapshot is not initialized. Call build_fixed_scene_once() first.')

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        snapshot = self.scene_snapshot
        self.mj_data.qpos[:] = np.asarray(snapshot.qpos, dtype=float)
        self.mj_data.qvel[:] = np.asarray(snapshot.qvel, dtype=float)
        if len(snapshot.act) > 0 and self.mj_data.act is not None:
            self.mj_data.act[:] = np.asarray(snapshot.act, dtype=float)
        self.mj_data.ctrl[:] = np.asarray(snapshot.ctrl, dtype=float)

        if self.mj_model.nmocap > 0 and len(snapshot.mocap_pos) > 0:
            self.mj_data.mocap_pos[:] = np.asarray(snapshot.mocap_pos, dtype=float)
            self.mj_data.mocap_quat[:] = np.asarray(snapshot.mocap_quat, dtype=float)

        if self.mj_model.nuserdata > 0 and len(snapshot.userdata) > 0:
            self.mj_data.userdata[:] = np.asarray(snapshot.userdata, dtype=float)

        self.mj_data.time = float(snapshot.time)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')

    def get_observation(self):
        obs = {
            'time': self.mj_data.time,
            'jq': self.mj_data.sensordata[self.joint_pos_sensor_idx].tolist(),
            'action': self.action[:self.mujoco_ctrl_dim].tolist(),
            'img': {},
            'sample_idx': self.current_sample_idx,
        }
        for camera_name in self.camera_cfgs.keys():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                continue
            self.set_renderer_size(self.camera_cfgs[camera_name]['width'], self.camera_cfgs[camera_name]['height'])
            img = self.get_rgb_image(camera_name).copy()
            obs['img'][camera_name] = img
        return obs

    def get_rgb_image(self, camera_name):
        scene_opt = mujoco.MjvOption()
        for i in range(len(scene_opt.sitegroup)):
            scene_opt.sitegroup[i] = 0
        self.renderer.update_scene(self.mj_data, camera_name, scene_option=scene_opt)
        return self.renderer.render()

    def set_renderer_size(self, width, height):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height

    def set_target_from_primitive(self, state_config):
        try:
            primitive = state_config['primitive']
            params = state_config.get('params', {})
            gripper_state = state_config.get('gripper_state', 'open')

            if primitive == 'move_to_object':
                object_name = params.get('object_name', '')
                offset = np.array(params.get('offset', [0, 0, 0]))
                if object_name:
                    object_tmat = get_body_tmat(self.mj_data, object_name)
                    target_pos = object_tmat[:3, 3] + offset
                    site_name = self.task.robot_interface.robot_config.end_effector_site
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                    full_current_qpos = self.mj_data.qpos.copy()
                    solution, converged, _ = self.task.robot_interface.ik_solver.solve_ik(
                        target_pos, target_rmat, full_current_qpos
                    )
                    if converged:
                        self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                        self.set_mocap_target(
                            'target',
                            target_pos,
                            Rotation.from_matrix(target_rmat).as_quat()[[3, 0, 1, 2]],
                        )
                    else:
                        return False

            elif primitive == 'move_relative':
                offset = np.array(params.get('offset', [0, 0, 0]))
                site_name = self.task.robot_interface.robot_config.end_effector_site
                site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                current_pos = self.mj_data.site_xpos[site_id].copy()
                target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                target_pos = current_pos + offset
                full_current_qpos = self.mj_data.qpos.copy()
                solution, converged, _ = self.task.robot_interface.ik_solver.solve_ik(
                    target_pos, target_rmat, full_current_qpos
                )
                if converged:
                    self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                    self.set_mocap_target(
                        'target',
                        target_pos,
                        Rotation.from_matrix(target_rmat).as_quat()[[3, 0, 1, 2]],
                    )
                else:
                    return False

            if gripper_state == 'open':
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.open()
            elif gripper_state == 'close':
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.close()

            current_ctrl = self.mj_data.ctrl[:self.mujoco_ctrl_dim].copy()
            dif = np.abs(current_ctrl - self.target_control)
            self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def set_mocap_target(self, target_name, target_pos, target_quat):
        mocap_id = self.mj_model.body(target_name).mocapid
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = target_pos
            self.mj_data.mocap_quat[mocap_id] = target_quat
            self.mj_model.geom(f'{target_name}_box').rgba = (0, 0, 0, 0.0)

    def check_action_done(self):
        current_qpos = self.get_current_qpos()
        position_error = np.linalg.norm(current_qpos[:self.n_arm_joints] - self.target_control[:self.n_arm_joints])
        position_done = position_error < 0.02
        if self.current_delay > 0 and self.delay_start_sim_time is not None:
            delay_elapsed = self.mj_data.time - self.delay_start_sim_time
            if not (delay_elapsed >= self.current_delay):
                return False
        return position_done

    def step(self, decimation=5):
        try:
            if self.stm.trigger():
                if self.stm.state_idx < self.total_states:
                    state_config = self.resolved_states[self.stm.state_idx]
                    self.current_delay = state_config.get('delay', 0.0)
                    if not self.set_target_from_primitive(state_config):
                        return False
                    if self.current_delay > 0:
                        self.delay_start_sim_time = self.mj_data.time
                else:
                    self.success = self.check_task_success()
                    self.running = False
                    return True
            elif self.mj_data.time > self.max_time:
                self.running = False
                return False
            else:
                self.stm.update()
                if self.check_action_done():
                    self.current_delay = 0.0
                    self.delay_start_sim_time = None
                    self.stm.next()

            for i in range(self.n_arm_joints):
                self.action[i] = step_func(
                    self.action[i],
                    self.target_control[i],
                    self.move_speed * float(decimation) * self.joint_move_ratio[i] * self.mj_model.opt.timestep,
                )
            self.action[self.gripper_ctrl_idx] = self.target_control[self.gripper_ctrl_idx]
            self.mj_data.ctrl[:self.mujoco_ctrl_dim] = self.action[:self.mujoco_ctrl_dim]
            for _ in range(decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)
            return True
        except Exception:
            self.running = False
            return False

    def check_task_success(self):
        return self.task.check_success()

    def prepare_episode(self, sample_idx: int):
        self.current_sample_idx = sample_idx
        self.restore_fixed_scene()
        self.reset_runtime_state()
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]

        sample_dir = os.path.join(self.output_root, f'sample_{sample_idx:02d}')
        os.makedirs(sample_dir, exist_ok=True)
        self.save_dir = sample_dir

        # 동일한 변수 정보 저장
        if self.fixed_variables is None:
            raise RuntimeError('fixed_variables is not initialized.')
        _safe_json_dump(os.path.join(sample_dir, 'variables.json'), asdict(self.fixed_variables))
        _safe_json_dump(
            os.path.join(sample_dir, 'scene_snapshot.json'),
            asdict(self.scene_snapshot),
        )

        self.camera_encoders = {}
        for cam_name in self.camera_cfgs.keys():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) >= 0:
                self.camera_encoders[cam_name] = PyavImageEncoder(
                    self.camera_cfgs[cam_name]['width'],
                    self.camera_cfgs[cam_name]['height'],
                    self.save_dir,
                    cam_name,
                )
            else:
                print(f' ⚠️ 경고: {cam_name} 카메라를 찾을 수 없습니다!')

    def run(self):
        step_count = 0
        obs_lst = []
        last_render_time = 0.0

        while self.running:
            if not self.step():
                break
            step_count += 1

            if self.viewer is not None:
                if not self.viewer.is_running():
                    self.viewer_closed = True
                    self.running = False
                    return False
                if self.mj_data.time - last_render_time > (1.0 / self.viewer_fps):
                    self.viewer.sync()
                    last_render_time = self.mj_data.time

            if len(obs_lst) < self.mj_data.time * self.record_frq:
                obs = self.get_observation()
                imgs = obs.pop('img')
                for cam_id, img in imgs.items():
                    if cam_id in self.camera_encoders:
                        self.camera_encoders[cam_id].encode(img, obs['time'])
                obs_lst.append(obs)

        for ec in self.camera_encoders.values():
            ec.close()

        episode_meta = {
            'success': bool(self.success),
            'step_count': int(step_count),
            'sample_idx': int(self.current_sample_idx),
            'fixed_variables': None if self.fixed_variables is None else asdict(self.fixed_variables),
        }
        _safe_json_dump(os.path.join(self.save_dir, 'episode_metadata.json'), episode_meta)

        if not self.success:
            print(f' ❌ Task failed. 삭제 처리: {self.save_dir}')
            shutil.rmtree(self.save_dir, ignore_errors=True)
        else:
            recoder_single_arm(self.save_dir, obs_lst)
            print(f' ✅ 동일 데이터와 변수 정보가 저장되었습니다: {self.save_dir}')
        return self.success


def generate_robot_task_model(robot_name, task_name):
    xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, 'mjcf/tmp', f'{robot_name}_{task_name}.xml')
    make_env(robot_name, task_name, xml_path)

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    camera_tag = '  <camera name="global_cam" pos="0 0.4 1.2" xyaxes="1 0 0 0 0.707 0.707"/>\n</worldbody>'
    if '</worldbody>' in xml_data and 'name="global_cam"' not in xml_data:
        xml_data = xml_data.replace('</worldbody>', camera_tag)
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_data)
    return xml_path


def create_simple_visualizer(mj_model, mj_data):
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    if mj_model.ncam > 0:
        viewer.cam.fixedcamid = 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return viewer


def main(
    robot_name='piper',
    task_name='place_block',
    sync=False,
    once=False,
    headless=False,
    num_samples=5,
    fixed_seed=1000,
):
    print(discoverse.__logo__)
    xml_path = generate_robot_task_model(robot_name, task_name)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    for i in range(mj_model.ngeom):
        body_id = mj_model.geom_bodyid[i]
        if mj_model.body_mocapid[body_id] != -1:
            mj_model.geom_rgba[i][3] = 0.0
    for i in range(mj_model.nsite):
        mj_model.site_rgba[i][3] = 0.0

    mj_data = mujoco.MjData(mj_model)
    viewer = None if headless else create_simple_visualizer(mj_model, mj_data)

    configs_root = os.path.join(DISCOVERSE_ROOT_DIR, 'discoverse', 'configs')
    robot_config_path = os.path.join(configs_root, 'robots', f'{robot_name}.yaml')
    task_config_path = os.path.join(configs_root, 'tasks', f'{task_name}.yaml')
    task = UniversalTaskBase(
        robot_config_path=robot_config_path,
        task_config_path=task_config_path,
        mj_model=mj_model,
        mj_data=mj_data,
    )

    dataset_root = os.path.join(
        DISCOVERSE_ROOT_DIR,
        'data',
        f'{robot_name}_{task_name}_identical_fixed_data',
    )
    os.makedirs(dataset_root, exist_ok=True)

    try:
        executor = UniversalRuntimeTaskExecutor(
            task=task,
            viewer=viewer,
            mj_model=mj_model,
            mj_data=mj_data,
            robot_name=robot_name,
            task_name=task_name,
            output_root=dataset_root,
            sync=sync,
        )

        # 랜덤화는 딱 1번만 수행하고, 그 결과를 모든 샘플이 공유
        executor.build_fixed_scene_once(seed=fixed_seed)

        _safe_json_dump(
            os.path.join(dataset_root, 'fixed_variables.json'),
            asdict(executor.fixed_variables),
        )
        _safe_json_dump(
            os.path.join(dataset_root, 'dataset_manifest.json'),
            {
                'robot_name': robot_name,
                'task_name': task_name,
                'num_samples': num_samples,
                'fixed_seed': fixed_seed,
                'all_samples_identical': True,
                'fixed_variables': asdict(executor.fixed_variables),
            },
        )

        success_count = 0
        attempt_count = 0
        while success_count < num_samples:
            attempt_count += 1
            print('\n' + '=' * 60)
            print(f'📦 동일 샘플 생성 중 | 시도 {attempt_count} | 저장 {success_count}/{num_samples}')
            print('=' * 60)
            executor.prepare_episode(sample_idx=success_count)
            success = executor.run()
            if success:
                success_count += 1
                print(f'🎉 동일 데이터 저장 완료 ({success_count}/{num_samples})')
            if once or executor.viewer_closed:
                break

        print(f'\n✅ 총 {success_count}/{num_samples}개의 동일한 데이터 생성 완료')
        print(f'📁 저장 위치: {dataset_root}')

    except Exception:
        traceback.print_exc()
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='5개의 완전히 동일한 데이터 + 변수 정보 저장 생성기')
    parser.add_argument('-r', '--robot', type=str, default='piper')
    parser.add_argument('-t', '--task', type=str, default='place_block')
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('-1', '--once', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--num-samples', type=int, default=5, help='동일하게 생성할 샘플 개수')
    parser.add_argument('--fixed-seed', type=int, default=1000, help='최초 1회 랜덤화에 사용할 고정 seed')
    args = parser.parse_args()

    main(
        robot_name=args.robot,
        task_name=args.task,
        sync=args.sync,
        once=args.once,
        headless=args.headless,
        num_samples=args.num_samples,
        fixed_seed=args.fixed_seed,
    )
