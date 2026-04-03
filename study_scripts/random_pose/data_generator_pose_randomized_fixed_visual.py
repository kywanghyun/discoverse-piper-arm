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

import discoverse
from discoverse.envs import make_env
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase, PyavImageEncoder, recoder_single_arm
from discoverse.utils import SimpleStateMachine, step_func, get_body_tmat


@dataclass
class EpisodeVariables:
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
    if isinstance(value, (np.bool_,)):
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
        object_name: str = 'block_green',
        bowl_name: str = 'bowl_pink',
        object_xy_range: float = 0.08,
        bowl_xy_range: float = 0.08,
        min_object_bowl_dist: float = 0.06,
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
        self.object_name = object_name
        self.bowl_name = bowl_name
        self.object_xy_range = object_xy_range
        self.bowl_xy_range = bowl_xy_range
        self.min_object_bowl_dist = min_object_bowl_dist

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

        self.camera_cfgs = {
            'global_cam': {'name': 'global_cam', 'width': 640, 'height': 480},
            'wrist_cam': {'name': 'wrist_cam', 'width': 640, 'height': 480},
        }
        self.camera_encoders = {}

        self.base_scene_snapshot: Optional[SceneStateSnapshot] = None
        self.base_object_pose: Optional[Dict[str, Any]] = None
        self.base_bowl_pose: Optional[Dict[str, Any]] = None
        self.current_episode_variables: Optional[EpisodeVariables] = None
        self.current_sample_idx = 0
        self.last_episode_seed = 0
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

    def _get_free_joint_qpos_adr(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f'Body not found: {body_name}')
        jnt_adr = int(self.mj_model.body_jntadr[body_id])
        if jnt_adr < 0:
            raise ValueError(f'Body has no joint: {body_name}')
        jnt_id = jnt_adr
        if int(self.mj_model.jnt_type[jnt_id]) != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f'Body joint is not free joint: {body_name}')
        return int(self.mj_model.jnt_qposadr[jnt_id])

    def _get_free_body_pose(self, body_name: str) -> Dict[str, Any]:
        qadr = self._get_free_joint_qpos_adr(body_name)
        q = self.mj_data.qpos[qadr:qadr+7].copy()
        return {
            'body_name': body_name,
            'qpos_adr': int(qadr),
            'pos': q[:3].tolist(),
            'quat': q[3:7].tolist(),
        }

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray):
        qadr = self._get_free_joint_qpos_adr(body_name)
        self.mj_data.qpos[qadr:qadr+3] = pos
        self.mj_data.qpos[qadr+3:qadr+7] = quat

    def build_base_scene_once(self):
        """
        랜덤화 없이 기본 scene을 1회 저장합니다.
        이후 각 episode에서는 object/bowl 위치만 랜덤화합니다.
        텍스처/재질/조명은 기본값을 그대로 유지합니다.
        """
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')
        self.base_scene_snapshot = self._capture_scene_snapshot()
        self.base_object_pose = self._get_free_body_pose(self.object_name)
        self.base_bowl_pose = self._get_free_body_pose(self.bowl_name)

    def restore_base_scene(self):
        if self.base_scene_snapshot is None:
            raise RuntimeError('base_scene_snapshot is not initialized. Call build_base_scene_once() first.')
        snapshot = self.base_scene_snapshot
        mujoco.mj_resetData(self.mj_model, self.mj_data)
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

    def randomize_object_and_bowl_positions(self, seed: int):
        if self.base_object_pose is None or self.base_bowl_pose is None:
            raise RuntimeError('Base poses are not initialized.')
        self.set_global_seed(seed)

        base_obj_pos = np.asarray(self.base_object_pose['pos'], dtype=float)
        base_obj_quat = np.asarray(self.base_object_pose['quat'], dtype=float)
        base_bowl_pos = np.asarray(self.base_bowl_pose['pos'], dtype=float)
        base_bowl_quat = np.asarray(self.base_bowl_pose['quat'], dtype=float)

        obj_z = base_obj_pos[2]
        bowl_z = base_bowl_pos[2]

        success = False
        max_trials = 200
        chosen_obj = None
        chosen_bowl = None
        for _ in range(max_trials):
            obj_xy = base_obj_pos[:2] + np.random.uniform(-self.object_xy_range, self.object_xy_range, size=2)
            bowl_xy = base_bowl_pos[:2] + np.random.uniform(-self.bowl_xy_range, self.bowl_xy_range, size=2)
            if np.linalg.norm(obj_xy - bowl_xy) < self.min_object_bowl_dist:
                continue
            chosen_obj = np.array([obj_xy[0], obj_xy[1], obj_z], dtype=float)
            chosen_bowl = np.array([bowl_xy[0], bowl_xy[1], bowl_z], dtype=float)
            success = True
            break
        if not success:
            raise RuntimeError('Failed to sample valid randomized object/bowl positions.')

        self._set_free_body_pose(self.object_name, chosen_obj, base_obj_quat)
        self._set_free_body_pose(self.bowl_name, chosen_bowl, base_bowl_quat)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.current_episode_variables = EpisodeVariables(
            seed=seed,
            object_initial_qpos=self.get_current_qpos().tolist(),
            material_choice='fixed_default_material',
            texture_choice='fixed_default_texture',
            randomizer_internal_choices={
                'position_randomization_only': True,
                'texture_randomized': False,
                'light_randomized': False,
                'object_name': self.object_name,
                'bowl_name': self.bowl_name,
                'object_pose': {
                    'pos': chosen_obj.tolist(),
                    'quat': base_obj_quat.tolist(),
                },
                'bowl_pose': {
                    'pos': chosen_bowl.tolist(),
                    'quat': base_bowl_quat.tolist(),
                },
                'object_xy_range': float(self.object_xy_range),
                'bowl_xy_range': float(self.bowl_xy_range),
                'min_object_bowl_dist': float(self.min_object_bowl_dist),
            },
        )
        self.last_episode_seed = seed

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
                            np.array([1.0, 0.0, 0.0, 0.0]),
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
                        np.array([1.0, 0.0, 0.0, 0.0]),
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
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f'{target_name}_box')
            if geom_id >= 0:
                self.mj_model.geom_rgba[geom_id][3] = 0.0

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
            traceback.print_exc()
            self.running = False
            return False

    def check_task_success(self):
        return self.task.check_success()

    def prepare_episode(self, sample_idx: int, episode_seed: int):
        self.current_sample_idx = sample_idx
        self.restore_base_scene()
        self.randomize_object_and_bowl_positions(seed=episode_seed)
        self.reset_runtime_state()
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]

        sample_dir = os.path.join(self.output_root, f'sample_{sample_idx:04d}')
        os.makedirs(sample_dir, exist_ok=True)
        self.save_dir = sample_dir

        if self.current_episode_variables is None:
            raise RuntimeError('current_episode_variables is not initialized.')
        _safe_json_dump(os.path.join(sample_dir, 'variables.json'), asdict(self.current_episode_variables))
        _safe_json_dump(os.path.join(sample_dir, 'scene_snapshot.json'), asdict(self._capture_scene_snapshot()))

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
                print(f'⚠️ 경고: {cam_name} 카메라를 찾을 수 없습니다!')

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
            'episode_seed': int(self.last_episode_seed),
            'episode_variables': None if self.current_episode_variables is None else asdict(self.current_episode_variables),
        }
        _safe_json_dump(os.path.join(self.save_dir, 'episode_metadata.json'), episode_meta)

        if not self.success:
            print(f'❌ Task failed. 실패한 데이터만 삭제: {self.save_dir}')
            shutil.rmtree(self.save_dir, ignore_errors=True)
        else:
            recoder_single_arm(self.save_dir, obs_lst)
            print(f'✅ 데이터 저장 완료: {self.save_dir}')
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
    num_samples=500,
    base_seed=1000,
    object_name='block_green',
    bowl_name='bowl_pink',
    object_xy_range=0.08,
    bowl_xy_range=0.08,
    min_object_bowl_dist=0.06,
):
    print(discoverse.__logo__)
    xml_path = generate_robot_task_model(robot_name, task_name)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Hide mocap geoms/sites in viewer too.
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
        f'{robot_name}_{task_name}_pose_randomized_fixed_visual_data',
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
            object_name=object_name,
            bowl_name=bowl_name,
            object_xy_range=object_xy_range,
            bowl_xy_range=bowl_xy_range,
            min_object_bowl_dist=min_object_bowl_dist,
        )

        # Base scene is captured once with no texture/light randomization.
        executor.build_base_scene_once()
        _safe_json_dump(
            os.path.join(dataset_root, 'dataset_manifest.json'),
            {
                'robot_name': robot_name,
                'task_name': task_name,
                'num_samples_target': num_samples,
                'base_seed': base_seed,
                'randomization': {
                    'object_position_only': True,
                    'bowl_position_only': True,
                    'texture_randomized': False,
                    'light_randomized': False,
                    'object_name': object_name,
                    'bowl_name': bowl_name,
                    'object_xy_range': object_xy_range,
                    'bowl_xy_range': bowl_xy_range,
                    'min_object_bowl_dist': min_object_bowl_dist,
                },
            },
        )

        success_count = 0
        attempt_count = 0
        while success_count < num_samples:
            attempt_count += 1
            episode_seed = base_seed + attempt_count
            print('\n' + '=' * 72)
            print(f'📦 Pose-randomized sample generation | attempt {attempt_count} | saved {success_count}/{num_samples} | seed={episode_seed}')
            print('=' * 72)
            executor.prepare_episode(sample_idx=success_count, episode_seed=episode_seed)
            success = executor.run()
            if success:
                success_count += 1
                print(f'🎉 저장 완료 ({success_count}/{num_samples})')
            if once or executor.viewer_closed:
                break

        print(f'\n✅ 총 {success_count}/{num_samples}개의 데이터 생성 완료')
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
    parser = argparse.ArgumentParser(description='목표 물체/접시 위치만 랜덤화하고 텍스처/조명은 고정한 데이터 생성기')
    parser.add_argument('-r', '--robot', type=str, default='piper')
    parser.add_argument('-t', '--task', type=str, default='place_block')
    parser.add_argument('-s', '--sync', action='store_true')
    parser.add_argument('-1', '--once', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--num-samples', type=int, default=500, help='생성할 성공 데이터 개수')
    parser.add_argument('--base-seed', type=int, default=1000, help='episode seed 시작값')
    parser.add_argument('--object-name', type=str, default='block_green')
    parser.add_argument('--bowl-name', type=str, default='bowl_pink')
    parser.add_argument('--object-xy-range', type=float, default=0.08, help='물체 XY 랜덤화 범위 (m)')
    parser.add_argument('--bowl-xy-range', type=float, default=0.08, help='접시 XY 랜덤화 범위 (m)')
    parser.add_argument('--min-object-bowl-dist', type=float, default=0.06, help='초기 object/bowl 최소 거리 (m)')
    args = parser.parse_args()
    main(
        robot_name=args.robot,
        task_name=args.task,
        sync=args.sync,
        once=args.once,
        headless=args.headless,
        num_samples=args.num_samples,
        base_seed=args.base_seed,
        object_name=args.object_name,
        bowl_name=args.bowl_name,
        object_xy_range=args.object_xy_range,
        bowl_xy_range=args.bowl_xy_range,
        min_object_bowl_dist=args.min_object_bowl_dist,
    )
