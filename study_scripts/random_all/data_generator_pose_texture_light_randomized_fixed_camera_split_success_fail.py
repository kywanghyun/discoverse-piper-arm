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
        success_subdir: str = 'success_samples',
        failed_subdir: str = 'failed_samples',
        save_failed: bool = True,
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
        self.success_subdir = success_subdir
        self.failed_subdir = failed_subdir
        self.save_failed = save_failed

        self.success_root = os.path.join(self.output_root, self.success_subdir)
        self.failed_root = os.path.join(self.output_root, self.failed_subdir)
        self.tmp_root = os.path.join(self.output_root, '_tmp_generation')
        os.makedirs(self.success_root, exist_ok=True)
        os.makedirs(self.tmp_root, exist_ok=True)
        if self.save_failed:
            os.makedirs(self.failed_root, exist_ok=True)

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
        self.base_camera_poses: Dict[str, Dict[str, Any]] = {}
        self.current_episode_variables: Optional[EpisodeVariables] = None
        self.current_sample_idx = 0
        self.current_attempt_idx = 0
        self.last_episode_seed = 0
        self.save_dir = ''
        self.final_episode_dir = ''
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

    def _capture_camera_poses(self) -> Dict[str, Dict[str, Any]]:
        camera_poses = {}
        for cam_name in self.camera_cfgs.keys():
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                continue
            camera_poses[cam_name] = {
                'pos': self.mj_model.cam_pos[cam_id].copy().tolist(),
                'quat': self.mj_model.cam_quat[cam_id].copy().tolist(),
            }
        return camera_poses

    def _restore_camera_poses(self):
        for cam_name, pose in self.base_camera_poses.items():
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                continue
            self.mj_model.cam_pos[cam_id] = np.asarray(pose['pos'], dtype=float)
            self.mj_model.cam_quat[cam_id] = np.asarray(pose['quat'], dtype=float)

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
        q = self.mj_data.qpos[qadr:qadr + 7].copy()
        return {
            'body_name': body_name,
            'qpos_adr': int(qadr),
            'pos': q[:3].tolist(),
            'quat': q[3:7].tolist(),
        }

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray):
        qadr = self._get_free_joint_qpos_adr(body_name)
        self.mj_data.qpos[qadr:qadr + 3] = pos
        self.mj_data.qpos[qadr + 3:qadr + 7] = quat

    def build_base_scene_once(self):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')
        self.base_scene_snapshot = self._capture_scene_snapshot()
        self.base_object_pose = self._get_free_body_pose(self.object_name)
        self.base_bowl_pose = self._get_free_body_pose(self.bowl_name)
        self.base_camera_poses = self._capture_camera_poses()

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
        self._restore_camera_poses()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')

    def randomize_visuals_keep_camera_fixed(self):
        if hasattr(self.task, 'randomize_scene'):
            try:
                self.task.randomize_scene()
            except Exception:
                traceback.print_exc()
        self._restore_camera_poses()
        mujoco.mj_forward(self.mj_model, self.mj_data)

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

        chosen_obj = None
        chosen_bowl = None
        for _ in range(200):
            obj_xy = base_obj_pos[:2] + np.random.uniform(-self.object_xy_range, self.object_xy_range, size=2)
            bowl_xy = base_bowl_pos[:2] + np.random.uniform(-self.bowl_xy_range, self.bowl_xy_range, size=2)
            if np.linalg.norm(obj_xy - bowl_xy) < self.min_object_bowl_dist:
                continue
            chosen_obj = np.array([obj_xy[0], obj_xy[1], obj_z], dtype=float)
            chosen_bowl = np.array([bowl_xy[0], bowl_xy[1], bowl_z], dtype=float)
            break
        if chosen_obj is None or chosen_bowl is None:
            raise RuntimeError('Failed to sample valid randomized object/bowl positions.')

        self._set_free_body_pose(self.object_name, chosen_obj, base_obj_quat)
        self._set_free_body_pose(self.bowl_name, chosen_bowl, base_bowl_quat)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        randomizer_snapshot = self._capture_randomizer_snapshot()
        randomizer_snapshot.update({
            'position_randomization_only': False,
            'texture_randomized': True,
            'light_randomized': True,
            'camera_pose_randomized': False,
            'object_name': self.object_name,
            'bowl_name': self.bowl_name,
            'object_pose': {'pos': chosen_obj.tolist(), 'quat': base_obj_quat.tolist()},
            'bowl_pose': {'pos': chosen_bowl.tolist(), 'quat': base_bowl_quat.tolist()},
            'fixed_camera_poses': _to_serializable(self.base_camera_poses),
            'object_xy_range': float(self.object_xy_range),
            'bowl_xy_range': float(self.bowl_xy_range),
            'min_object_bowl_dist': float(self.min_object_bowl_dist),
        })

        self.current_episode_variables = EpisodeVariables(
            seed=seed,
            object_initial_qpos=self.get_current_qpos().tolist(),
            material_choice=self._extract_named_choice(
                ['material_choice', 'current_material', 'material_name', 'material'],
                'randomized_material',
            ),
            texture_choice=self._extract_named_choice(
                ['texture_choice', 'current_texture', 'texture_name', 'texture'],
                'randomized_texture',
            ),
            randomizer_internal_choices=randomizer_snapshot,
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
                        self.set_mocap_target('target', target_pos, np.array([1.0, 0.0, 0.0, 0.0]))
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
                    self.set_mocap_target('target', target_pos, np.array([1.0, 0.0, 0.0, 0.0]))
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

    def prepare_episode(self, episode_idx: int, episode_seed: int):
        self.current_sample_idx = episode_idx
        self.restore_base_scene()
        self.randomize_visuals_keep_camera_fixed()
        self.randomize_object_and_bowl_positions(seed=episode_seed)
        self.reset_runtime_state()
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]

    def run_episode(self):
        step_count = 0
        obs_lst = []
        while self.running:
            if not self.step():
                break
            step_count += 1
            if len(obs_lst) < self.mj_data.time * self.record_frq:
                obs_lst.append(self.get_observation())
        return bool(self.success), step_count, obs_lst

    def _check_success(self) -> bool:
        if self.args.quiet_success_check:
            with contextlib.redirect_stdout(io.StringIO()):
                return bool(self.task.check_success())
        return bool(self.task.check_success())

    def _make_side_by_side_frame(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        if 'global_cam' not in obs or 'wrist_cam' not in obs:
            return None
        left = cv2.cvtColor(obs['global_cam'], cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(obs['wrist_cam'], cv2.COLOR_RGB2BGR)
        combined = np.concatenate([left, right], axis=1)
        cv2.putText(combined, 'GLOBAL', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, 'WRIST', (left.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return combined

    def evaluate_one_episode(self, episode_idx: int, seed: int) -> Dict[str, Any]:
        self.prepare_episode(episode_idx=episode_idx, episode_seed=seed)

        video_writer = None
        if self.args.save_video and episode_idx < self.args.max_video_episodes:
            combined_width = self.camera_cfgs['global_cam']['width'] + self.camera_cfgs['wrist_cam']['width']
            combined_height = self.camera_cfgs['global_cam']['height']
            video_path = str(self.output_dir_path / 'videos' / f'episode_{episode_idx:04d}_seed_{seed}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.args.video_fps, (combined_width, combined_height))

        trajectory = []
        success = False
        success_first_step = None
        extra_after_success = 0
        visual_info = self._capture_randomizer_snapshot()
        visual_info.update({
            'texture_randomized': bool(self.args.randomize_visuals),
            'light_randomized': bool(self.args.randomize_visuals),
            'camera_pose_randomized': False,
            'fixed_camera_poses': _to_serializable(self.base_camera_poses),
            'material_choice': self._extract_named_choice(
                ['material_choice', 'current_material', 'material_name', 'material'],
                'randomized_material' if self.args.randomize_visuals else 'base_material',
            ),
            'texture_choice': self._extract_named_choice(
                ['texture_choice', 'current_texture', 'texture_name', 'texture'],
                'randomized_texture' if self.args.randomize_visuals else 'base_texture',
            ),
        })
        sampled_pose_info = {
            'seed': int(seed),
            'object_name': self.args.object_name,
            'bowl_name': self.args.bowl_name,
            'object_xy_range': float(self.args.object_xy_range),
            'bowl_xy_range': float(self.args.bowl_xy_range),
            'min_object_bowl_dist': float(self.args.min_object_bowl_dist),
        }

        for step in range(self.args.max_steps):
            obs = self.get_policy_observation()
            action = self._predict_action(obs)
            self.mj_data.ctrl[: self.mujoco_ctrl_dim] = np.asarray(action[: self.mujoco_ctrl_dim], dtype=float)
            for _ in range(self.args.decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)

            trajectory.append({
                'step': int(step),
                'time': float(self.mj_data.time),
                'action': action.tolist(),
                'joint_state': self.mj_data.sensordata[self.joint_pos_sensor_idx].tolist(),
            })

            if video_writer is not None:
                side_by_side = self._make_side_by_side_frame(obs)
                if side_by_side is not None:
                    video_writer.write(side_by_side)

            current_success = self._check_success()
            if current_success and success_first_step is None:
                success_first_step = step
                extra_after_success = int(self.args.success_hold_steps)
                print(f'[INFO] episode={episode_idx} seed={seed}: success at step={step}, recording {extra_after_success} extra steps...')

            if success_first_step is not None:
                success = True
                if extra_after_success <= 0:
                    break
                extra_after_success -= 1

            if self.args.log_every > 0 and step % self.args.log_every == 0:
                print(f'[INFO] episode={episode_idx} seed={seed}: step={step} success={current_success} time={self.mj_data.time:.3f}')

        if video_writer is not None:
            video_writer.release()

        if self.args.save_trajectories:
            save_json(self.output_dir_path / f'episode_{episode_idx:04d}_trajectory.json', {'trajectory': trajectory})

        return {
            'episode_idx': int(episode_idx),
            'seed': int(seed),
            'success': bool(success),
            'success_first_step': None if success_first_step is None else int(success_first_step),
            'num_steps': len(trajectory),
            'sampled_pose_info': sampled_pose_info,
            'visual_info': visual_info,
        }

    def run(self):
        all_results: List[Dict[str, Any]] = []
        for episode_idx in range(self.args.num_eval_episodes):
            seed = int(self.args.eval_seed_start + episode_idx)
            print('\n' + '=' * 80)
            print(f'[EVAL {episode_idx + 1}/{self.args.num_eval_episodes}] unseen pose+visual seed={seed}')
            print('=' * 80)
            try:
                result = self.evaluate_one_episode(episode_idx=episode_idx, seed=seed)
                all_results.append(result)
                print(f"[RESULT] episode={episode_idx} seed={seed} success={result['success']} steps={result['num_steps']}")
            except Exception as e:
                traceback.print_exc()
                all_results.append({
                    'episode_idx': int(episode_idx),
                    'seed': int(seed),
                    'success': False,
                    'error': str(e),
                })

        valid_results = [r for r in all_results if 'error' not in r]
        success_results = [r for r in valid_results if r.get('success', False)]
        aggregate = {
            'checkpoint': str(self.checkpoint_path),
            'dataset_repo_id': self.args.dataset_repo_id,
            'dataset_root': None if self.dataset_root_path is None else str(self.dataset_root_path),
            'robot': self.args.robot,
            'task': self.args.task,
            'task_description': self.args.task_description,
            'num_eval_episodes_requested': int(self.args.num_eval_episodes),
            'num_eval_episodes_valid': len(valid_results),
            'num_eval_episodes_success': len(success_results),
            'success_rate': None if len(valid_results) == 0 else float(len(success_results) / len(valid_results)),
            'avg_success_first_step': None if len(success_results) == 0 else float(np.mean([r['success_first_step'] for r in success_results if r['success_first_step'] is not None])),
            'avg_num_steps': None if len(valid_results) == 0 else float(np.mean([r['num_steps'] for r in valid_results if 'num_steps' in r])),
            'eval_seed_start': int(self.args.eval_seed_start),
            'object_name': self.args.object_name,
            'bowl_name': self.args.bowl_name,
            'object_xy_range': float(self.args.object_xy_range),
            'bowl_xy_range': float(self.args.bowl_xy_range),
            'min_object_bowl_dist': float(self.args.min_object_bowl_dist),
            'randomize_visuals': bool(self.args.randomize_visuals),
            'note': 'All evaluated object/bowl poses are generated at evaluation time using seeds intended to be outside the training seed range. Visuals are also randomized while camera poses remain fixed.',
        }

        save_json(self.output_dir_path / 'aggregate_eval_summary.json', aggregate)
        save_json(self.output_dir_path / 'per_episode_results.json', {'results': all_results})

        print('\n' + '=' * 80)
        print('✅ Unseen pose + texture/light evaluation completed')
        print(f"- num_eval_episodes_valid: {aggregate['num_eval_episodes_valid']}")
        print(f"- num_eval_episodes_success: {aggregate['num_eval_episodes_success']}")
        print(f"- success_rate: {aggregate['success_rate']}")
        print(f"- avg_success_first_step: {aggregate['avg_success_first_step']}")
        print(f"- avg_num_steps: {aggregate['avg_num_steps']}")
        print(f"- output_dir: {self.output_dir_path}")
        print('=' * 80)


def build_argparser():
    parser = argparse.ArgumentParser(description='학습된 데이터를 바탕으로 scripted policy를 success/fail과 함께 평가 (texture/light 랜덤화, 카메라 고정)')
    parser.add_argument('--robot', type=str, default='piper')
    parser.add_argument('--task', type=str, default='place_block')
    parser.add_argument('--task-description', type=str, default='place the block')
    parser.add_argument('--checkpoint', type=str, required=True, help='평가할 pretrained_model checkpoint 경로')
    parser.add_argument('--dataset-repo-id', type=str, default='apple/piper_place_block_pose_texture_light_randomized_fixed_camera')
    parser.add_argument('--dataset-root', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/lerobot_dataset_pose_texture_light_randomized_fixed_camera'))
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/outputs/eval/piper_place_block_pose_texture_light_randomized_fixed_camera_unseen_pose_visual_eval'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-eval-episodes', type=int, default=100)
    parser.add_argument('--eval-seed-start', type=int, default=100000, help='학습에 사용하지 않은 시드 범위 시작값')
    parser.add_argument('--object-name', type=str, default='block_green')
    parser.add_argument('--bowl-name', type=str, default='bowl_pink')
    parser.add_argument('--object-xy-range', type=float, default=0.08)
    parser.add_argument('--bowl-xy-range', type=float, default=0.08)
    parser.add_argument('--min-object-bowl-dist', type=float, default=0.06)
    parser.add_argument('--max-sample-trials', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--decimation', type=int, default=5)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--max-video-episodes', type=int, default=10)
    parser.add_argument('--save-trajectories', action='store_true')
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--quiet-success-check', action='store_true', default=True)
    parser.add_argument('--success-hold-steps', type=int, default=40)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--randomize-visuals', action='store_true', default=True)
    return parser


def main():
    args = build_argparser().parse_args()
    UnseenPoseTextureLightPolicyEvaluator(args).run()


if __name__ == '__main__':
    main()
