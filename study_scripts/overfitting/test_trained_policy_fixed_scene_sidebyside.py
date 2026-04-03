import os
import io
import json
import random
import argparse
import traceback
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import mink
import mujoco
import numpy as np
import torch

from discoverse.envs import make_env
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.universal_manipulation import UniversalTaskBase

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


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


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str | Path, payload: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_robot_task_model(robot_name: str, task_name: str) -> str:
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


class FixedScenePolicyTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        self.scene_snapshot_path = Path(args.scene_snapshot).expanduser().resolve()
        self.variables_path = Path(args.variables).expanduser().resolve()
        self.reference_json_path = Path(args.reference_json).expanduser().resolve() if args.reference_json else None
        self.dataset_root_path = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
        self.output_dir_path = Path(args.output_dir).expanduser().resolve()

        for pth, name in [
            (self.checkpoint_path, 'checkpoint'),
            (self.scene_snapshot_path, 'scene_snapshot'),
            (self.variables_path, 'variables'),
        ]:
            if not pth.exists():
                raise FileNotFoundError(f'{name} 경로를 찾을 수 없습니다: {pth}')
        if self.dataset_root_path is not None and not self.dataset_root_path.exists():
            raise FileNotFoundError(f'dataset_root 경로를 찾을 수 없습니다: {self.dataset_root_path}')

        self.scene_snapshot = SceneStateSnapshot(**load_json(self.scene_snapshot_path))
        self.variables = load_json(self.variables_path)
        self.reference = load_json(self.reference_json_path) if self.reference_json_path and self.reference_json_path.exists() else None
        set_global_seed(int(self.variables.get('seed', 1000)))

        self.xml_path = generate_robot_task_model(args.robot, args.task)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model)

        # Hide helper markers / coordinate axes / mocap boxes so they do not appear in rendered inputs.
        self._hide_visual_markers()

        configs_root = os.path.join(DISCOVERSE_ROOT_DIR, 'discoverse', 'configs')
        robot_config_path = os.path.join(configs_root, 'robots', f'{args.robot}.yaml')
        task_config_path = os.path.join(configs_root, 'tasks', f'{args.task}.yaml')
        self.task = UniversalTaskBase(
            robot_config_path=robot_config_path,
            task_config_path=task_config_path,
            mj_model=self.mj_model,
            mj_data=self.mj_data,
        )
        self.task.randomizer.set_viewer(None)
        self.task.randomizer.set_renderer(self.renderer)

        self.mujoco_ctrl_dim = self.mj_model.nu
        self.joint_pos_sensor_idx = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            for sensor_name in self.task.robot_interface.joint_pos_sensors
        ]
        self.camera_cfgs = {
            'global_cam': {'name': 'global_cam', 'width': 640, 'height': 480},
            'wrist_cam': {'name': 'wrist_cam', 'width': 640, 'height': 480},
        }

        self.policy = DiffusionPolicy.from_pretrained(self.checkpoint_path, local_files_only=True).to(self.device).eval()
        self.dataset_meta = self._load_dataset_metadata(args.dataset_repo_id, self.dataset_root_path)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint_path,
            dataset_stats=self.dataset_meta.stats,
            preprocessor_overrides={'device_processor': {'device': str(self.device)}},
        )
        self.expected_obs_keys = [k for k in self.dataset_meta.features.keys() if k.startswith('observation.')]
        print(f'[DEBUG] dataset expected observation keys: {self.expected_obs_keys}')
        os.makedirs(self.output_dir_path, exist_ok=True)

    def _load_dataset_metadata(self, repo_id: str, root: Optional[Path]):
        try:
            if root is not None:
                return LeRobotDatasetMetadata(repo_id=repo_id, root=root)
            return LeRobotDatasetMetadata(repo_id=repo_id)
        except TypeError:
            if root is not None:
                return LeRobotDatasetMetadata(repo_id, root=root)
            return LeRobotDatasetMetadata(repo_id)

    def _hide_visual_markers(self):
        for i in range(self.mj_model.ngeom):
            body_id = int(self.mj_model.geom_bodyid[i])
            if self.mj_model.body_mocapid[body_id] != -1:
                self.mj_model.geom_rgba[i][3] = 0.0
        for i in range(self.mj_model.nsite):
            self.mj_model.site_rgba[i][3] = 0.0
        helper_geom_names = [
            'target_box', 'target_x', 'target_y', 'target_z',
            'origin_x', 'origin_y', 'origin_z',
            'axis_x', 'axis_y', 'axis_z',
        ]
        for geom_name in helper_geom_names:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                self.mj_model.geom_rgba[geom_id][3] = 0.0

    def _set_renderer_size(self, width: int, height: int):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height

    def _render_rgb(self, camera_name: str) -> np.ndarray:
        scene_opt = mujoco.MjvOption()
        for i in range(len(scene_opt.sitegroup)):
            scene_opt.sitegroup[i] = 0
        self.renderer.update_scene(self.mj_data, camera_name, scene_option=scene_opt)
        return self.renderer.render().copy()

    def _apply_camera_settings_from_variables(self):
        choices = self.variables.get('randomizer_internal_choices', {})
        cam_cfg = choices.get('initial_camera_poses', {})
        for camera_name, info in cam_cfg.items():
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                continue
            pos = np.asarray(info.get('pos', []), dtype=float)
            quat = np.asarray(info.get('quat', []), dtype=float)
            if pos.shape == (3,):
                self.mj_model.cam_pos[cam_id] = pos
            if quat.shape == (4,):
                self.mj_model.cam_quat[cam_id] = quat

    def _apply_light_settings_from_variables(self):
        choices = self.variables.get('randomizer_internal_choices', {})
        light_cfg = choices.get('initial_light_states', {})
        if not light_cfg:
            return
        try:
            pos = np.asarray(light_cfg.get('pos', []), dtype=float)
            dr = np.asarray(light_cfg.get('dir', []), dtype=float)
            amb = np.asarray(light_cfg.get('ambient', []), dtype=float)
            dif = np.asarray(light_cfg.get('diffuse', []), dtype=float)
            spc = np.asarray(light_cfg.get('specular', []), dtype=float)
            active = np.asarray(light_cfg.get('active', []), dtype=int)
            n = min(self.mj_model.nlight, len(pos))
            if n > 0:
                self.mj_model.light_pos[:n] = pos[:n]
                self.mj_model.light_dir[:n] = dr[:n]
                self.mj_model.light_ambient[:n] = amb[:n]
                self.mj_model.light_diffuse[:n] = dif[:n]
                self.mj_model.light_specular[:n] = spc[:n]
                self.mj_model.light_active[:n] = active[:n]
        except Exception:
            pass

    def restore_training_scene(self):
        seed = int(self.variables.get('seed', 1000))
        set_global_seed(seed)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')
        if hasattr(self.task, 'randomize_scene'):
            try:
                self.task.randomize_scene()
            except Exception:
                traceback.print_exc()
        self._apply_camera_settings_from_variables()
        self._apply_light_settings_from_variables()
        self._hide_visual_markers()

        snap = self.scene_snapshot
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = np.asarray(snap.qpos, dtype=float)
        self.mj_data.qvel[:] = np.asarray(snap.qvel, dtype=float)
        if len(snap.act) > 0 and self.mj_data.act is not None:
            self.mj_data.act[:] = np.asarray(snap.act, dtype=float)
        self.mj_data.ctrl[:] = np.asarray(snap.ctrl, dtype=float)
        if self.mj_model.nmocap > 0 and len(snap.mocap_pos) > 0:
            self.mj_data.mocap_pos[:] = np.asarray(snap.mocap_pos, dtype=float)
            self.mj_data.mocap_quat[:] = np.asarray(snap.mocap_quat, dtype=float)
        if self.mj_model.nuserdata > 0 and len(snap.userdata) > 0:
            self.mj_data.userdata[:] = np.asarray(snap.userdata, dtype=float)
        self.mj_data.time = float(snap.time)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')
        self.policy.reset()

    def get_policy_observation(self) -> Dict[str, Any]:
        obs = {
            'state': self.mj_data.sensordata[self.joint_pos_sensor_idx].copy().astype(np.float32),
            'task': self.args.task_description,
        }
        for cam_name, cfg in self.camera_cfgs.items():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
                continue
            self._set_renderer_size(cfg['width'], cfg['height'])
            obs[cam_name] = self._render_rgb(cam_name)
        return obs

    def _extract_action_array(self, action_out) -> np.ndarray:
        arr = action_out['action'] if isinstance(action_out, dict) and 'action' in action_out else action_out
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[0]
        return arr

    def _build_model_input(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        batch['observation.state'] = torch.from_numpy(obs_dict['state']).float()
        if 'global_cam' in obs_dict:
            img = torch.from_numpy(obs_dict['global_cam']).permute(2, 0, 1).contiguous().float() / 255.0
            batch['observation.images.global_cam'] = img
        if 'wrist_cam' in obs_dict:
            img = torch.from_numpy(obs_dict['wrist_cam']).permute(2, 0, 1).contiguous().float() / 255.0
            batch['observation.images.wrist_cam'] = img
        batch['task'] = obs_dict.get('task', self.args.task_description)
        return batch

    def _predict_action(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        model_input = self._build_model_input(obs_dict)
        batch = self.preprocess(model_input)
        with torch.inference_mode():
            pred = self.policy.select_action(batch)
        pred = self.postprocess(pred)
        return self._extract_action_array(pred)

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
        h = max(left.shape[0], right.shape[0])
        if left.shape[0] != h:
            left = cv2.resize(left, (left.shape[1], h), interpolation=cv2.INTER_AREA)
        if right.shape[0] != h:
            right = cv2.resize(right, (right.shape[1], h), interpolation=cv2.INTER_AREA)
        combined = np.concatenate([left, right], axis=1)
        cv2.putText(combined, 'GLOBAL', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, 'WRIST', (left.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return combined

    def run(self):
        self.restore_training_scene()
        video_writer = None
        combined_width = self.camera_cfgs['global_cam']['width'] + self.camera_cfgs['wrist_cam']['width']
        combined_height = max(self.camera_cfgs['global_cam']['height'], self.camera_cfgs['wrist_cam']['height'])
        if self.args.save_video:
            video_path = str(self.output_dir_path / 'policy_rollout_global_wrist_side_by_side.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.args.video_fps, (combined_width, combined_height))

        action_mse_list, action_l2_list, trajectory = [], [], []
        success = False
        success_first_step = None
        extra_after_success = 0

        for step in range(self.args.max_steps):
            obs = self.get_policy_observation()
            action = self._predict_action(obs)
            self.mj_data.ctrl[: self.mujoco_ctrl_dim] = np.asarray(action[: self.mujoco_ctrl_dim], dtype=float)
            for _ in range(self.args.decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)

            if self.reference is not None and step < len(self.reference.get('act', [])):
                gt_action = np.asarray(self.reference['act'][step], dtype=np.float32)
                common_dim = min(len(gt_action), len(action))
                diff = action[:common_dim] - gt_action[:common_dim]
                action_mse_list.append(float(np.mean(diff ** 2)))
                action_l2_list.append(float(np.linalg.norm(diff)))

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
                print(f'[INFO] success condition reached at step={step}. recording {extra_after_success} extra steps...')

            if success_first_step is not None:
                success = True
                if extra_after_success <= 0:
                    break
                extra_after_success -= 1

            if self.args.log_every > 0 and step % self.args.log_every == 0:
                print(f'[INFO] step={step} success={current_success} time={self.mj_data.time:.3f}')

        if video_writer is not None:
            video_writer.release()

        summary = {
            'checkpoint': str(self.checkpoint_path),
            'device': str(self.device),
            'robot': self.args.robot,
            'task': self.args.task,
            'task_description': self.args.task_description,
            'success': bool(success),
            'success_first_step': None if success_first_step is None else int(success_first_step),
            'recorded_extra_steps_after_success': int(self.args.success_hold_steps) if success_first_step is not None else 0,
            'num_steps': len(trajectory),
            'seed': int(self.variables.get('seed', 1000)),
            'scene_snapshot_path': str(self.scene_snapshot_path),
            'variables_path': str(self.variables_path),
            'reference_json': None if self.reference_json_path is None else str(self.reference_json_path),
            'avg_action_mse_vs_demo': None if len(action_mse_list) == 0 else float(np.mean(action_mse_list)),
            'avg_action_l2_vs_demo': None if len(action_l2_list) == 0 else float(np.mean(action_l2_list)),
        }
        save_json(self.output_dir_path / 'eval_summary.json', summary)
        save_json(self.output_dir_path / 'trajectory.json', {'trajectory': trajectory})
        print('\\n' + '=' * 70)
        print('✅ 테스트 완료')
        print(f"- success: {summary['success']}")
        print(f"- success_first_step: {summary['success_first_step']}")
        print(f"- recorded_extra_steps_after_success: {summary['recorded_extra_steps_after_success']}")
        print(f"- num_steps: {summary['num_steps']}")
        print(f"- avg_action_mse_vs_demo: {summary['avg_action_mse_vs_demo']}")
        print(f"- avg_action_l2_vs_demo: {summary['avg_action_l2_vs_demo']}")
        print(f"- output_dir: {self.output_dir_path}")
        print('=' * 70)


def build_argparser():
    parser = argparse.ArgumentParser(description='학습된 LeRobot Diffusion Policy를 Discoverse 고정 scene에서 테스트 (마커 제거 + 조용한 로그 + 성공 후 추가 녹화 + 좌우 영상)')
    parser.add_argument('--robot', type=str, default='piper')
    parser.add_argument('--task', type=str, default='place_block')
    parser.add_argument('--task-description', type=str, default='place the block')
    parser.add_argument('--checkpoint', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/outputs/train/piper_place_block_identical_fixed_diffusion/checkpoints/050000/pretrained_model'))
    parser.add_argument('--dataset-repo-id', type=str, default='apple/piper_place_block_identical_fixed')
    parser.add_argument('--dataset-root', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/lerobot_dataset_identical_fixed'))
    parser.add_argument('--scene-snapshot', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/data/piper_place_block_identical_fixed_data/sample_00/scene_snapshot.json'))
    parser.add_argument('--variables', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/data/piper_place_block_identical_fixed_data/sample_00/variables.json'))
    parser.add_argument('--reference-json', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/data/piper_place_block_identical_fixed_data/sample_00/obs_action.json'))
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/outputs/eval/piper_place_block_identical_fixed_diffusion_050000_sidebyside'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--decimation', type=int, default=5)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--quiet-success-check', action='store_true', default=True)
    parser.add_argument('--success-hold-steps', type=int, default=40)
    parser.add_argument('--log-every', type=int, default=20)
    return parser


def main():
    args = build_argparser().parse_args()
    FixedScenePolicyTester(args).run()


if __name__ == '__main__':
    main()
