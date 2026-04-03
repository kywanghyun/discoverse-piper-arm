import os
import io
import json
import random
import argparse
import traceback
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class UnseenPosePolicyEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        self.dataset_root_path = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
        self.output_dir_path = Path(args.output_dir).expanduser().resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'checkpoint 경로를 찾을 수 없습니다: {self.checkpoint_path}')
        if self.dataset_root_path is not None and not self.dataset_root_path.exists():
            raise FileNotFoundError(f'dataset_root 경로를 찾을 수 없습니다: {self.dataset_root_path}')

        self.xml_path = generate_robot_task_model(args.robot, args.task)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model)
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

        self.base_scene = None
        self.base_object_pose = None
        self.base_bowl_pose = None

        os.makedirs(self.output_dir_path, exist_ok=True)
        os.makedirs(self.output_dir_path / 'videos', exist_ok=True)

        self._capture_base_scene_once()

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

    def _capture_state(self) -> Dict[str, Any]:
        return {
            'qpos': self.mj_data.qpos.copy(),
            'qvel': self.mj_data.qvel.copy(),
            'act': None if self.mj_data.act is None else self.mj_data.act.copy(),
            'ctrl': self.mj_data.ctrl.copy(),
            'mocap_pos': None if self.mj_model.nmocap <= 0 else self.mj_data.mocap_pos.copy(),
            'mocap_quat': None if self.mj_model.nmocap <= 0 else self.mj_data.mocap_quat.copy(),
            'userdata': None if self.mj_model.nuserdata <= 0 else self.mj_data.userdata.copy(),
            'time': float(self.mj_data.time),
        }

    def _restore_state(self, state: Dict[str, Any]):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = state['qpos']
        self.mj_data.qvel[:] = state['qvel']
        if state['act'] is not None and self.mj_data.act is not None:
            self.mj_data.act[:] = state['act']
        self.mj_data.ctrl[:] = state['ctrl']
        if state['mocap_pos'] is not None and self.mj_model.nmocap > 0:
            self.mj_data.mocap_pos[:] = state['mocap_pos']
            self.mj_data.mocap_quat[:] = state['mocap_quat']
        if state['userdata'] is not None and self.mj_model.nuserdata > 0:
            self.mj_data.userdata[:] = state['userdata']
        self.mj_data.time = state['time']
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')

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
            'pos': q[:3].copy(),
            'quat': q[3:7].copy(),
        }

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray):
        qadr = self._get_free_joint_qpos_adr(body_name)
        self.mj_data.qpos[qadr:qadr + 3] = pos
        self.mj_data.qpos[qadr + 3:qadr + 7] = quat

    def _capture_base_scene_once(self):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')
        self.base_scene = self._capture_state()
        self.base_object_pose = self._get_free_body_pose(self.args.object_name)
        self.base_bowl_pose = self._get_free_body_pose(self.args.bowl_name)

    def restore_base_scene(self):
        if self.base_scene is None:
            raise RuntimeError('Base scene is not initialized')
        self._restore_state(self.base_scene)
        self._hide_visual_markers()
        self.policy.reset()

    def randomize_object_and_bowl_positions(self, seed: int) -> Dict[str, Any]:
        if self.base_object_pose is None or self.base_bowl_pose is None:
            raise RuntimeError('Base object/bowl poses are not initialized')

        set_global_seed(seed)
        base_obj_pos = self.base_object_pose['pos']
        base_obj_quat = self.base_object_pose['quat']
        base_bowl_pos = self.base_bowl_pose['pos']
        base_bowl_quat = self.base_bowl_pose['quat']

        obj_z = float(base_obj_pos[2])
        bowl_z = float(base_bowl_pos[2])

        sampled_obj = None
        sampled_bowl = None
        for _ in range(self.args.max_sample_trials):
            obj_xy = base_obj_pos[:2] + np.random.uniform(-self.args.object_xy_range, self.args.object_xy_range, size=2)
            bowl_xy = base_bowl_pos[:2] + np.random.uniform(-self.args.bowl_xy_range, self.args.bowl_xy_range, size=2)
            if np.linalg.norm(obj_xy - bowl_xy) < self.args.min_object_bowl_dist:
                continue
            sampled_obj = np.array([obj_xy[0], obj_xy[1], obj_z], dtype=float)
            sampled_bowl = np.array([bowl_xy[0], bowl_xy[1], bowl_z], dtype=float)
            break

        if sampled_obj is None or sampled_bowl is None:
            raise RuntimeError('Failed to sample unseen object/bowl positions')

        self._set_free_body_pose(self.args.object_name, sampled_obj, base_obj_quat)
        self._set_free_body_pose(self.args.bowl_name, sampled_bowl, base_bowl_quat)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, 'target', 'endpoint', 'site')

        return {
            'seed': int(seed),
            'object_name': self.args.object_name,
            'bowl_name': self.args.bowl_name,
            'object_pose': {'pos': sampled_obj.tolist(), 'quat': base_obj_quat.tolist()},
            'bowl_pose': {'pos': sampled_bowl.tolist(), 'quat': base_bowl_quat.tolist()},
            'object_xy_range': float(self.args.object_xy_range),
            'bowl_xy_range': float(self.args.bowl_xy_range),
            'min_object_bowl_dist': float(self.args.min_object_bowl_dist),
            'note': 'This pose is generated at evaluation time and is intended to be unseen during training.',
        }

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
        combined = np.concatenate([left, right], axis=1)
        cv2.putText(combined, 'GLOBAL', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, 'WRIST', (left.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return combined

    def evaluate_one_episode(self, episode_idx: int, seed: int) -> Dict[str, Any]:
        self.restore_base_scene()
        sampled_pose_info = self.randomize_object_and_bowl_positions(seed)

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
        }

    def run(self):
        all_results: List[Dict[str, Any]] = []
        for episode_idx in range(self.args.num_eval_episodes):
            seed = int(self.args.eval_seed_start + episode_idx)
            print('\n' + '=' * 80)
            print(f'[EVAL {episode_idx + 1}/{self.args.num_eval_episodes}] unseen pose seed={seed}')
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
            'note': 'All evaluated object/bowl poses are generated at evaluation time using seeds intended to be outside the training seed range.',
        }

        save_json(self.output_dir_path / 'aggregate_eval_summary.json', aggregate)
        save_json(self.output_dir_path / 'per_episode_results.json', {'results': all_results})

        print('\n' + '=' * 80)
        print('✅ Unseen-pose evaluation completed')
        print(f"- num_eval_episodes_valid: {aggregate['num_eval_episodes_valid']}")
        print(f"- num_eval_episodes_success: {aggregate['num_eval_episodes_success']}")
        print(f"- success_rate: {aggregate['success_rate']}")
        print(f"- avg_success_first_step: {aggregate['avg_success_first_step']}")
        print(f"- avg_num_steps: {aggregate['avg_num_steps']}")
        print(f"- output_dir: {self.output_dir_path}")
        print('=' * 80)


def build_argparser():
    parser = argparse.ArgumentParser(description='학습에 없던 새로운 object/bowl 위치를 생성해 LeRobot Diffusion Policy를 평가')
    parser.add_argument('--robot', type=str, default='piper')
    parser.add_argument('--task', type=str, default='place_block')
    parser.add_argument('--task-description', type=str, default='place the block')
    parser.add_argument('--checkpoint', type=str, required=True, help='평가할 pretrained_model checkpoint 경로')
    parser.add_argument('--dataset-repo-id', type=str, default='apple/piper_place_block_pose_randomized_fixed_visual')
    parser.add_argument('--dataset-root', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/lerobot_dataset_pose_randomized_fixed_visual'))
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~/lerobot/DISCOVERSE/outputs/eval/piper_place_block_pose_randomized_fixed_visual_unseen_pose_eval'))
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
    return parser


def main():
    args = build_argparser().parse_args()
    UnseenPosePolicyEvaluator(args).run()


if __name__ == '__main__':
    main()
