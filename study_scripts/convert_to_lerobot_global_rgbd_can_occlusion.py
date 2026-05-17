#!/usr/bin/env python3
"""
Convert DISCOVERSE global RGBD occlusion data to LeRobotDataset format.

Expected source episode structure:
  sample_0000/
    cam_global_cam.mp4
    global_cam_depth.npz
    depth_vis_frames/depth_000000.png ...
    obs_action.json
    episode_metadata.json / scene_snapshot.json / variables.json optional

This converter stores RGBD as two image/video features:
  - observation.images.global_cam          : RGB global camera, 3 x 480 x 640
  - observation.images.global_depth_vis    : 3-channel depth visualization, 3 x 480 x 640

The metric depth npz files are copied to dataset_root/source_metadata/depth_npz_preview
for traceability, but are not inserted as float tensors into LeRobotDataset by default.
This keeps compatibility with standard LeRobot image pipelines that expect 3-channel images.
"""

import os
import json
import glob
import gc
import shutil
import argparse
from typing import Dict, List, Tuple, Optional

import cv2
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def natural_sample_key(path: str):
    name = os.path.basename(path.rstrip(os.sep))
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def find_episode_jsons(data_dir: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, "sample_*", "obs_action*.json"),
        os.path.join(data_dir, "**", "obs_action*.json"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    return sorted(set(found), key=lambda p: natural_sample_key(os.path.dirname(p)))


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_state_and_action_dims(example_json: Dict) -> Tuple[int, int, str]:
    if "act" not in example_json:
        raise KeyError(f"'act' key not found in episode json. keys={list(example_json.keys())}")
    if "obs" not in example_json:
        raise KeyError(f"'obs' key not found in episode json. keys={list(example_json.keys())}")

    actions = example_json["act"]
    if len(actions) == 0:
        raise ValueError("action sequence is empty")
    action_dim = len(actions[0]) if isinstance(actions[0], list) else 1

    obs_keys = list(example_json["obs"].keys())
    if len(obs_keys) == 0:
        raise ValueError("obs dict is empty")
    obs_key = obs_keys[0]

    states = example_json["obs"][obs_key]
    if len(states) == 0:
        raise ValueError("state sequence is empty")
    state_dim = len(states[0]) if isinstance(states[0], list) else 1

    return state_dim, action_dim, obs_key


def make_dataset(
    dataset_root: str,
    repo_id: str,
    fps: int,
    state_dim: int,
    action_dim: int,
    image_height: int,
    image_width: int,
    overwrite: bool = True,
) -> LeRobotDataset:
    if os.path.exists(dataset_root) and overwrite:
        print(f"🧹 기존 데이터셋 폴더를 삭제하고 새로 생성합니다: {dataset_root}")
        shutil.rmtree(dataset_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=fps,
        features={
            "observation.images.global_cam": {
                "dtype": "video",
                "shape": (3, image_height, image_width),
                "names": ["c", "h", "w"],
            },
            "observation.images.global_depth_vis": {
                "dtype": "video",
                "shape": (3, image_height, image_width),
                "names": ["c", "h", "w"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["dim"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["dim"],
            },
        },
    )


def frame_bgr_to_chw_rgb_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()


def image_path_to_chw_rgb_tensor(path: str):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read depth image: {path}")
    return frame_bgr_to_chw_rgb_tensor(img_bgr)


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, height, width, fps


def maybe_copy_source_metadata(source_data_dir: str, dataset_root: str, max_depth_npz_preview: int = 10):
    metadata_dir = os.path.join(dataset_root, "source_metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    copied_any = False
    for fname in ["dataset_manifest.json", "fixed_variables.json", "source_metadata.json"]:
        src = os.path.join(source_data_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(metadata_dir, fname))
            copied_any = True

    sample_dirs = sorted(glob.glob(os.path.join(source_data_dir, "sample_*")), key=natural_sample_key)
    preview_dir = os.path.join(metadata_dir, "sample_previews")
    depth_preview_dir = os.path.join(metadata_dir, "depth_npz_preview")
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(depth_preview_dir, exist_ok=True)

    for sample_dir in sample_dirs[:10]:
        sample_name = os.path.basename(sample_dir)
        for fname in ["variables.json", "scene_snapshot.json", "episode_metadata.json"]:
            src = os.path.join(sample_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(preview_dir, f"{sample_name}_{fname}"))
                copied_any = True

    for sample_dir in sample_dirs[:max_depth_npz_preview]:
        sample_name = os.path.basename(sample_dir)
        src = os.path.join(sample_dir, "global_cam_depth.npz")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(depth_preview_dir, f"{sample_name}_global_cam_depth.npz"))
            copied_any = True

    if copied_any:
        print(f"📝 원본 메타데이터 preview를 복사했습니다: {metadata_dir}")


def validate_episode_lengths(json_path: str) -> Tuple[int, int, int, int, int, str]:
    episode_dir = os.path.dirname(json_path)
    rgb_video_path = os.path.join(episode_dir, "cam_global_cam.mp4")
    depth_vis_paths = sorted(glob.glob(os.path.join(episode_dir, "depth_vis_frames", "depth_*.png")))

    if not os.path.exists(rgb_video_path):
        raise FileNotFoundError(f"Missing RGB video: {rgb_video_path}")
    if len(depth_vis_paths) == 0:
        raise FileNotFoundError(f"No depth visualization frames found in: {episode_dir}/depth_vis_frames")

    data = load_json(json_path)
    obs_key = list(data["obs"].keys())[0]
    act_len = len(data["act"])
    state_len = len(data["obs"][obs_key])
    rgb_len, height, width, _ = get_video_info(rgb_video_path)
    depth_len = len(depth_vis_paths)
    return act_len, state_len, rgb_len, depth_len, height, width, obs_key


def convert_episode(
    dataset: LeRobotDataset,
    json_path: str,
    obs_key: str,
    task_name: str,
    strict_lengths: bool = True,
) -> int:
    episode_dir = os.path.dirname(json_path)
    rgb_video_path = os.path.join(episode_dir, "cam_global_cam.mp4")
    depth_vis_paths = sorted(glob.glob(os.path.join(episode_dir, "depth_vis_frames", "depth_*.png")))

    if not os.path.exists(rgb_video_path):
        print(f"⚠️ RGB 비디오가 누락되어 건너뜁니다: {episode_dir}")
        return 0
    if len(depth_vis_paths) == 0:
        print(f"⚠️ depth_vis_frames가 누락되어 건너뜁니다: {episode_dir}")
        return 0

    data = load_json(json_path)
    actions = torch.tensor(data["act"], dtype=torch.float32)
    states = torch.tensor(data["obs"][obs_key], dtype=torch.float32)

    cap_rgb = cv2.VideoCapture(rgb_video_path)
    total_frames_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_depth = len(depth_vis_paths)

    lengths = [total_frames_rgb, total_frames_depth, len(actions), len(states)]
    min_len = min(lengths)
    if strict_lengths and len(set(lengths)) != 1:
        print(f"⚠️ 길이 불일치 episode: {episode_dir}, lengths(rgb,depth,act,state)={lengths}; min_len={min_len}")

    step = 0
    last_rgb = None
    last_depth = None

    while cap_rgb.isOpened() and step < min_len:
        ret_rgb, frame_rgb_bgr = cap_rgb.read()
        if not ret_rgb:
            break

        last_rgb = frame_bgr_to_chw_rgb_tensor(frame_rgb_bgr)
        last_depth = image_path_to_chw_rgb_tensor(depth_vis_paths[step])

        dataset.add_frame(
            {
                "observation.images.global_cam": last_rgb,
                "observation.images.global_depth_vis": last_depth,
                "observation.state": states[step],
                "action": actions[step],
                "task": task_name,
            }
        )
        step += 1

    cap_rgb.release()

    if step > 0:
        dataset.save_episode()
        print(f"✅ 변환 완료: {episode_dir} | {step} 프레임")
    else:
        print(f"⚠️ 프레임이 0개라서 저장하지 않습니다: {episode_dir}")

    del actions, states
    if last_rgb is not None:
        del last_rgb
    if last_depth is not None:
        del last_depth
    gc.collect()
    return step


def main(
    data_dir: str,
    dataset_root: str,
    repo_id: str,
    fps: int = 30,
    task_name: str = "place the block under occlusion using global RGBD",
    overwrite: bool = True,
    strict_lengths: bool = True,
    max_episodes: Optional[int] = None,
):
    json_files = find_episode_jsons(data_dir)
    if max_episodes is not None and max_episodes > 0:
        json_files = json_files[:max_episodes]

    num_episodes = len(json_files)
    if num_episodes == 0:
        print(f"❌ 데이터를 찾을 수 없습니다. 경로를 확인해주세요: {data_dir}")
        return

    print(f"📦 총 {num_episodes}개의 에피소드 변환을 시작합니다...")

    example_json = load_json(json_files[0])
    state_dim, action_dim, obs_key = infer_state_and_action_dims(example_json)
    act_len, state_len, rgb_len, depth_len, height, width, obs_key2 = validate_episode_lengths(json_files[0])
    if obs_key != obs_key2:
        print(f"⚠️ obs_key mismatch: inferred={obs_key}, validated={obs_key2}. Using inferred={obs_key}")

    print(f"🔎 상태 차원={state_dim}, 액션 차원={action_dim}, observation key='{obs_key}'")
    print(f"🔎 첫 episode 길이 act/state/rgb/depth={act_len}/{state_len}/{rgb_len}/{depth_len}")
    print(f"🔎 이미지 크기 HxW={height}x{width}, fps={fps}")

    dataset = make_dataset(
        dataset_root=dataset_root,
        repo_id=repo_id,
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
        image_height=height,
        image_width=width,
        overwrite=overwrite,
    )

    maybe_copy_source_metadata(data_dir, dataset_root)

    converted_episodes = 0
    converted_frames = 0
    skipped = 0

    for i, json_path in enumerate(json_files, start=1):
        print(f"\n[{i}/{num_episodes}] 변환 중: {json_path}")
        try:
            n_frames = convert_episode(
                dataset,
                json_path,
                obs_key=obs_key,
                task_name=task_name,
                strict_lengths=strict_lengths,
            )
        except Exception as e:
            print(f"❌ 변환 실패: {json_path} | {repr(e)}")
            n_frames = 0

        if n_frames > 0:
            converted_episodes += 1
            converted_frames += n_frames
        else:
            skipped += 1

    print("\n⏳ 데이터셋 메타데이터와 압축을 최종 마무리(Finalize) 중입니다...")
    dataset.finalize()

    print("🎉 LeRobot 학습용 RGBD 포맷 변환 완료!")
    print(f" - 저장된 에피소드 수: {converted_episodes}")
    print(f" - 건너뛴 에피소드 수: {skipped}")
    print(f" - 저장된 총 프레임 수: {converted_frames}")
    print(f" - 출력 경로: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DISCOVERSE global RGBD can-occlusion 데이터를 LeRobot 포맷으로 변환"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/lerobot/DISCOVERSE/data/piper_place_block_global_rgbd_can_occlusion_gt_data"),
        help="원본 DISCOVERSE RGBD occlusion 데이터 루트 경로",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.expanduser("~/lerobot/DISCOVERSE/lerobot_dataset_global_rgbd_can_occlusion_gt"),
        help="출력 LeRobot 데이터셋 경로",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="apple/piper_place_block_global_rgbd_can_occlusion_gt",
        help="LeRobot dataset repo_id",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-name", type=str, default="place the block under occlusion using global RGBD")
    parser.add_argument("--no-overwrite", action="store_true", help="기존 dataset_root가 있어도 삭제하지 않음")
    parser.add_argument("--no-strict-lengths", action="store_true", help="길이 불일치 경고를 완화하고 min_len으로 변환")
    parser.add_argument("--max-episodes", type=int, default=None, help="테스트용으로 변환할 최대 episode 수")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        fps=args.fps,
        task_name=args.task_name,
        overwrite=not args.no_overwrite,
        strict_lengths=not args.no_strict_lengths,
        max_episodes=args.max_episodes,
    )
