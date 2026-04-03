import os
import json
import glob
import gc
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_episode_jsons(data_dir: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, "**", "obs_action*.json"),
        os.path.join(data_dir, "sample_*", "obs_action*.json"),
        os.path.join(data_dir, "domain_*", "episode_*", "obs_action*.json"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    return sorted(set(found))


def find_video(episode_dir: str, camera_name: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(episode_dir, f"*{camera_name}*.mp4")))
    return candidates[0] if candidates else None


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
) -> LeRobotDataset:
    if os.path.exists(dataset_root):
        print(f"🧹 기존 데이터셋 폴더를 삭제하고 새로 생성합니다: {dataset_root}")
        shutil.rmtree(dataset_root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=fps,
        features={
            "observation.images.global_cam": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
            "observation.images.wrist_cam": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["dim"]},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": ["dim"]},
        },
    )


def frame_to_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()


def maybe_copy_training_metadata(source_data_dir: str, dataset_root: str):
    manifest_candidates = [
        os.path.join(source_data_dir, "dataset_manifest.json"),
        os.path.join(source_data_dir, "fixed_variables.json"),
    ]
    metadata_dir = os.path.join(dataset_root, "source_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    copied_any = False
    for src in manifest_candidates:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(metadata_dir, os.path.basename(src)))
            copied_any = True
    if copied_any:
        print(f"📝 원본 변수/매니페스트 정보를 복사했습니다: {metadata_dir}")


def convert_episode(
    dataset: LeRobotDataset,
    json_path: str,
    obs_key: str,
    task_name: str,
) -> int:
    episode_dir = os.path.dirname(json_path)
    global_video_path = find_video(episode_dir, "global_cam")
    wrist_video_path = find_video(episode_dir, "wrist_cam")

    if global_video_path is None or wrist_video_path is None:
        print(f"⚠️ 비디오가 누락되어 건너뜁니다: {episode_dir}")
        return 0

    data = load_json(json_path)
    actions = torch.tensor(data["act"], dtype=torch.float32)
    states = torch.tensor(data["obs"][obs_key], dtype=torch.float32)

    cap_global = cv2.VideoCapture(global_video_path)
    cap_wrist = cv2.VideoCapture(wrist_video_path)

    total_frames_global = int(cap_global.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_wrist = int(cap_wrist.get(cv2.CAP_PROP_FRAME_COUNT))
    min_len = min(total_frames_global, total_frames_wrist, len(actions), len(states))

    step = 0
    last_frame_tensor_g = None
    last_frame_tensor_w = None
    while cap_global.isOpened() and cap_wrist.isOpened() and step < min_len:
        ret_g, frame_g = cap_global.read()
        ret_w, frame_w = cap_wrist.read()
        if not ret_g or not ret_w:
            break

        last_frame_tensor_g = frame_to_tensor(frame_g)
        last_frame_tensor_w = frame_to_tensor(frame_w)

        dataset.add_frame(
            {
                "observation.images.global_cam": last_frame_tensor_g,
                "observation.images.wrist_cam": last_frame_tensor_w,
                "observation.state": states[step],
                "action": actions[step],
                "task": task_name,
            }
        )
        step += 1

    cap_global.release()
    cap_wrist.release()

    if step > 0:
        dataset.save_episode()
        print(f"✅ 변환 완료: {episode_dir} | {step} 프레임")
    else:
        print(f"⚠️ 프레임이 0개라서 저장하지 않습니다: {episode_dir}")

    del actions, states
    if last_frame_tensor_g is not None:
        del last_frame_tensor_g
    if last_frame_tensor_w is not None:
        del last_frame_tensor_w
    gc.collect()
    return step


def main():
    # 기본 경로를 '완전히 동일한 5개 샘플' 데이터셋으로 변경
    data_dir = os.path.expanduser("~/lerobot/DISCOVERSE/data/piper_place_block_identical_fixed_data")
    dataset_root = os.path.expanduser("~/lerobot/DISCOVERSE/lerobot_dataset_identical_fixed")
    repo_id = "apple/piper_place_block_identical_fixed"
    fps = 30
    task_name = "place the block"

    json_files = find_episode_jsons(data_dir)
    num_episodes = len(json_files)
    if num_episodes == 0:
        print(f"❌ 데이터를 찾을 수 없습니다. 경로를 확인해주세요: {data_dir}")
        return

    print(f"📦 총 {num_episodes}개의 에피소드 변환을 시작합니다...")

    example_json = load_json(json_files[0])
    state_dim, action_dim, obs_key = infer_state_and_action_dims(example_json)
    print(f"🔎 상태 차원={state_dim}, 액션 차원={action_dim}, 사용 observation key='{obs_key}'")

    dataset = make_dataset(
        dataset_root=dataset_root,
        repo_id=repo_id,
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    maybe_copy_training_metadata(data_dir, dataset_root)

    converted_episodes = 0
    converted_frames = 0
    for i, json_path in enumerate(json_files, start=1):
        print(f"\n[{i}/{num_episodes}] 변환 중: {json_path}")
        n_frames = convert_episode(dataset, json_path, obs_key=obs_key, task_name=task_name)
        if n_frames > 0:
            converted_episodes += 1
            converted_frames += n_frames

    print("⏳ 데이터셋 메타데이터와 압축을 최종 마무리(Finalize) 중입니다...")
    dataset.finalize()
    print("🎉 LeRobot 학습용 포맷 변환 완료!")
    print(f"   - 저장된 에피소드 수: {converted_episodes}")
    print(f"   - 저장된 총 프레임 수: {converted_frames}")
    print(f"   - 출력 경로: {dataset_root}")


if __name__ == "__main__":
    main()
