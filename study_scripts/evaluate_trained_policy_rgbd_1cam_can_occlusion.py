#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGBD 1-camera LeRobot DiffusionPolicy evaluator with optional can occlusion.

Outputs:
- aggregate_eval_summary.json
- per_episode_results.json
- optional RGB+depth videos
"""

import os, io, json, math, random, argparse, traceback, contextlib, re, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def to_jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (list, tuple)): return [to_jsonable(v) for v in x]
    if isinstance(x, dict): return {str(k): to_jsonable(v) for k, v in x.items()}
    return x


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)




def _try_float(v):
    try:
        return float(v)
    except Exception:
        return None


def parse_lerobot_train_log(log_path: str):
    rows = []
    num_pat = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    step_alt_pat = re.compile(r"\b(?:step|steps|global_step)\D+(\d+)\b", re.IGNORECASE)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pairs = {}
            for k, v in num_pat.findall(line):
                val = _try_float(v)
                if val is not None:
                    pairs[k.strip("'\"")] = val
            if "step" not in pairs and "steps" not in pairs and "global_step" not in pairs:
                m = step_alt_pat.search(line)
                if m:
                    pairs["step"] = float(m.group(1))
            metric_like = any(any(tok in k.lower() for tok in ["loss", "lr", "grad", "epoch", "step", "eta"]) for k in pairs)
            if pairs and metric_like:
                pairs["row_idx"] = len(rows)
                rows.append(pairs)
    return rows


def save_train_log_plots(log_path: str, output_dir: str):
    import matplotlib.pyplot as plt
    rows = parse_lerobot_train_log(log_path)
    plot_dir = Path(output_dir).expanduser().resolve() / "train_log_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = {"log_path": str(Path(log_path).expanduser().resolve()), "plot_dir": str(plot_dir), "num_metric_rows": len(rows), "csv_path": None, "plots": [], "metric_keys": []}
    if not rows:
        save_json(plot_dir / "train_log_plot_summary.json", summary)
        print(f"[WARN] 학습 로그에서 numeric metric row를 찾지 못했습니다: {log_path}")
        return summary
    keys = sorted(set().union(*(r.keys() for r in rows)))
    csv_path = plot_dir / "parsed_train_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    summary["csv_path"] = str(csv_path)
    x_key = next((k for k in ["step", "steps", "global_step", "train/step"] if k in keys), "row_idx")
    metric_keys = [k for k in keys if k not in {x_key, "row_idx"}]
    preferred = [k for k in metric_keys if any(tok in k.lower() for tok in ["loss", "lr", "grad", "reward", "success"])]
    metric_keys = preferred if preferred else metric_keys
    summary["metric_keys"] = metric_keys
    xs = np.asarray([r.get(x_key, r.get("row_idx", i)) for i, r in enumerate(rows)], dtype=float)
    for metric in metric_keys:
        ys = np.asarray([r.get(metric, np.nan) for r in rows], dtype=float)
        if np.all(np.isnan(ys)):
            continue
        fig = plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(x_key); plt.ylabel(metric); plt.title(f"{metric} vs {x_key}"); plt.grid(True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
        png_path = plot_dir / f"{safe}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        summary["plots"].append(str(png_path))
    save_json(plot_dir / "train_log_plot_summary.json", summary)
    print(f"[INFO] 학습 로그 그래프 저장 완료: {plot_dir}")
    return summary

def generate_robot_task_model(robot_name: str, task_name: str,
                              global_cam_pos="0 0.4 1.2",
                              global_cam_xyaxes="1 0 0 0 0.707 0.707") -> str:
    xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/tmp",
                            f"{robot_name}_{task_name}_rgbd_1cam_can_occlusion_eval.xml")
    make_env(robot_name, task_name, xml_path)
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    if 'name="global_cam"' not in xml:
        cam_xml = f"""
        <camera name="global_cam" pos="{global_cam_pos}" xyaxes="{global_cam_xyaxes}"/>
        """
        if "</worldbody>" not in xml: raise RuntimeError("MJCF에 </worldbody> 태그가 없습니다.")
        xml = xml.replace("</worldbody>", cam_xml + "\n</worldbody>", 1)

    if 'name="occluder_0"' not in xml:
        coke_mesh = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes/object/coke/coke.obj")
        coke_tex = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes/object/coke/coke.png")
        asset_xml = f"""
        <mesh name="occluder_coke_mesh" file="{coke_mesh}"/>
        <texture name="occluder_coke_tex" type="2d" file="{coke_tex}"/>
        <material name="occluder_coke_mat" texture="occluder_coke_tex" rgba="1 1 1 1"/>
        """
        if 'name="occluder_coke_mesh"' not in xml:
            if "</asset>" in xml:
                xml = xml.replace("</asset>", asset_xml + "\n</asset>", 1)
            else:
                # XML declaration(<?xml ...?>)이 있는 경우 xml.find(">")는 <mujoco>가 아니라
                # XML declaration 끝을 가리킵니다. 그 위치에 <asset>을 넣으면 <asset>이
                # 문서 root처럼 해석되어 MuJoCo가 "Unrecognized XML model type: 'asset'"를 냅니다.
                m = re.search(r"<mujoco\b[^>]*>", xml)
                if m is None:
                    raise RuntimeError("MJCF <mujoco ...> root tag를 찾을 수 없습니다.")
                root_end = m.end()
                xml = xml[:root_end] + "\n<asset>" + asset_xml + "\n</asset>" + xml[root_end:]
        body_xml = """
        <body name="occluder_0" pos="10 10 10" quat="1 0 0 0">
            <geom name="occluder_0_geom" type="mesh" mesh="occluder_coke_mesh"
                  material="occluder_coke_mat" contype="0" conaffinity="0" rgba="1 1 1 0"/>
        </body>
        """
        if "</worldbody>" not in xml: raise RuntimeError("MJCF에 </worldbody> 태그가 없습니다.")
        xml = xml.replace("</worldbody>", body_xml + "\n</worldbody>", 1)

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return xml_path


class RGBDOneCamCanOcclusionEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        self.dataset_root_path = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        if not self.checkpoint_path.exists(): raise FileNotFoundError(self.checkpoint_path)
        if self.dataset_root_path and not self.dataset_root_path.exists(): raise FileNotFoundError(self.dataset_root_path)

        self.xml_path = generate_robot_task_model(args.robot, args.task, args.global_cam_pos, args.global_cam_xyaxes)
        print(f"[INFO] Generated MJCF: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        cfg_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
        self.task = UniversalTaskBase(
            robot_config_path=os.path.join(cfg_root, "robots", f"{args.robot}.yaml"),
            task_config_path=os.path.join(cfg_root, "tasks", f"{args.task}.yaml"),
            mj_model=self.model, mj_data=self.data)
        self.task.randomizer.set_viewer(None); self.task.randomizer.set_renderer(self.renderer)
        self.nu = self.model.nu
        self.joint_pos_sensor_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, s)
                                     for s in self.task.robot_interface.joint_pos_sensors]
        if any(i < 0 for i in self.joint_pos_sensor_idx): raise RuntimeError(self.joint_pos_sensor_idx)
        self.hide_markers()

        self.policy = DiffusionPolicy.from_pretrained(self.checkpoint_path, local_files_only=True).to(self.device).eval()
        self.meta = self.load_meta(args.dataset_repo_id, self.dataset_root_path)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config, pretrained_path=self.checkpoint_path, dataset_stats=self.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.obs_keys = [k for k in self.meta.features.keys() if k.startswith("observation.")]
        print("[INFO] observation keys:")
        for k in self.obs_keys: print(f"  - {k}: {self.meta.features.get(k)}")
        self.rgb_key, self.depth_key, self.rgbd_key = self.resolve_rgbd_keys()
        print(f"[INFO] RGBD mapping: rgb_key={self.rgb_key}, depth_key={self.depth_key}, rgbd_key={self.rgbd_key}")

        self.base_state = None; self.base_obj = None; self.base_bowl = None; self.base_cam = {}; self.base_occ = None
        self.output_dir.mkdir(parents=True, exist_ok=True); (self.output_dir / "videos").mkdir(exist_ok=True)
        self.capture_base_scene()

    def load_meta(self, repo_id, root):
        try:
            return LeRobotDatasetMetadata(repo_id=repo_id, root=root) if root else LeRobotDatasetMetadata(repo_id=repo_id)
        except TypeError:
            return LeRobotDatasetMetadata(repo_id, root=root) if root else LeRobotDatasetMetadata(repo_id)

    def feature_shape(self, key):
        feat = self.meta.features.get(key)
        if feat is None: return None
        shape = feat.get("shape") if isinstance(feat, dict) else getattr(feat, "shape", None)
        try: return tuple(int(v) for v in shape) if shape is not None else None
        except Exception: return None

    def resolve_rgbd_keys(self):
        if self.args.rgbd_concat_key: return None, None, self.args.rgbd_concat_key
        if self.args.rgb_key or self.args.depth_key: return self.args.rgb_key, self.args.depth_key, None
        cam = self.args.camera_name; keys = list(self.meta.features.keys())
        concat_candidates = [f"observation.images.{cam}", f"observation.rgbd.{cam}", f"observation.images.{cam}.rgbd"]
        for k in concat_candidates:
            if k in keys:
                s = self.feature_shape(k)
                if s and len(s) >= 3 and (s[0] == 4 or s[-1] == 4): return None, None, k
        rgb_candidates = [f"observation.images.{cam}", f"observation.image.{cam}", f"observation.images.{cam}.rgb"]
        depth_candidates = [f"observation.depth.{cam}", f"observation.depths.{cam}",
                            f"observation.images.{cam}.depth", f"observation.image.{cam}.depth",
                            f"observation.{cam}.depth"]
        rgb = next((k for k in rgb_candidates if k in keys), None)
        dep = next((k for k in depth_candidates if k in keys), None)
        if rgb is None:
            xs = [k for k in keys if k.startswith("observation.images.") and cam in k and "depth" not in k.lower()]
            rgb = xs[0] if xs else None
        if dep is None:
            xs = [k for k in keys if k.startswith("observation.") and cam in k and "depth" in k.lower()]
            dep = xs[0] if xs else None
        if self.args.rgbd_mode == "concat" and rgb is not None: return None, None, rgb
        if self.args.rgbd_mode == "rgb_only": return rgb, None, None
        return rgb, dep, None

    def hide_markers(self):
        for i in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[i])
            if self.model.body_mocapid[bid] != -1: self.model.geom_rgba[i][3] = 0
        for i in range(self.model.nsite): self.model.site_rgba[i][3] = 0
        for name in ["target_box","target_x","target_y","target_z","origin_x","origin_y","origin_z","axis_x","axis_y","axis_z"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0: self.model.geom_rgba[gid][3] = 0

    def set_renderer_size(self):
        self.renderer._width = self.args.camera_width; self.renderer._height = self.args.camera_height
        self.renderer._rect.width = self.args.camera_width; self.renderer._rect.height = self.args.camera_height

    def scene_opt(self):
        opt = mujoco.MjvOption()
        for i in range(len(opt.sitegroup)): opt.sitegroup[i] = 0
        return opt

    def render_rgb(self):
        self.renderer.update_scene(self.data, self.args.camera_name, scene_option=self.scene_opt())
        return self.renderer.render().copy()

    def render_depth(self):
        self.renderer.update_scene(self.data, self.args.camera_name, scene_option=self.scene_opt())
        self.renderer.enable_depth_rendering(); d = self.renderer.render().copy().astype(np.float32); self.renderer.disable_depth_rendering()
        return d

    def depth_tensor(self, depth):
        d = depth.astype(np.float32); d[~np.isfinite(d)] = self.args.depth_far_clip
        if self.args.depth_normalization == "clip01":
            d = np.clip(d, self.args.depth_near_clip, self.args.depth_far_clip)
            d = (d - self.args.depth_near_clip) / (self.args.depth_far_clip - self.args.depth_near_clip + 1e-6)
        return torch.from_numpy(d).unsqueeze(0).contiguous().float()

    def depth_vis(self, depth):
        d = depth.copy(); d[~np.isfinite(d)] = self.args.depth_far_clip
        d = np.clip(d, self.args.depth_near_clip, self.args.depth_far_clip)
        x = (d - self.args.depth_near_clip) / (self.args.depth_far_clip - self.args.depth_near_clip + 1e-6)
        u = (255 * (1 - x)).astype(np.uint8)
        return np.stack([u,u,u], -1)

    def capture_state(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                    act=None if self.data.act is None else self.data.act.copy(), ctrl=self.data.ctrl.copy(),
                    mocap_pos=None if self.model.nmocap <= 0 else self.data.mocap_pos.copy(),
                    mocap_quat=None if self.model.nmocap <= 0 else self.data.mocap_quat.copy(),
                    userdata=None if self.model.nuserdata <= 0 else self.data.userdata.copy(), time=float(self.data.time))

    def restore_state(self, s):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = s["qpos"]; self.data.qvel[:] = s["qvel"]; self.data.ctrl[:] = s["ctrl"]
        if s["act"] is not None and self.data.act is not None: self.data.act[:] = s["act"]
        if s["mocap_pos"] is not None and self.model.nmocap > 0:
            self.data.mocap_pos[:] = s["mocap_pos"]; self.data.mocap_quat[:] = s["mocap_quat"]
        if s["userdata"] is not None and self.model.nuserdata > 0: self.data.userdata[:] = s["userdata"]
        self.data.time = s["time"]; mujoco.mj_forward(self.model, self.data)
        mink.move_mocap_to_frame(self.model, self.data, "target", "endpoint", "site")

    def free_qadr(self, body):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0: raise ValueError(f"Body not found: {body}")
        jid = int(self.model.body_jntadr[bid])
        if jid < 0 or int(self.model.jnt_type[jid]) != mujoco.mjtJoint.mjJNT_FREE: raise ValueError(f"No free joint: {body}")
        return int(self.model.jnt_qposadr[jid])

    def get_body_pose(self, body):
        qadr = self.free_qadr(body); q = self.data.qpos[qadr:qadr+7].copy()
        return dict(pos=q[:3].copy(), quat=q[3:7].copy(), qpos_adr=qadr, body_name=body)

    def set_body_pose(self, body, pos, quat):
        qadr = self.free_qadr(body); self.data.qpos[qadr:qadr+3] = pos; self.data.qpos[qadr+3:qadr+7] = quat

    def occ_ids(self):
        return (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.args.occluder_name),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.args.occluder_geom_name))

    def capture_base_scene(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(0).id); mujoco.mj_forward(self.model, self.data)
        mink.move_mocap_to_frame(self.model, self.data, "target", "endpoint", "site"); self.hide_markers()
        self.base_state = self.capture_state(); self.base_obj = self.get_body_pose(self.args.object_name); self.base_bowl = self.get_body_pose(self.args.bowl_name)
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.args.camera_name)
        if camid >= 0: self.base_cam = dict(pos=self.model.cam_pos[camid].copy(), quat=self.model.cam_quat[camid].copy())
        bid, gid = self.occ_ids()
        if bid >= 0 and gid >= 0:
            self.base_occ = dict(body_pos=self.model.body_pos[bid].copy(), body_quat=self.model.body_quat[bid].copy(), rgba=self.model.geom_rgba[gid].copy())

    def restore_base_scene(self):
        self.restore_state(self.base_state)
        camid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.args.camera_name)
        if camid >= 0 and self.base_cam:
            self.model.cam_pos[camid] = self.base_cam["pos"]; self.model.cam_quat[camid] = self.base_cam["quat"]
        bid, gid = self.occ_ids()
        if bid >= 0 and gid >= 0 and self.base_occ:
            self.model.body_pos[bid] = self.base_occ["body_pos"]; self.model.body_quat[bid] = self.base_occ["body_quat"]; self.model.geom_rgba[gid] = self.base_occ["rgba"]
        mujoco.mj_forward(self.model, self.data); self.hide_markers(); self.policy.reset()

    def randomize_visuals(self):
        info = {}
        if self.args.randomize_visuals and hasattr(self.task, "randomize_scene"):
            try: self.task.randomize_scene()
            except Exception: traceback.print_exc()
        mujoco.mj_forward(self.model, self.data)
        info.update(texture_randomized=bool(self.args.randomize_visuals), light_randomized=bool(self.args.randomize_visuals), camera_pose_randomized=False)
        return info

    def randomize_object_bowl(self, seed):
        set_seed(seed); op = self.base_obj["pos"]; oq = self.base_obj["quat"]; bp = self.base_bowl["pos"]; bq = self.base_bowl["quat"]
        so = sb = None
        for _ in range(self.args.max_sample_trials):
            oxy = op[:2] + np.random.uniform(-self.args.object_xy_range, self.args.object_xy_range, 2)
            bxy = bp[:2] + np.random.uniform(-self.args.bowl_xy_range, self.args.bowl_xy_range, 2)
            if np.linalg.norm(oxy-bxy) < self.args.min_object_bowl_dist: continue
            so = np.array([oxy[0], oxy[1], op[2]], float); sb = np.array([bxy[0], bxy[1], bp[2]], float); break
        if so is None: raise RuntimeError("Failed to sample object/bowl positions")
        self.set_body_pose(self.args.object_name, so, oq); self.set_body_pose(self.args.bowl_name, sb, bq)
        mujoco.mj_forward(self.model, self.data); mink.move_mocap_to_frame(self.model, self.data, "target", "endpoint", "site")
        return dict(seed=int(seed), object_pose=dict(pos=so.tolist(), quat=oq.tolist()), bowl_pose=dict(pos=sb.tolist(), quat=bq.tolist()))

    def set_occluder_visible(self, visible):
        bid, gid = self.occ_ids()
        if bid < 0 or gid < 0: print(f"[WARN] occluder not found: {bid}, {gid}"); return
        self.model.geom_rgba[gid][3] = float(self.args.occluder_alpha) if visible else 0.0
        if not visible: self.model.body_pos[bid] = np.array([10,10,10], float)
        self.model.geom_contype[gid] = 0; self.model.geom_conaffinity[gid] = 0
        mujoco.mj_forward(self.model, self.data)

    def _estimate_table_z_under_object(self, object_pos=None, bowl_pos=None):
        candidates = []
        for gi in range(self.model.ngeom):
            try:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            except Exception:
                name = ""
            nl = name.lower()
            if any(tok in nl for tok in ["table", "desk", "plane", "floor", "top"]):
                try:
                    candidates.append(float(self.data.geom_xpos[gi][2]))
                except Exception:
                    pass
        if object_pos is not None:
            try:
                obj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.args.object_name)
                for gi in range(self.model.ngeom):
                    if int(self.model.geom_bodyid[gi]) == obj_bid:
                        candidates.append(float(self.data.geom_xpos[gi][2] - self.model.geom_size[gi][2]))
            except Exception:
                pass
        if object_pos is not None:
            candidates.append(float(object_pos[2]))
        if bowl_pos is not None:
            candidates.append(float(bowl_pos[2]))
        return float(max(candidates)) if candidates else 0.0

    def _mesh_world_bottom_z(self, geom_id: int):
        mesh_id = int(self.model.geom_dataid[geom_id])
        if mesh_id < 0:
            return float(self.data.geom_xpos[geom_id][2] - self.model.geom_size[geom_id][2])
        adr = int(self.model.mesh_vertadr[mesh_id])
        num = int(self.model.mesh_vertnum[mesh_id])
        verts = self.model.mesh_vert[adr:adr + num]
        geom_pos = self.data.geom_xpos[geom_id]
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        world_verts = geom_pos + verts @ geom_mat.T
        return float(np.min(world_verts[:, 2]))

    def randomize_can(self, seed):
        bid, gid = self.occ_ids()
        if bid < 0 or gid < 0:
            return dict(enabled=False, reason="occluder not found")

        rng = np.random.default_rng(seed + self.args.occluder_seed_offset)
        obj = np.asarray(self.get_body_pose(self.args.object_name)["pos"], float)
        try:
            bowl = np.asarray(self.get_body_pose(self.args.bowl_name)["pos"], float)
        except Exception:
            bowl = obj.copy()

        table_z = self._estimate_table_z_under_object(obj, bowl)
        x = float(np.clip(obj[0] + rng.uniform(-self.args.occluder_x_jitter, self.args.occluder_x_jitter), self.args.occluder_x_min, self.args.occluder_x_max))
        y = float(np.clip(obj[1] - rng.uniform(self.args.occluder_y_min_offset, self.args.occluder_y_max_offset), self.args.occluder_y_min, self.args.occluder_y_max))
        yaw = float(rng.uniform(-math.pi, math.pi))
        quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], float)

        mesh_id = int(self.model.geom_dataid[gid])
        mesh_min_z = None
        mesh_max_z = None
        if mesh_id >= 0:
            adr = int(self.model.mesh_vertadr[mesh_id])
            num = int(self.model.mesh_vertnum[mesh_id])
            verts = self.model.mesh_vert[adr:adr + num]
            mesh_min_z = float(np.min(verts[:, 2]))
            mesh_max_z = float(np.max(verts[:, 2]))
            z = table_z - mesh_min_z - 0.001
        else:
            z = table_z + float(self.model.geom_size[gid][2]) - 0.001

        pos = np.array([x, y, z], float)
        self.model.body_pos[bid] = pos
        self.model.body_quat[bid] = quat
        self.model.geom_rgba[gid][3] = float(self.args.occluder_alpha)
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0
        mujoco.mj_forward(self.model, self.data)

        target_bottom_z = float(table_z - 0.001)
        try:
            actual_bottom_z = self._mesh_world_bottom_z(gid)
            dz = target_bottom_z - actual_bottom_z
            self.model.body_pos[bid][2] += dz
            mujoco.mj_forward(self.model, self.data)
            pos = np.asarray(self.model.body_pos[bid], dtype=float).copy()
            final_bottom_z = self._mesh_world_bottom_z(gid)
            print(f"[OCCLUDER-Z-FIX] table_z={table_z:.6f}, target_bottom_z={target_bottom_z:.6f}, actual_bottom_z_before={actual_bottom_z:.6f}, dz={dz:.6f}, final_bottom_z={final_bottom_z:.6f}, body_z={float(self.model.body_pos[bid][2]):.6f}")
        except Exception as e:
            print("[WARN] occluder z-fix failed:", repr(e))
            final_bottom_z = None

        return dict(enabled=True, type="can", visual_only=True, pose=dict(pos=pos.tolist(), quat=quat.tolist()),
                    sampling=dict(object_pos=obj.tolist(), bowl_pos=bowl.tolist(), table_z_estimate=table_z,
                                  target_bottom_z=target_bottom_z, final_bottom_z=final_bottom_z,
                                  mesh_min_z=mesh_min_z, mesh_max_z=mesh_max_z))

    def observation(self):
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.args.camera_name) < 0: raise RuntimeError(f"Camera not found: {self.args.camera_name}")
        self.set_renderer_size(); return dict(state=self.data.sensordata[self.joint_pos_sensor_idx].copy().astype(np.float32),
                                             task=self.args.task_description, rgb=self.render_rgb(), depth=self.render_depth())

    def model_input(self, obs):
        batch = {"observation.state": torch.from_numpy(obs["state"]).float()}

        # RGB image: 3 x H x W, [0, 1]
        rgb = torch.from_numpy(obs["rgb"]).permute(2,0,1).contiguous().float() / 255.0

        # Depth as a single-channel tensor. This is used only when the dataset/policy
        # has a true depth feature key, not when depth was trained as a visual image.
        dep = self.depth_tensor(obs["depth"])

        # Depth visualization image: 3 x H x W, [0, 1].
        # Your checkpoint expects this key: observation.images.global_depth_vis.
        depth_vis_rgb = torch.from_numpy(self.depth_vis(obs["depth"])).permute(2,0,1).contiguous().float() / 255.0

        # First, satisfy exactly what the trained LeRobot DiffusionPolicy expects.
        # DiffusionPolicy.select_action stacks all keys in self.policy.config.image_features.
        image_features = list(getattr(self.policy.config, "image_features", []) or [])
        for key in image_features:
            kl = key.lower()
            if "depth_vis" in kl or "depthvis" in kl:
                batch[key] = depth_vis_rgb
            elif "depth" in kl and key.startswith("observation.images"):
                # If a depth-like item is registered as an image feature, it must be 3ch image-like.
                batch[key] = depth_vis_rgb
            elif key.endswith(self.args.camera_name) or "global_cam" in key or "rgb" in kl:
                batch[key] = rgb
            else:
                # Safe fallback for unknown image feature names.
                batch[key] = rgb

        # Keep compatibility with the previous auto/separate/concat logic.
        if self.rgbd_key is not None and self.rgbd_key not in batch:
            batch[self.rgbd_key] = rgb if self.args.rgbd_mode == "rgb_only" else torch.cat([rgb, dep], 0).contiguous()
        else:
            if self.rgb_key is not None and self.rgb_key not in batch:
                batch[self.rgb_key] = rgb
            if self.depth_key is not None and self.depth_key not in batch:
                batch[self.depth_key] = dep

        if not image_features and self.rgbd_key is None and self.rgb_key is None and self.depth_key is None:
            raise RuntimeError("RGBD key를 찾지 못했습니다. --rgbd-concat-key 또는 --rgb-key/--depth-key를 지정하세요.")

        batch["task"] = obs.get("task", self.args.task_description)
        return batch

    def predict(self, obs):
        batch = self.preprocess(self.model_input(obs))
        with torch.inference_mode(): out = self.postprocess(self.policy.select_action(batch))
        arr = out["action"] if isinstance(out, dict) and "action" in out else out
        if isinstance(arr, torch.Tensor): arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, np.float32)
        return arr[0] if arr.ndim > 1 else arr

    def success(self):
        if self.args.quiet_success_check:
            with contextlib.redirect_stdout(io.StringIO()): return bool(self.task.check_success())
        return bool(self.task.check_success())

    def video_frame(self, obs, cond, step):
        rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR); dep = cv2.cvtColor(self.depth_vis(obs["depth"]), cv2.COLOR_RGB2BGR)
        fr = np.concatenate([rgb, dep], 1)
        cv2.putText(fr, f"RGBD  OCCLUSION={cond}  STEP={step}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
        return fr

    def eval_one(self, episode_idx, seed, cond):
        self.restore_base_scene(); visual = self.randomize_visuals(); pose = self.randomize_object_bowl(seed)
        occ = self.randomize_can(seed) if cond == "can" else dict(enabled=False, condition="none", type="can")
        self.set_occluder_visible(cond == "can")
        writer = None; video_path = None
        if self.args.save_video and episode_idx < self.args.max_video_episodes:
            video_path = str(self.output_dir / "videos" / f"episode_{episode_idx:04d}_seed_{seed}_{cond}.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.args.video_fps, (self.args.camera_width*2, self.args.camera_height))
        traj = []; success = False; first = None; hold = 0
        for step in range(self.args.max_steps):
            obs = self.observation(); action = self.predict(obs)
            self.data.ctrl[:self.nu] = np.asarray(action[:self.nu], float)
            for _ in range(self.args.decimation): mujoco.mj_step(self.model, self.data)
            traj.append(dict(step=int(step), time=float(self.data.time), action=action.tolist(), joint_state=self.data.sensordata[self.joint_pos_sensor_idx].tolist()))
            if writer is not None: writer.write(self.video_frame(obs, cond, step))
            ok = self.success()
            if ok and first is None: first = step; hold = self.args.success_hold_steps; print(f"[INFO] ep={episode_idx} seed={seed} cond={cond} success step={step}")
            if first is not None:
                success = True
                if hold <= 0: break
                hold -= 1
            if self.args.log_every > 0 and step % self.args.log_every == 0: print(f"[INFO] ep={episode_idx} seed={seed} cond={cond} step={step} success={ok}")
        if writer is not None: writer.release()
        if self.args.save_trajectories: save_json(self.output_dir / f"episode_{episode_idx:04d}_seed_{seed}_{cond}_trajectory.json", {"trajectory": traj})
        return dict(episode_idx=int(episode_idx), seed=int(seed), occlusion_condition=cond, success=bool(success),
                    success_first_step=None if first is None else int(first), num_steps=len(traj), video_path=video_path,
                    sampled_pose_info=pose, visual_info=visual, occluder_info=occ)

    def aggregate(self, results):
        valid = [r for r in results if "error" not in r]; by = {}
        for cond in sorted(set(r["occlusion_condition"] for r in valid)):
            rs = [r for r in valid if r["occlusion_condition"] == cond]; ss = [r for r in rs if r.get("success")]
            steps = [r["success_first_step"] for r in ss if r.get("success_first_step") is not None]
            by[cond] = dict(num_valid=len(rs), num_success=len(ss), success_rate=None if not rs else len(ss)/len(rs),
                            avg_success_first_step=None if not steps else float(np.mean(steps)), avg_num_steps=None if not rs else float(np.mean([r["num_steps"] for r in rs])))
        paired = None
        if "none" in by and "can" in by:
            n = {r["seed"]: r for r in valid if r["occlusion_condition"] == "none"}; c = {r["seed"]: r for r in valid if r["occlusion_condition"] == "can"}; seeds = sorted(set(n)&set(c))
            rows=[]; degraded=improved=both_success=both_fail=0
            for s in seeds:
                ns=bool(n[s].get("success")); cs=bool(c[s].get("success"))
                if ns and not cs: degraded += 1; outcome="none_success_can_fail"
                elif (not ns) and cs: improved += 1; outcome="none_fail_can_success"
                elif ns and cs: both_success += 1; outcome="both_success"
                else: both_fail += 1; outcome="both_fail"
                rows.append(dict(seed=int(s), none_success=ns, can_success=cs, outcome=outcome,
                                 none_success_first_step=n[s].get("success_first_step"), can_success_first_step=c[s].get("success_first_step")))
            paired = dict(num_paired=len(seeds), none_success_can_fail=degraded, none_fail_can_success=improved,
                          both_success=both_success, both_fail=both_fail, occlusion_drop_rate=None if not seeds else degraded/len(seeds), paired_rows=rows)
        return dict(checkpoint=str(self.checkpoint_path), dataset_repo_id=self.args.dataset_repo_id,
                    dataset_root=None if self.dataset_root_path is None else str(self.dataset_root_path), robot=self.args.robot, task=self.args.task,
                    task_description=self.args.task_description, camera_name=self.args.camera_name, input_type="rgbd_1camera",
                    rgbd_mode=self.args.rgbd_mode, rgb_key=self.rgb_key, depth_key=self.depth_key, rgbd_concat_key=self.rgbd_key,
                    num_eval_episodes_requested=int(self.args.num_eval_episodes), eval_seed_start=int(self.args.eval_seed_start),
                    occlusion_mode=self.args.occlusion_mode, occluder_type="can", by_condition=by, paired_comparison=paired)

    def run(self):
        conditions = ["none", "can"] if self.args.occlusion_mode == "both" else [self.args.occlusion_mode]
        results = []
        for ep in range(self.args.num_eval_episodes):
            seed = self.args.eval_seed_start + ep
            print("\n" + "="*80 + f"\n[EVAL {ep+1}/{self.args.num_eval_episodes}] seed={seed} conditions={conditions}\n" + "="*80)
            for cond in conditions:
                try:
                    r = self.eval_one(ep, seed, cond); results.append(r)
                    print(f"[RESULT] ep={ep} seed={seed} cond={cond} success={r['success']} steps={r['num_steps']}")
                except Exception as e:
                    traceback.print_exc(); results.append(dict(episode_idx=int(ep), seed=int(seed), occlusion_condition=cond, success=False, error=str(e)))
        agg = self.aggregate(results); save_json(self.output_dir / "aggregate_eval_summary.json", agg); save_json(self.output_dir / "per_episode_results.json", {"results": results})
        print("\n✅ RGBD 1-camera can-occlusion evaluation completed")
        print(json.dumps(agg, indent=2, ensure_ascii=False)); print(f"- output_dir: {self.output_dir}")


def build_argparser():
    p = argparse.ArgumentParser(description="LeRobot RGBD 1-camera DiffusionPolicy can-occlusion evaluator")
    p.add_argument("--robot", default="piper"); p.add_argument("--task", default="place_block"); p.add_argument("--task-description", default="place the block")
    p.add_argument("--checkpoint", required=True); p.add_argument("--dataset-repo-id", default="apple/piper_place_block_global_rgbd_can_occlusion")
    p.add_argument("--dataset-root", default=os.path.expanduser("~/lerobot/DISCOVERSE/lerobot_dataset_global_rgbd_can_occlusion"))
    p.add_argument("--output-dir", default=os.path.expanduser("~/lerobot/DISCOVERSE/outputs/eval/piper_place_block_rgbd_1cam_can_occlusion_eval"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--camera-name", default="global_cam"); p.add_argument("--camera-width", type=int, default=640); p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--global-cam-pos", default="0 0.4 1.2"); p.add_argument("--global-cam-xyaxes", default="1 0 0 0 0.707 0.707")
    p.add_argument("--rgbd-mode", default="auto", choices=["auto","concat","separate","rgb_only"])
    p.add_argument("--rgbd-concat-key", default=None); p.add_argument("--rgb-key", default=None); p.add_argument("--depth-key", default=None)
    p.add_argument("--depth-normalization", default="clip01", choices=["clip01","raw"]); p.add_argument("--depth-near-clip", type=float, default=0.01); p.add_argument("--depth-far-clip", type=float, default=2.0)
    p.add_argument("--num-eval-episodes", type=int, default=100); p.add_argument("--eval-seed-start", type=int, default=100000)
    p.add_argument("--max-steps", type=int, default=400); p.add_argument("--decimation", type=int, default=5); p.add_argument("--success-hold-steps", type=int, default=40)
    p.add_argument("--quiet-success-check", action="store_true", default=True); p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--object-name", default="block_green"); p.add_argument("--bowl-name", default="bowl_pink")
    p.add_argument("--object-xy-range", type=float, default=0.08); p.add_argument("--bowl-xy-range", type=float, default=0.08); p.add_argument("--min-object-bowl-dist", type=float, default=0.06); p.add_argument("--max-sample-trials", type=int, default=200)
    p.add_argument("--randomize-visuals", action="store_true", default=False)
    p.add_argument("--occlusion-mode", default="both", choices=["none","can","both"])
    p.add_argument("--occluder-name", default="occluder_0"); p.add_argument("--occluder-geom-name", default="occluder_0_geom"); p.add_argument("--occluder-alpha", type=float, default=1.0)
    p.add_argument("--occluder-seed-offset", type=int, default=7777); p.add_argument("--occluder-x-jitter", type=float, default=0.055)
    p.add_argument("--occluder-y-min-offset", type=float, default=0.045); p.add_argument("--occluder-y-max-offset", type=float, default=0.120)
    p.add_argument("--occluder-x-min", type=float, default=-0.18); p.add_argument("--occluder-x-max", type=float, default=0.18); p.add_argument("--occluder-y-min", type=float, default=0.72); p.add_argument("--occluder-y-max", type=float, default=1.02)
    p.add_argument("--save-video", action="store_true"); p.add_argument("--max-video-episodes", type=int, default=10); p.add_argument("--video-fps", type=int, default=30); p.add_argument("--save-trajectories", action="store_true")
    p.add_argument("--plot-train-log", default=None, help="lerobot-train 로그 파일 경로. 지정하면 학습 loss/lr 그래프를 output_dir/train_log_plots에 저장합니다.")
    p.add_argument("--plot-only", action="store_true", help="평가는 실행하지 않고 --plot-train-log 그래프만 저장합니다.")
    return p


def main():
    args = build_argparser().parse_args()
    if args.plot_train_log:
        save_train_log_plots(args.plot_train_log, args.output_dir)
        if args.plot_only:
            return
    RGBDOneCamCanOcclusionEvaluator(args).run()


if __name__ == "__main__": main()
