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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def _occluder_xml(occluder_type: str) -> str:
    """Return a static visual-only occluder body. No joint means no gravity/falling."""
    if occluder_type == "can":
        return """
        <body name="occluder_0" pos="0 0.80 0.70" quat="1 0 0 0">
            <geom name="occluder_0_geom"
                  type="mesh"
                  mesh="occluder_coke_mesh"
                  material="occluder_coke_mat"
                  contype="0"
                  conaffinity="0"/>
        </body>
        """
    if occluder_type == "bottle":
        return """
        <body name="occluder_0" pos="0.0 0.75 0.82" quat="1 0 0 0">
            <geom name="occluder_0_geom" type="cylinder" size="0.026 0.100"
                  rgba="0.05 0.25 0.85 0.90" contype="0" conaffinity="0"/>
            <geom name="occluder_0_neck" type="cylinder" pos="0 0 0.125" size="0.014 0.035"
                  rgba="0.05 0.25 0.85 0.90" contype="0" conaffinity="0"/>
            <geom name="occluder_0_cap" type="cylinder" pos="0 0 0.165" size="0.016 0.009"
                  rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
        </body>
        """
    if occluder_type == "block":
        return """
        <body name="occluder_0" pos="0.0 0.80 0.73" quat="1 0 0 0">
            <geom name="occluder_0_geom" type="box" size="0.035 0.035 0.035"
                  rgba="0.10 0.70 0.10 1" contype="0" conaffinity="0"/>
        </body>
        """
    return """
    <body name="occluder_0" pos="0.0 0.80 0.78" quat="1 0 0 0">
        <geom name="occluder_0_geom" type="box" size="0.030 0.030 0.100"
              rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
    </body>
    """

def generate_robot_task_model(
    robot_name: str,
    task_name: str,
    add_occluder: bool = True,
    occluder_type: str = "can",
    global_cam_pos: str = "0 0.4 1.2",
    global_cam_xyaxes: str = "1 0 0 0 0.707 0.707",
) -> str:
    xml_path = os.path.join(
        DISCOVERSE_ASSETS_DIR,
        "mjcf/tmp",
        f"{robot_name}_{task_name}_global_rgbd_{occluder_type}_occlusion.xml",
    )
    make_env(robot_name, task_name, xml_path)

    with open(xml_path, "r", encoding="utf-8") as f:
        xml_data = f.read()

    if add_occluder and occluder_type == "can" and 'name="occluder_coke_mesh"' not in xml_data:
        coke_mesh = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes/object/coke/coke.obj")
        coke_tex = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes/object/coke/coke.png")
        asset_block = f"""
  <asset>
    <texture name="occluder_coke_tex" type="2d" file="{coke_tex}"/>
    <material name="occluder_coke_mat" texture="occluder_coke_tex" specular="0.35" shininess="0.45"/>
    <mesh name="occluder_coke_mesh" file="{coke_mesh}" scale="1 1 1"/>
  </asset>
"""
        if "<worldbody>" not in xml_data:
            raise RuntimeError("MJCF에 <worldbody> 태그를 찾을 수 없습니다.")
        xml_data = xml_data.replace("<worldbody>", asset_block + "\n<worldbody>", 1)

    insertion = ""
    if 'name="global_cam"' not in xml_data:
        insertion += f"""
        <camera name="global_cam" pos="{global_cam_pos}" xyaxes="{global_cam_xyaxes}"/>
        """
    if add_occluder and 'name="occluder_0"' not in xml_data:
        insertion += _occluder_xml(occluder_type)

    if insertion.strip():
        if "</worldbody>" not in xml_data:
            raise RuntimeError("MJCF에 </worldbody> 태그를 찾을 수 없습니다.")
        xml_data = xml_data.replace("</worldbody>", insertion + "\n</worldbody>")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_data)
    return xml_path

class GlobalRGBDObjectOcclusionGTExecutor:
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
        object_name: str = "block_green",
        bowl_name: str = "bowl_pink",
        object_xy_range: float = 0.08,
        bowl_xy_range: float = 0.08,
        min_object_bowl_dist: float = 0.06,
        occluder_type: str = "can",
        occluder_name: str = "occluder_0",
        occluder_geom_name: str = "occluder_0_geom",
        occluder_enabled: bool = True,
        occluder_visual_only: bool = True,
        occluder_alpha_min: float = 0.35,
        occluder_alpha_max: float = 0.55,
        occluder_xy_jitter: float = 0.08,
        occluder_z_jitter: float = 0.10,
        occluder_allow_lying: bool = False,
        save_depth: bool = True,
        save_depth_vis: bool = True,
        keep_failed: bool = True,
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
        self.occluder_type = occluder_type
        self.occluder_name = occluder_name
        self.occluder_geom_name = occluder_geom_name
        self.occluder_enabled = occluder_enabled
        self.occluder_visual_only = occluder_visual_only
        self.occluder_alpha_min = occluder_alpha_min
        self.occluder_alpha_max = occluder_alpha_max
        self.occluder_xy_jitter = occluder_xy_jitter
        self.occluder_z_jitter = occluder_z_jitter
        self.occluder_allow_lying = occluder_allow_lying
        self.global_cam_pos = np.array([0.0, 0.4, 1.2], dtype=float)
        self.save_depth = save_depth
        self.save_depth_vis = save_depth_vis
        self.keep_failed = keep_failed
        self.depth_near_clip = 0.01
        self.depth_far_clip = 2.0

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
        self.camera_cfgs = {"global_cam": {"name": "global_cam", "width": 640, "height": 480}}
        self.camera_encoders = {}
        self.base_scene_snapshot: Optional[SceneStateSnapshot] = None
        self.base_object_pose: Optional[Dict[str, Any]] = None
        self.base_bowl_pose: Optional[Dict[str, Any]] = None
        self.current_episode_variables: Optional[EpisodeVariables] = None
        self.current_sample_idx = 0
        self.last_episode_seed = 0
        self.save_dir = None
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
            raise ValueError(f"Body not found: {body_name}")
        jnt_adr = int(self.mj_model.body_jntadr[body_id])
        if jnt_adr < 0:
            raise ValueError(f"Body has no joint: {body_name}")
        jnt_id = jnt_adr
        if int(self.mj_model.jnt_type[jnt_id]) != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body joint is not free joint: {body_name}")
        return int(self.mj_model.jnt_qposadr[jnt_id])

    def _get_free_body_pose(self, body_name: str) -> Dict[str, Any]:
        qadr = self._get_free_joint_qpos_adr(body_name)
        q = self.mj_data.qpos[qadr:qadr + 7].copy()
        return {"body_name": body_name, "qpos_adr": int(qadr), "pos": q[:3].tolist(), "quat": q[3:7].tolist()}

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray):
        qadr = self._get_free_joint_qpos_adr(body_name)
        self.mj_data.qpos[qadr:qadr + 3] = pos
        self.mj_data.qpos[qadr + 3:qadr + 7] = quat

    def hide_visual_markers(self):
        for i in range(self.mj_model.ngeom):
            body_id = int(self.mj_model.geom_bodyid[i])
            if self.mj_model.body_mocapid[body_id] != -1:
                self.mj_model.geom_rgba[i][3] = 0.0
        for i in range(self.mj_model.nsite):
            self.mj_model.site_rgba[i][3] = 0.0
        for geom_name in ["target_box", "target_x", "target_y", "target_z", "origin_x", "origin_y", "origin_z", "axis_x", "axis_y", "axis_z"]:
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                self.mj_model.geom_rgba[geom_id][3] = 0.0

    def build_base_scene_once(self):
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, "target", "endpoint", "site")
        self.hide_visual_markers()
        self.base_scene_snapshot = self._capture_scene_snapshot()
        self.base_object_pose = self._get_free_body_pose(self.object_name)
        self.base_bowl_pose = self._get_free_body_pose(self.bowl_name)

    def restore_base_scene(self):
        if self.base_scene_snapshot is None:
            raise RuntimeError("base_scene_snapshot is not initialized")
        s = self.base_scene_snapshot
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = np.asarray(s.qpos, dtype=float)
        self.mj_data.qvel[:] = np.asarray(s.qvel, dtype=float)
        if len(s.act) > 0 and self.mj_data.act is not None:
            self.mj_data.act[:] = np.asarray(s.act, dtype=float)
        self.mj_data.ctrl[:] = np.asarray(s.ctrl, dtype=float)
        if self.mj_model.nmocap > 0 and len(s.mocap_pos) > 0:
            self.mj_data.mocap_pos[:] = np.asarray(s.mocap_pos, dtype=float)
            self.mj_data.mocap_quat[:] = np.asarray(s.mocap_quat, dtype=float)
        if self.mj_model.nuserdata > 0 and len(s.userdata) > 0:
            self.mj_data.userdata[:] = np.asarray(s.userdata, dtype=float)
        self.mj_data.time = float(s.time)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, "target", "endpoint", "site")
        self.hide_visual_markers()

    def randomize_object_and_bowl_positions(self, seed: int) -> Dict[str, Any]:
        if self.base_object_pose is None or self.base_bowl_pose is None:
            raise RuntimeError("Base object/bowl poses are not initialized")
        self.set_global_seed(seed)
        base_obj_pos = np.asarray(self.base_object_pose["pos"], dtype=float)
        base_obj_quat = np.asarray(self.base_object_pose["quat"], dtype=float)
        base_bowl_pos = np.asarray(self.base_bowl_pose["pos"], dtype=float)
        base_bowl_quat = np.asarray(self.base_bowl_pose["quat"], dtype=float)
        chosen_obj = None
        chosen_bowl = None
        for _ in range(200):
            obj_xy = base_obj_pos[:2] + np.random.uniform(-self.object_xy_range, self.object_xy_range, size=2)
            bowl_xy = base_bowl_pos[:2] + np.random.uniform(-self.bowl_xy_range, self.bowl_xy_range, size=2)
            if np.linalg.norm(obj_xy - bowl_xy) < self.min_object_bowl_dist:
                continue
            chosen_obj = np.array([obj_xy[0], obj_xy[1], base_obj_pos[2]], dtype=float)
            chosen_bowl = np.array([bowl_xy[0], bowl_xy[1], base_bowl_pos[2]], dtype=float)
            break
        if chosen_obj is None or chosen_bowl is None:
            raise RuntimeError("Failed to sample object/bowl positions")
        self._set_free_body_pose(self.object_name, chosen_obj, base_obj_quat)
        self._set_free_body_pose(self.bowl_name, chosen_bowl, base_bowl_quat)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return {
            "object_name": self.object_name,
            "bowl_name": self.bowl_name,
            "object_pose": {"pos": chosen_obj.tolist(), "quat": base_obj_quat.tolist()},
            "bowl_pose": {"pos": chosen_bowl.tolist(), "quat": base_bowl_quat.tolist()},
            "object_xy_range": float(self.object_xy_range),
            "bowl_xy_range": float(self.bowl_xy_range),
            "min_object_bowl_dist": float(self.min_object_bowl_dist),
        }

    def _sample_occluder_size_and_color(self, rng: np.random.Generator):
        if self.occluder_type == "can":
            radius = rng.uniform(0.025, 0.040)
            half_height = rng.uniform(0.070, 0.115)
            rgba = rng.choice([
                np.array([0.85, 0.05, 0.05, 1.0]),
                np.array([0.05, 0.25, 0.85, 1.0]),
                np.array([0.95, 0.75, 0.05, 1.0]),
            ])
            return np.array([radius, half_height], dtype=float), rgba
        if self.occluder_type == "bottle":
            radius = rng.uniform(0.018, 0.032)
            half_height = rng.uniform(0.090, 0.135)
            rgba = rng.choice([
                np.array([0.05, 0.25, 0.85, 0.90]),
                np.array([0.05, 0.55, 0.25, 0.90]),
                np.array([0.75, 0.75, 0.85, 0.85]),
            ])
            return np.array([radius, half_height], dtype=float), rgba
        if self.occluder_type == "block":
            size = rng.uniform([0.025, 0.025, 0.025], [0.050, 0.050, 0.055])
            rgba = rng.choice([
                np.array([0.10, 0.70, 0.10, 1.0]),
                np.array([0.90, 0.20, 0.10, 1.0]),
                np.array([0.10, 0.20, 0.90, 1.0]),
            ])
            return size.astype(float), rgba
        # box/panel
        size = rng.uniform([0.020, 0.020, 0.070], [0.045, 0.045, 0.130])
        rgba = np.array([0.05, 0.05, 0.05, 1.0])
        return size.astype(float), rgba

    def randomize_occluder_pose(self, seed: int) -> Optional[Dict[str, Any]]:
        if not self.occluder_enabled:
            print("[OCCLUDER] disabled")
            return None

        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.occluder_name)
        geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, self.occluder_geom_name)
        if body_id < 0 or geom_id < 0:
            print(f"⚠️ occluder not found: body_id={body_id}, geom_id={geom_id}")
            return None

        rng = np.random.default_rng(seed + 7777)
        object_pos = np.asarray(self._get_free_body_pose(self.object_name)["pos"], dtype=float)
        try:
            bowl_pos = np.asarray(self._get_free_body_pose(self.bowl_name)["pos"], dtype=float)
        except Exception:
            bowl_pos = object_pos.copy()

        table_z_candidates = []
        try:
            obj_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
            for gi in range(self.mj_model.ngeom):
                if int(self.mj_model.geom_bodyid[gi]) == obj_body_id:
                    # block geom bottom: center_z - half_size_z
                    table_z_candidates.append(float(self.mj_data.geom_xpos[gi][2] - self.mj_model.geom_size[gi][2]))
        except Exception:
            pass
        table_z_candidates.append(float(min(object_pos[2], bowl_pos[2])))
        table_z = float(max(table_z_candidates))

        x = object_pos[0] + rng.uniform(-0.055, 0.055)
        y = object_pos[1] - rng.uniform(0.045, 0.120)
        x = float(np.clip(x, -0.18, 0.18))
        y = float(np.clip(y, 0.72, 1.02))

        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)
        orientation_type = "upright"

        sampled_size = None
        rgba = None
        mesh_min_z = None
        mesh_max_z = None

        if self.occluder_type == "can":
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            if mesh_id >= 0:
                adr = int(self.mj_model.mesh_vertadr[mesh_id])
                num = int(self.mj_model.mesh_vertnum[mesh_id])
                verts = self.mj_model.mesh_vert[adr:adr + num]
                mesh_min_z = float(np.min(verts[:, 2]))
                mesh_max_z = float(np.max(verts[:, 2]))
                CAN_TABLE_MARGIN = -0.001  # visual grounding: 1 mm slightly into table
                center_z = table_z - mesh_min_z + CAN_TABLE_MARGIN
                sampled_size = [float(mesh_max_z - mesh_min_z)]
            else:
                CAN_TABLE_MARGIN = -0.001  # visual grounding fallback
                center_z = table_z + 0.09 + CAN_TABLE_MARGIN
                sampled_size = [0.18]
            occluder_pos = np.array([x, y, center_z], dtype=float)
            self.mj_model.body_pos[body_id] = occluder_pos
            self.mj_model.body_quat[body_id] = quat
            self.mj_model.geom_contype[geom_id] = 0
            self.mj_model.geom_conaffinity[geom_id] = 0
            bottom_z = table_z + CAN_TABLE_MARGIN
        else:
            if self.occluder_type == "bottle":
                radius = float(rng.uniform(0.020, 0.030))
                half_height = float(rng.uniform(0.090, 0.125))
                sampled_size = np.array([radius, half_height], dtype=float)
                rgba = np.array([0.05, 0.25, 0.85, 0.90])
            elif self.occluder_type == "block":
                sampled_size = rng.uniform([0.030, 0.030, 0.030], [0.050, 0.050, 0.055]).astype(float)
                half_height = float(sampled_size[2])
                rgba = np.array([0.10, 0.70, 0.10, 1.0])
            else:
                sampled_size = rng.uniform([0.020, 0.020, 0.070], [0.045, 0.045, 0.130]).astype(float)
                half_height = float(sampled_size[2])
                rgba = np.array([0.05, 0.05, 0.05, 1.0])

            center_z = table_z + half_height + 0.002
            occluder_pos = np.array([x, y, center_z], dtype=float)
            self.mj_model.body_pos[body_id] = occluder_pos
            self.mj_model.body_quat[body_id] = quat
            self.mj_model.geom_size[geom_id, :len(sampled_size)] = sampled_size
            self.mj_model.geom_rgba[geom_id] = rgba
            if self.occluder_visual_only:
                self.mj_model.geom_contype[geom_id] = 0
                self.mj_model.geom_conaffinity[geom_id] = 0
            bottom_z = center_z - half_height

        mujoco.mj_forward(self.mj_model, self.mj_data)
        

        # ---- CAN_GROUND_Z_FIX_BEGIN ----

        # Compute the rendered coke mesh bottom in world coordinates and shift it

        # to sit 1 mm below the table top, preventing a visible floating gap.

        if self.occluder_type == "can":

            try:

                mesh_id_fix = int(self.mj_model.geom_dataid[geom_id])

                if mesh_id_fix >= 0:

                    adr_fix = int(self.mj_model.mesh_vertadr[mesh_id_fix])

                    num_fix = int(self.mj_model.mesh_vertnum[mesh_id_fix])

                    verts_fix = self.mj_model.mesh_vert[adr_fix:adr_fix + num_fix]


                    geom_pos_fix = self.mj_data.geom_xpos[geom_id]

                    geom_mat_fix = self.mj_data.geom_xmat[geom_id].reshape(3, 3)

                    world_verts_fix = geom_pos_fix + verts_fix @ geom_mat_fix.T


                    actual_bottom_z = float(np.min(world_verts_fix[:, 2]))

                    target_bottom_z = float(table_z - 0.001)

                    dz = target_bottom_z - actual_bottom_z


                    self.mj_model.body_pos[body_id][2] += dz

                    mujoco.mj_forward(self.mj_model, self.mj_data)


                    # Keep log/metadata variables consistent with the final corrected pose.

                    occluder_pos = np.asarray(self.mj_model.body_pos[body_id], dtype=float).copy()

                    bottom_z = target_bottom_z


                    print(

                        f"[OCCLUDER-Z-FIX] actual_bottom_z={actual_bottom_z:.6f}, "

                        f"target_bottom_z={target_bottom_z:.6f}, dz={dz:.6f}, "

                        f"final_body_z={float(self.mj_model.body_pos[body_id][2]):.6f}"

                    )

            except Exception as e:

                print("[OCCLUDER-Z-FIX] failed:", repr(e))

        # ---- CAN_GROUND_Z_FIX_END ----print(f"[OCCLUDER] type={self.occluder_type} mesh_real_coke={self.occluder_type == 'can'} orient={orientation_type} pos={occluder_pos.round(4).tolist()} table_z={table_z:.4f} bottom_z={bottom_z:.4f} object_pos={object_pos.round(4).tolist()} bowl_pos={bowl_pos.round(4).tolist()} mesh_z=[{mesh_min_z},{mesh_max_z}] size={sampled_size}")

        return {
            "enabled": True,
            "type": self.occluder_type,
            "visual_only": bool(self.occluder_visual_only),
            "static_body_no_gravity": True,
            "orientation_type": orientation_type,
            "occluder_name": self.occluder_name,
            "occluder_geom_name": self.occluder_geom_name,
            "pose": {"pos": occluder_pos.tolist(), "quat": quat.tolist()},
            "size": sampled_size if isinstance(sampled_size, list) else sampled_size.tolist(),
            "rgba": None if rgba is None else rgba.tolist(),
            "sampling": {
                "object_pos": object_pos.tolist(),
                "bowl_pos": bowl_pos.tolist(),
                "table_z_estimate": table_z,
                "bottom_z": bottom_z,
                "mesh_min_z": mesh_min_z,
                "mesh_max_z": mesh_max_z,
                "placement": "real_coke_mesh_bottom_on_table_near_object",
            },
        }

    def set_renderer_size(self, width: int, height: int):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height

    def get_rgb_image(self, camera_name: str) -> np.ndarray:
        scene_opt = mujoco.MjvOption()
        for i in range(len(scene_opt.sitegroup)):
            scene_opt.sitegroup[i] = 0
        self.renderer.update_scene(self.mj_data, camera_name, scene_option=scene_opt)
        return self.renderer.render().copy()

    def get_depth_image(self, camera_name: str) -> np.ndarray:
        scene_opt = mujoco.MjvOption()
        for i in range(len(scene_opt.sitegroup)):
            scene_opt.sitegroup[i] = 0
        self.renderer.update_scene(self.mj_data, camera_name, scene_option=scene_opt)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy().astype(np.float32)
        self.renderer.disable_depth_rendering()
        return depth

    def make_depth_visualization(self, depth: np.ndarray) -> np.ndarray:
        d = depth.copy()
        d[~np.isfinite(d)] = self.depth_far_clip
        d = np.clip(d, self.depth_near_clip, self.depth_far_clip)
        d_norm = (d - self.depth_near_clip) / (self.depth_far_clip - self.depth_near_clip + 1e-6)
        d_u8 = (255.0 * (1.0 - d_norm)).astype(np.uint8)
        return np.stack([d_u8, d_u8, d_u8], axis=-1)

    def get_observation(self):
        obs = {
            "time": self.mj_data.time,
            "jq": self.mj_data.sensordata[self.joint_pos_sensor_idx].tolist(),
            "action": self.action[:self.mujoco_ctrl_dim].tolist(),
            "img": {},
            "depth": {},
            "sample_idx": self.current_sample_idx,
        }
        camera_name = "global_cam"
        cfg = self.camera_cfgs[camera_name]
        if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
            raise RuntimeError(f"Camera not found: {camera_name}")
        self.set_renderer_size(cfg["width"], cfg["height"])
        obs["img"][camera_name] = self.get_rgb_image(camera_name)
        if self.save_depth:
            obs["depth"][camera_name] = self.get_depth_image(camera_name)
        return obs

    def set_target_from_primitive(self, state_config):
        try:
            primitive = state_config["primitive"]
            params = state_config.get("params", {})
            gripper_state = state_config.get("gripper_state", "open")

            current_ctrl = self.mj_data.ctrl[:self.mujoco_ctrl_dim].copy()
            if current_ctrl.shape[0] == self.target_control.shape[0]:
                self.target_control[:] = current_ctrl
            else:
                self.target_control[:] = self.action[:self.mujoco_ctrl_dim]

            gripper_only_primitive = False

            if primitive == "move_to_object":
                object_name = params.get("object_name", "")
                offset = np.array(params.get("offset", [0, 0, 0]), dtype=float)
                if object_name:
                    object_tmat = get_body_tmat(self.mj_data, object_name)
                    target_pos = object_tmat[:3, 3] + offset
                    site_name = self.task.robot_interface.robot_config.end_effector_site
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                    solution, converged, _ = self.task.robot_interface.ik_solver.solve_ik(target_pos, target_rmat, self.mj_data.qpos.copy())
                    if converged:
                        self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                        self.set_mocap_target("target", target_pos, np.array([1.0, 0.0, 0.0, 0.0]))
                    else:
                        print(f"⚠️ IK failed for move_to_object target_pos={target_pos}")
                        return False

            elif primitive == "move_relative":
                offset = np.array(params.get("offset", [0, 0, 0]), dtype=float)
                site_name = self.task.robot_interface.robot_config.end_effector_site
                site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                current_pos = self.mj_data.site_xpos[site_id].copy()
                target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                target_pos = current_pos + offset
                solution, converged, _ = self.task.robot_interface.ik_solver.solve_ik(target_pos, target_rmat, self.mj_data.qpos.copy())
                if converged:
                    self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                    self.set_mocap_target("target", target_pos, np.array([1.0, 0.0, 0.0, 0.0]))
                else:
                    print(f"⚠️ IK failed for move_relative target_pos={target_pos}")
                    return False

            elif primitive in ["grasp_object", "close_gripper", "grasp", "pick"]:
                gripper_state = "close"
                gripper_only_primitive = True

            elif primitive in ["release_object", "open_gripper", "release", "place_object"]:
                gripper_state = "open"
                gripper_only_primitive = True

            else:
                print(f"⚠️ Unknown primitive: {primitive}. Treating as no-op with gripper_state={gripper_state}")

            if gripper_state == "open":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.open()
            elif gripper_state == "close":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.close()

            if gripper_only_primitive and self.current_delay <= 0:
                self.current_delay = 0.60

            dif = np.abs(self.mj_data.ctrl[:self.mujoco_ctrl_dim].copy() - self.target_control)
            self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            print(f"[PRIMITIVE] {primitive} gripper_state={gripper_state} delay={self.current_delay:.3f}")
            return True
        except Exception:
            traceback.print_exc()
            return False

    def set_mocap_target(self, target_name, target_pos, target_quat):
        mocap_id = self.mj_model.body(target_name).mocapid
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = target_pos
            self.mj_data.mocap_quat[mocap_id] = target_quat
        geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_box")
        if geom_id >= 0:
            self.mj_model.geom_rgba[geom_id][3] = 0.0

    def check_action_done(self):
        position_error = np.linalg.norm(self.get_current_qpos()[:self.n_arm_joints] - self.target_control[:self.n_arm_joints])
        if position_error >= 0.02:
            return False
        if self.current_delay > 0 and self.delay_start_sim_time is not None:
            return (self.mj_data.time - self.delay_start_sim_time) >= self.current_delay
        return True

    def step(self, decimation=5):
        try:
            if self.stm.trigger():
                if self.stm.state_idx < self.total_states:
                    state_config = self.resolved_states[self.stm.state_idx]
                    self.current_delay = state_config.get("delay", 0.0)
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
                    self.action[i], self.target_control[i],
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
        return bool(self.task.check_success())

    def prepare_episode(self, sample_idx: int, episode_seed: int):
        self.current_sample_idx = sample_idx
        self.last_episode_seed = episode_seed
        self.restore_base_scene()
        pose_info = self.randomize_object_and_bowl_positions(seed=episode_seed)
        occluder_info = self.randomize_occluder_pose(seed=episode_seed)
        self.reset_runtime_state()
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]
        sample_dir = os.path.join(self.output_root, f"sample_{sample_idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        self.save_dir = sample_dir
        self.current_episode_variables = EpisodeVariables(
            seed=episode_seed,
            object_initial_qpos=self.get_current_qpos().tolist(),
            material_choice="fixed_default_material",
            texture_choice="fixed_default_texture",
            randomizer_internal_choices={
                "global_rgbd_camera_only": True,
                "camera_names": ["global_cam"],
                "depth_saved": bool(self.save_depth),
                "object_bowl_randomization": pose_info,
                "occlusion": occluder_info,
            },
        )
        _safe_json_dump(os.path.join(sample_dir, "variables.json"), asdict(self.current_episode_variables))
        _safe_json_dump(os.path.join(sample_dir, "scene_snapshot.json"), asdict(self._capture_scene_snapshot()))
        self.camera_encoders = {}
        for cam_name, cfg in self.camera_cfgs.items():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) >= 0:
                self.camera_encoders[cam_name] = PyavImageEncoder(cfg["width"], cfg["height"], self.save_dir, cam_name)
        if self.save_depth_vis:
            os.makedirs(os.path.join(self.save_dir, "depth_vis_frames"), exist_ok=True)

    def run(self):
        step_count = 0
        obs_lst = []
        depth_lst = []
        depth_time_lst = []
        last_render_time = 0.0
        recorded_frame_idx = 0
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
                imgs = obs.pop("img")
                depths = obs.pop("depth", {})
                for cam_id, img in imgs.items():
                    if cam_id in self.camera_encoders:
                        self.camera_encoders[cam_id].encode(img, obs["time"])
                if self.save_depth and "global_cam" in depths:
                    depth = depths["global_cam"].astype(np.float32)
                    depth_lst.append(depth)
                    depth_time_lst.append(float(obs["time"]))
                    if self.save_depth_vis:
                        try:
                            import cv2
                            depth_vis = self.make_depth_visualization(depth)
                            path = os.path.join(self.save_dir, "depth_vis_frames", f"depth_{recorded_frame_idx:06d}.png")
                            cv2.imwrite(path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
                        except Exception:
                            pass
                obs_lst.append(obs)
                recorded_frame_idx += 1
        for ec in self.camera_encoders.values():
            ec.close()
        if self.save_depth and len(depth_lst) > 0:
            np.savez_compressed(
                os.path.join(self.save_dir, "global_cam_depth.npz"),
                depth=np.stack(depth_lst, axis=0).astype(np.float32),
                time=np.asarray(depth_time_lst, dtype=np.float32),
                camera_name=np.array(["global_cam"]),
                depth_unit=np.array(["mujoco_depth"]),
                near_clip=np.array([self.depth_near_clip], dtype=np.float32),
                far_clip=np.array([self.depth_far_clip], dtype=np.float32),
            )
        episode_meta = {
            "success": bool(self.success),
            "step_count": int(step_count),
            "sample_idx": int(self.current_sample_idx),
            "episode_seed": int(self.last_episode_seed),
            "episode_variables": None if self.current_episode_variables is None else asdict(self.current_episode_variables),
            "num_recorded_frames": int(len(obs_lst)),
            "num_depth_frames": int(len(depth_lst)),
            "rgb_video": "global_cam.mp4",
            "depth_file": "global_cam_depth.npz" if self.save_depth else None,
            "kept_even_if_failed": bool(self.keep_failed),
        }
        _safe_json_dump(os.path.join(self.save_dir, "episode_metadata.json"), episode_meta)
        if self.success or self.keep_failed:
            recoder_single_arm(self.save_dir, obs_lst)
            if not self.success:
                with open(os.path.join(self.save_dir, "FAILED_EPISODE.txt"), "w", encoding="utf-8") as f:
                    f.write("This episode failed task success check, but was kept for debugging.\n")
                print(f"⚠️ Task failed, but debug data kept: {self.save_dir}")
            else:
                print(f"✅ RGBD object-occlusion GT 데이터 저장 완료: {self.save_dir}")
        else:
            print(f"❌ Task failed. 실패한 데이터 삭제: {self.save_dir}")
            shutil.rmtree(self.save_dir, ignore_errors=True)
        return bool(self.success)


def create_simple_visualizer(mj_model, mj_data):
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    if mj_model.ncam > 0:
        global_cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "global_cam")
        viewer.cam.fixedcamid = global_cam_id if global_cam_id >= 0 else 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return viewer


def main(
    robot_name="piper",
    task_name="place_block",
    sync=False,
    once=False,
    headless=False,
    num_samples=500,
    base_seed=1000,
    object_name="block_green",
    bowl_name="bowl_pink",
    object_xy_range=0.08,
    bowl_xy_range=0.08,
    min_object_bowl_dist=0.06,
    output_root=None,
    occluder_enabled=True,
    occluder_type="can",
    occluder_visual_only=True,
    occluder_alpha_min=0.35,
    occluder_alpha_max=0.55,
    occluder_xy_jitter=0.08,
    occluder_z_jitter=0.10,
    occluder_allow_lying=False,
    save_depth=True,
    save_depth_vis=True,
    keep_failed=True,
):
    if occluder_type not in ["can", "bottle", "block", "box"]:
        raise ValueError("--occluder-type must be one of: can, bottle, block, box")

    print(discoverse.__logo__)
    xml_path = generate_robot_task_model(robot_name, task_name, add_occluder=occluder_enabled, occluder_type=occluder_type)
    print(f"📄 Generated MJCF: {xml_path}")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    viewer = None if headless else create_simple_visualizer(mj_model, mj_data)
    configs_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
    task = UniversalTaskBase(
        robot_config_path=os.path.join(configs_root, "robots", f"{robot_name}.yaml"),
        task_config_path=os.path.join(configs_root, "tasks", f"{task_name}.yaml"),
        mj_model=mj_model,
        mj_data=mj_data,
    )
    if output_root is None:
        output_root = os.path.join(DISCOVERSE_ROOT_DIR, "data", f"{robot_name}_{task_name}_global_rgbd_{occluder_type}_occlusion_gt_data")
    os.makedirs(output_root, exist_ok=True)
    try:
        executor = GlobalRGBDObjectOcclusionGTExecutor(
            task=task,
            viewer=viewer,
            mj_model=mj_model,
            mj_data=mj_data,
            robot_name=robot_name,
            task_name=task_name,
            output_root=output_root,
            sync=sync,
            object_name=object_name,
            bowl_name=bowl_name,
            object_xy_range=object_xy_range,
            bowl_xy_range=bowl_xy_range,
            min_object_bowl_dist=min_object_bowl_dist,
            occluder_type=occluder_type,
            occluder_enabled=occluder_enabled,
            occluder_visual_only=occluder_visual_only,
            occluder_alpha_min=occluder_alpha_min,
            occluder_alpha_max=occluder_alpha_max,
            occluder_xy_jitter=occluder_xy_jitter,
            occluder_z_jitter=occluder_z_jitter,
            occluder_allow_lying=occluder_allow_lying,
            save_depth=save_depth,
            save_depth_vis=save_depth_vis,
            keep_failed=keep_failed,
        )
        executor.build_base_scene_once()
        _safe_json_dump(os.path.join(output_root, "dataset_manifest.json"), {
            "robot_name": robot_name,
            "task_name": task_name,
            "num_samples_target": int(num_samples),
            "base_seed": int(base_seed),
            "data_type": "gt_pick_and_place_global_rgbd_object_occlusion",
            "camera": {"camera_names": ["global_cam"], "rgb_saved_as_video": True, "depth_saved_as_npz": bool(save_depth), "width": 640, "height": 480},
            "occlusion": {
                "enabled": bool(occluder_enabled),
                "type": occluder_type,
                "visual_only": bool(occluder_visual_only),
                "static_body_no_gravity": True,
                "allow_lying": bool(occluder_allow_lying),
                "alpha_min": float(occluder_alpha_min),
                "alpha_max": float(occluder_alpha_max),
                "xy_jitter": float(occluder_xy_jitter),
                "z_jitter": float(occluder_z_jitter),
            },
            "keep_failed": bool(keep_failed),
        })
        success_count = 0
        attempt_count = 0
        while success_count < num_samples:
            attempt_count += 1
            episode_seed = base_seed + attempt_count
            print("\n" + "=" * 80)
            print(f"📦 Global RGBD {occluder_type} occlusion GT | attempt={attempt_count} | saved_success={success_count}/{num_samples} | seed={episode_seed}")
            print("=" * 80)
            executor.prepare_episode(sample_idx=attempt_count - 1, episode_seed=episode_seed)
            success = executor.run()
            if success:
                success_count += 1
                print(f"🎉 성공 데이터 저장 완료 ({success_count}/{num_samples})")
            if once or executor.viewer_closed:
                break
        print("\n✅ 데이터 생성 루프 종료")
        print(f" - saved success episodes: {success_count}/{num_samples}")
        print(f" - output_root: {output_root}")
    except Exception:
        traceback.print_exc()
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global RGBD camera 1개와 static object occluder(can/bottle/block/box)를 사용해 pick-and-place GT 데이터를 생성합니다.")
    parser.add_argument("-r", "--robot", type=str, default="piper")
    parser.add_argument("-t", "--task", type=str, default="place_block")
    parser.add_argument("-s", "--sync", action="store_true")
    parser.add_argument("-1", "--once", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--object-name", type=str, default="block_green")
    parser.add_argument("--bowl-name", type=str, default="bowl_pink")
    parser.add_argument("--object-xy-range", type=float, default=0.08)
    parser.add_argument("--bowl-xy-range", type=float, default=0.08)
    parser.add_argument("--min-object-bowl-dist", type=float, default=0.06)
    parser.add_argument("--disable-occluder", action="store_true")
    parser.add_argument("--occluder-type", type=str, default="can", choices=["can", "bottle", "block", "box"])
    parser.add_argument("--occluder-physical", action="store_true", help="현재 occluder는 static body이므로 기본은 visual-only입니다. 이 옵션은 geom contact bit만 켭니다.")
    parser.add_argument("--occluder-alpha-min", type=float, default=0.35)
    parser.add_argument("--occluder-alpha-max", type=float, default=0.55)
    parser.add_argument("--occluder-xy-jitter", type=float, default=0.08)
    parser.add_argument("--occluder-z-jitter", type=float, default=0.10)
    parser.add_argument("--occluder-allow-lying", action="store_true", help="can/bottle이 일정 확률로 누운 자세가 되도록 합니다.")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-depth-vis", action="store_true")
    parser.add_argument("--delete-failed", action="store_true", help="실패 episode를 저장하지 않고 삭제합니다. 기본값은 실패 episode도 디버깅용으로 보존합니다.")
    args = parser.parse_args()
    main(
        robot_name=args.robot,
        task_name=args.task,
        sync=args.sync,
        once=args.once,
        headless=args.headless,
        num_samples=args.num_samples,
        base_seed=args.base_seed,
        output_root=args.output_root,
        object_name=args.object_name,
        bowl_name=args.bowl_name,
        object_xy_range=args.object_xy_range,
        bowl_xy_range=args.bowl_xy_range,
        min_object_bowl_dist=args.min_object_bowl_dist,
        occluder_enabled=not args.disable_occluder,
        occluder_type=args.occluder_type,
        occluder_visual_only=not args.occluder_physical,
        occluder_alpha_min=args.occluder_alpha_min,
        occluder_alpha_max=args.occluder_alpha_max,
        occluder_xy_jitter=args.occluder_xy_jitter,
        occluder_z_jitter=args.occluder_z_jitter,
        occluder_allow_lying=args.occluder_allow_lying,
        save_depth=not args.no_depth,
        save_depth_vis=not args.no_depth_vis,
        keep_failed=not args.delete_failed,
    )
