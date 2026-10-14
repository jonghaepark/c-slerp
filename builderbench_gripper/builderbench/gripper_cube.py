import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax
from etils import epath
from ml_collections import config_dict
from mujoco.mjx._src import math as mjx_math

from builderbench.constants import _CUSTOM_COLORS
from builderbench.env_utils import State, mjx_make_data, mjx_step_data
from builderbench.goal_layouts import build_goal_layout

_ARM_JOINTS = [
    "arm_tx",
    "arm_ty",
    "arm_tz",
    "arm_yaw",
]


@dataclass(frozen=True)
class Dof:
    """Forearm degree of freedom."""

    joint_type: int
    axis: Tuple[int, int, int]
    stiffness: float
    range: Tuple[float, float]
    transmission: int


_FOREARM_DOFS: Dict[str, Dof] = {
    "arm_tx": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0, 1, 0),
        stiffness=300,
        range=(-0.05, 0.45),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_ty": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(1, 0, 0),
        stiffness=300,
        range=(-0.35, 0.35),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_tz": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0, 0, -1),
        stiffness=300,
        range=(0.07, 0.5),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_yaw": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0, 0, -1),
        stiffness=300,
        range=(-1.5707, 1.5707),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
}


def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        nconmax=16 * 1024,
        njmax=32,
        num_cubes=1,
        action_scale=config_dict.create(
            finger_scale=255.0,
            xyz_scale=0.0075,
            yaw_scale=0.05,
        ),
        delta_control=True,
        episode_length=150,
        task_id=0,
        reward_sensitivity=5.0,
        success_threshold=0.02,
        easy_success_threshold=0.05,
        stacking_bottom_xy_threshold=0.04,
        stacking_z_threshold=0.01,
        env_early_termination=True,
        permutation_invariant_reward=False,
        permute_init_at_reset=True,
        goal_noise_at_reset=True,
        # Procedural structure spacing (empty goal_layout → static npz goals)
        goal_layout="",
        randomize_structure_spacing=False,
        structure_spacing_scale_min=1.5,
        structure_spacing_scale_max=1.5,
        cube_size=0.04,
        impl="warp",
    )
    return config


def _load_assets(xml_dir: epath.Path) -> Dict[str, bytes]:
    assets: Dict[str, bytes] = {}
    for f in xml_dir.glob("*.xml"):
        if f.is_file():
            assets[f.name] = f.read_bytes()
    assets_dir = xml_dir / "assets"
    for f in assets_dir.glob("*"):
        if f.is_file():
            assets[f.name] = f.read_bytes()
    return assets


class GripperCube:
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        xml_path = (
            epath.Path(__file__).resolve().parent
            / "xmls"
            / "gripper_scene.xml"
        )
        self._init(xml_path=xml_path, config=config)

        self._config = config
        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._episode_length = config.episode_length

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))

    @property
    def hybrid_action(self) -> bool:
        return False

    @property
    def force_dim(self) -> int:
        """Continuous control dims for SAC entropy scaling."""
        return 5

    @property
    def num_cubes(self) -> int:
        return self._config.num_cubes

    @property
    def action_size(self) -> int:
        return 5

    @property
    def observation_size(self) -> int:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return obs.shape[-1]

    @property
    def goal_size(self) -> int:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        goal = abstract_state.info["achieved_goal"]
        return goal.shape[-1]

    @property
    def unwrapped(self):
        return self

    def _init(self, xml_path, config):
        xml_dir = epath.Path(xml_path).parent
        spec = self._prepare_spec(xml_path, config, xml_dir)
        spec, self._object_names = self._add_objects(spec, config.num_cubes)
        self._spec = spec

        self._mj_model = spec.compile()
        self._apply_grasp_tuning(self._mj_model, config)
        self._settle_init_state(self._mj_model, config)

        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)

        self._sensor_dim = self._mj_model.nsensordata
        self._pstate_dim = mujoco.mj_stateSize(
            self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        self._qpos_dim = self._mj_model.nq
        self._qvel_dim = self._mj_model.nv
        self._ctrl_dim = self._mj_model.nu

        self._mocap_targets = np.array(
            [
                self._mj_model.body(f"target_mocap_{i}").mocapid[0]
                for i in range(config.num_cubes)
            ]
        )
        self._mocap_targets_geom = np.array(
            [
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"target_mocap_{i}"
                )
                for i in range(config.num_cubes)
            ]
        )

        self._objs_ids = np.array(
            [self._mj_model.body(obj_name).id for obj_name in self._object_names]
        )

        self._objs_qposadr = np.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.body(obj_name).jntadr[0]]
                for obj_name in self._object_names
            ]
        )
        self._objs_qveladr = np.array(
            [
                self._mj_model.jnt_dofadr[self._mj_model.body(obj_name).jntadr[0]]
                for obj_name in self._object_names
            ]
        )

        self._objs_pos_qpos_idxs = np.concatenate(
            [obj_adr + np.arange(3) for obj_adr in self._objs_qposadr]
        )
        self._objs_quat_qpos_idxs = np.concatenate(
            [obj_adr + 3 + np.arange(4) for obj_adr in self._objs_qposadr]
        )

        self._fingers_qposadr = np.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(joint_name).id]
                for joint_name in ["left_driver_joint", "right_driver_joint"]
            ]
        )

        self._gripper_pos_sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framepos_sensor"
        )
        self._gripper_quat_sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framequat_sensor"
        )
        self._gripper_linvel_sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framelinvel_sensor"
        )

        self._gripper_pos_sensor_adr = self._mj_model.sensor_adr[
            self._gripper_pos_sensor_id
        ]
        self._gripper_pos_sensor_dim = self._mj_model.sensor_dim[
            self._gripper_pos_sensor_id
        ]
        self._gripper_quat_sensor_adr = self._mj_model.sensor_adr[
            self._gripper_quat_sensor_id
        ]
        self._gripper_quat_sensor_dim = self._mj_model.sensor_dim[
            self._gripper_quat_sensor_id
        ]
        self._gripper_linvel_sensor_adr = self._mj_model.sensor_adr[
            self._gripper_linvel_sensor_id
        ]
        self._gripper_linvel_sensor_dim = self._mj_model.sensor_dim[
            self._gripper_linvel_sensor_id
        ]

        self._init_q = jnp.array(self._init_qpos, dtype=jnp.float32)
        self._init_v = jnp.array(self._init_qvel, dtype=jnp.float32)
        self._init_ctrl = jnp.array(self._init_ctrl_np, dtype=jnp.float32)
        self._init_gripper_quat_inverse = jnp.array([0.0, 0.0, -1.0, 0.0])

        self._action_scale = np.array(
            [
                config.action_scale.finger_scale,
                config.action_scale.xyz_scale,
                config.action_scale.xyz_scale,
                config.action_scale.xyz_scale,
                config.action_scale.yaw_scale,
            ]
        )

        self._workspace_bounds = jnp.array(
            [[-0.05, -0.35, 0.00], [0.45, 0.35, 0.5]]
        )
        self._target_sampling_bounds = jnp.array(
            [[0.22, -0.1, 0.02], [0.32, +0.1, 0.02]]
        )
        self._ctrl_bounds = jnp.array(self._mj_model.actuator_ctrlrange.T)
        self._ctrl_median = (self._ctrl_bounds[1] + self._ctrl_bounds[0]) / 2
        self._ctrl_halfspan = (self._ctrl_bounds[1] - self._ctrl_bounds[0]) / 2

        task_data = np.load(
            epath.Path(__file__).resolve().parent
            / f"tasks/creative-{config.num_cubes}.npz"
        )
        self._starts_data = jnp.array(task_data["starts"][config.task_id])
        self._target_goal_data = jnp.array(task_data["goals"][config.task_id])

    def _prepare_spec(self, xml_path, config, xml_dir):
        spec = mujoco.MjSpec.from_file(str(xml_path), assets=_load_assets(xml_dir))

        spec.option.timestep = config.sim_dt

        base_body = spec.body("base_mount")
        base_body.pos = np.array([0.0, 0.0, 0.12])
        base_body.quat = np.array([0.0, 0.707106, 0.707106, 0.0])

        for name, dof in _FOREARM_DOFS.items():
            base_body.add_joint(
                name=name,
                type=dof.joint_type,
                axis=dof.axis,
                range=dof.range,
            )
            act = spec.add_actuator(
                name=name,
                target=name,
                trntype=dof.transmission,
                ctrlrange=dof.range,
            )
            act.set_to_position(kp=dof.stiffness)

        spec.stat.center = np.array([0.4, 0.0, 0.4])
        spec.stat.extent = 0.8
        spec.visual.global_.elevation = -30.0
        spec.visual.global_.azimuth = 180

        return spec

    def _add_objects(self, spec, num_cubes):
        object_names = []
        for i in range(num_cubes):
            x = 0.1
            y = 0.0
            z = 0.02 * (2 * i + 1)

            yaw = np.random.uniform(-1, 1) * np.pi
            quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

            body = spec.worldbody.add_body(
                name=f"block_{i}",
                pos=[x, y, z],
                quat=quat,
            )
            body.add_joint(
                name=f"block_joint_{i}",
                type=mujoco.mjtJoint.mjJNT_FREE,
                # Free-joint damping applies to all 6 DOFs. The previous 0.1
                # overdamps roll/pitch on these light cubes (I ~ 2e-5).
                damping=0.005,
                armature=0.001,
            )
            body.add_geom(
                name=f"block_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                contype=3,
                conaffinity=1,
                condim=6,
                solref=[0.01, 1],
                size=[0.02, 0.02, 0.02],
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)],
                density=400,
                friction=[1, 0.005, 0.0001],
            )

            body = spec.worldbody.add_body(
                name=f"target_mocap_{i}",
                mocap=True,
                pos=[x, y, z],
                quat=quat,
            )
            body.add_geom(
                name=f"target_mocap_{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)]
                - np.array([0.0, 0.0, 0.0, 0.8]),
                contype=0,
                conaffinity=0,
            )

            object_names.append(f"block_{i}")

        return spec, object_names

    def _apply_grasp_tuning(self, model: mujoco.MjModel, config):
        model.opt.timestep = config.sim_dt
        model.opt.impratio = 50
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0

        for name in ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            model.geom_friction[gid] = [10, 0.01, 0.0001]
            model.geom_condim[gid] = 6

        for i in range(config.num_cubes):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{i}")
            model.geom_friction[gid] = [1, 0.005, 0.0001]
            model.geom_condim[gid] = 6

        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        model.actuator_forcerange[aid] = [-20, 20]

    def _settle_init_state(self, model: mujoco.MjModel, config):
        data = mujoco.MjData(model)
        data.qpos[:12] = [
            0,
            0,
            0.07,
            0,
            0.0029,
            0.00044,
            0.0053,
            -0.0077,
            0.0029,
            0.00044,
            0.0053,
            -0.0077,
        ]
        data.ctrl = np.array([0, 0, 0, 0.07, 0], dtype=np.float64)
        mujoco.mj_step(
            model, data, nstep=int(round(config.ctrl_dt / config.sim_dt))
        )

        for jn in _ARM_JOINTS:
            ji = model.joint(jn).id
            joint_mass = model.dof_M0[model.jnt_dofadr[ji]]
            stiffness = model.actuator(jn).gainprm[0]
            model.joint(jn).damping = 2 * math.sqrt(joint_mass * stiffness)

        self._init_qpos = np.array(data.qpos, dtype=np.float32)
        self._init_qvel = np.zeros(model.nv, dtype=np.float32)
        self._init_ctrl_np = np.array(data.ctrl, dtype=np.float32)

    def reset(self, rng: jax.Array):
        rng, rng_box, rng_target, rng_starts, rng_spacing = jax.random.split(rng, 5)

        _starts_data = (
            jax.random.permutation(rng_starts, self._starts_data, axis=0)
            if self._config.permute_init_at_reset
            else self._starts_data
        )

        object_pos = jax.random.uniform(
            rng_box,
            (self._config.num_cubes, 3),
            minval=_starts_data[:, 0],
            maxval=_starts_data[:, 1],
        )
        achieved_goal = object_pos.reshape(-1)
        object_pos = object_pos.reshape(-1)
        object_quat = jnp.tile(
            jnp.array([1, 0, 0, 0], dtype=jnp.float32),
            (self._config.num_cubes, 1),
        ).reshape(-1)

        # Procedural structure (pyramid) or static npz goals
        if self._config.goal_layout and self._config.randomize_structure_spacing:
            scale = jax.random.uniform(
                rng_spacing,
                (),
                minval=self._config.structure_spacing_scale_min,
                maxval=self._config.structure_spacing_scale_max,
            )
            s = self._config.cube_size * scale
            target_goal_data = build_goal_layout(
                self._config.goal_layout, s, z_step=self._config.cube_size
            )
        else:
            target_goal_data = self._target_goal_data

        target_pos = jax.random.uniform(
            rng_target,
            (1, 3),
            minval=self._target_sampling_bounds[0],
            maxval=self._target_sampling_bounds[1],
        )
        target_object_pos = (
            (target_pos + target_goal_data)
            if self._config.goal_noise_at_reset
            else target_goal_data
        )
        target_object_quat = jnp.tile(
            jnp.array([1, 0, 0, 0], dtype=jnp.float32),
            (self._config.num_cubes, 1),
        )

        init_q = self._init_q.at[self._objs_pos_qpos_idxs].set(object_pos)
        init_q = init_q.at[self._objs_quat_qpos_idxs].set(object_quat)

        data = mjx_make_data(
            self._mj_model,
            qpos=init_q,
            qvel=self._init_v,
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_targets].set(target_object_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_targets].set(target_object_quat),
        )
        data = mjx.forward(self._mjx_model, data)

        metrics = {
            "easy_success": jnp.array(0.0, dtype=float),
            "success": jnp.array(0.0, dtype=float),
            "stacking": jnp.array(0.0, dtype=float),
            "out_of_bounds": jnp.array(0.0, dtype=float),
            "obj_lifted": jnp.array(0.0, dtype=float),
            "obj_moved": jnp.array(0.0, dtype=float),
            "obj_goal_dist": jnp.array(0.0, dtype=float),
        }
        info = {
            "rng": rng,
            "achieved_goal": achieved_goal,
            "target_goal": target_object_pos.reshape(-1),
            "target_mocap_pos": target_object_pos,
            "target_mocap_quat": target_object_quat,
        }

        obs, info = self.get_obs(data, info)

        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(
                data, info
            )
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(
                data, info
            )
        done, out_of_bounds = self.get_termination(data)
        metrics.update(
            out_of_bounds=out_of_bounds.astype(float),
            **reward_info,
        )

        state = State(data, obs, reward, done, metrics, info)
        return state

    def get_obs(self, data, info):
        sensor_data = data.sensordata

        gripper_pos = sensor_data[
            self._gripper_pos_sensor_adr : self._gripper_pos_sensor_adr
            + self._gripper_pos_sensor_dim
        ]
        gripper_quat = sensor_data[
            self._gripper_quat_sensor_adr : self._gripper_quat_sensor_adr
            + self._gripper_quat_sensor_dim
        ]
        gripper_quat = mjx_math.quat_mul(
            gripper_quat, self._init_gripper_quat_inverse
        )
        gripper_linvel = sensor_data[
            self._gripper_linvel_sensor_adr : self._gripper_linvel_sensor_adr
            + self._gripper_linvel_sensor_dim
        ]

        finger_pos = data.qpos[self._fingers_qposadr]

        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos.reshape(-1)
        obj_pos = obj_pos.reshape(-1)
        obj_quat = data.qpos[
            (self._objs_qposadr + 3)[:, None] + np.arange(4)
        ].reshape(-1)
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)
        obj_angvel = data.qvel[
            (self._objs_qveladr + 3)[:, None] + np.arange(3)
        ].reshape(-1)

        obs = jnp.concatenate(
            [
                gripper_pos,
                gripper_quat,
                gripper_linvel,
                finger_pos,
                obj_pos,
                obj_quat,
                obj_linvel,
                obj_angvel,
            ]
        )

        info.update({"achieved_goal": achieved_goal})
        return obs, info

    def _stacking_metric(self, obj_pos, target_goal):
        bottom_idx = jnp.argmin(target_goal[:, 2])
        bottom_xy_err = jnp.linalg.norm(
            obj_pos[bottom_idx, :2] - target_goal[bottom_idx, :2]
        )
        bottom_ok = bottom_xy_err < self._config.stacking_bottom_xy_threshold
        z_err = jnp.abs(obj_pos[:, 2] - target_goal[:, 2])
        return (
            bottom_ok & jnp.all(z_err < self._config.stacking_z_threshold)
        ).astype(float)

    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum(
            (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1
        )
        cube_ids, target_ids = optax.assignment.hungarian_algorithm(
            obj_target_pos_squared_pairwise_err
        )
        obj_target_pos_err = jnp.sqrt(
            obj_target_pos_squared_pairwise_err[cube_ids, target_ids]
        )

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        reward = jnp.sum(
            1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)
        )

        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info

    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        reward = jnp.sum(
            1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)
        )

        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info

    def get_termination(self, data):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]

        out_of_bounds = 1 - (
            jnp.all(
                (obj_pos >= self._workspace_bounds[0])
                & jnp.all(obj_pos <= self._workspace_bounds[1])
            )
        )
        termination = (
            out_of_bounds
            | jnp.isnan(data.qpos).any()
            | jnp.isnan(data.qvel).any()
        )
        termination = (termination * self._config.env_early_termination).astype(float)
        termination = termination.astype(float)
        return termination, out_of_bounds

    def step(self, state, action):
        if self._config.delta_control:
            new_ctrl = state.data.ctrl + action * self._action_scale
            new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])
        else:
            new_ctrl = action * self._ctrl_halfspan + self._ctrl_median
            new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])

        data = mjx_step_data(
            self._mjx_model, state.data, new_ctrl, self.n_substeps
        )

        obs, info = self.get_obs(data, state.info)
        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(
                data, info
            )
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(
                data, info
            )

        done, out_of_bounds = self.get_termination(data)
        state.metrics.update(
            out_of_bounds=out_of_bounds.astype(float),
            **reward_info,
        )

        state = State(data, obs, reward, done, state.metrics, state.info)
        return state

    def render_from_info(
        self,
        qpos,
        qvel,
        mocap_pos,
        mocap_quat,
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
    ):
        renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
        camera = camera or -1

        def get_image(qpos, qvel, mocap_pos, mocap_quat) -> np.ndarray:
            d = mujoco.MjData(self._mj_model)
            d.qpos, d.qvel = qpos, qvel
            d.mocap_pos[self._mocap_targets], d.mocap_quat[
                self._mocap_targets
            ] = mocap_pos.reshape(self._config.num_cubes, 3), mocap_quat.reshape(
                self._config.num_cubes, 4
            )
            mujoco.mj_forward(self._mj_model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            return renderer.render()

        out = get_image(qpos, qvel, mocap_pos, mocap_quat)
        renderer.close()
        return out


class TruncatedRewardGripperCube(GripperCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        super().__init__(config=config)

    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum(
            (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1
        )
        cube_ids, target_ids = optax.assignment.hungarian_algorithm(
            obj_target_pos_squared_pairwise_err
        )
        obj_target_pos_err = jnp.sqrt(
            obj_target_pos_squared_pairwise_err[cube_ids, target_ids]
        )

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        reward = jnp.sum(
            1
            - jnp.tanh(
                self._config.reward_sensitivity
                * jnp.clip(obj_target_pos_err, min=self._config.success_threshold)
            )
        ).astype(float)

        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info

    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        reward = jnp.sum(
            1
            - jnp.tanh(
                self._config.reward_sensitivity
                * jnp.clip(obj_target_pos_err, min=self._config.success_threshold)
            )
        ).astype(float)

        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info


class SparseGripperCube(GripperCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        super().__init__(config=config)
        assert (
            self._config.env_early_termination is False
        ), "sparse gripper env requires env_early_termination to be False"

    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum(
            (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1
        )
        cube_ids, target_ids = optax.assignment.hungarian_algorithm(
            obj_target_pos_squared_pairwise_err
        )
        obj_target_pos_err = jnp.sqrt(
            obj_target_pos_squared_pairwise_err[cube_ids, target_ids]
        )

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        dense_reward = jnp.sum(
            1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)
        ).astype(float)
        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)
        reward = (
            jnp.sum(obj_target_pos_err < self._config.success_threshold).astype(float)
            - self._config.num_cubes
        )

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "dense_reward": dense_reward,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info

    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[
            self._objs_qveladr[:, None] + np.arange(3)
        ].reshape(-1)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum(obj_pos[:, 2] > 0.05).astype(float)
        obj_moved = jnp.any(obj_linvel > 0.001).astype(float)

        dense_reward = jnp.sum(
            1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)
        ).astype(float)
        success = jnp.all(
            obj_target_pos_err < self._config.success_threshold
        ).astype(float)
        easy_success = jnp.all(
            obj_target_pos_err < self._config.easy_success_threshold
        ).astype(float)
        reward = (
            jnp.sum(obj_target_pos_err < self._config.success_threshold).astype(float)
            - self._config.num_cubes
        )

        reward_info = {
            "success": success,
            "easy_success": easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "dense_reward": dense_reward,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum(obj_target_pos_err),
        }

        return reward, reward_info

    def get_permutation_invariant_reward_from_goals(self, achieved_goal, target_goal):
        obj_target_pos_squared_pairwise_err = jnp.sum(
            (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1
        )
        cube_ids, target_ids = optax.assignment.hungarian_algorithm(
            obj_target_pos_squared_pairwise_err
        )
        obj_target_pos_err = jnp.sqrt(
            obj_target_pos_squared_pairwise_err[cube_ids, target_ids]
        )
        reward = (
            jnp.sum(obj_target_pos_err < self._config.success_threshold).astype(float)
            - self._config.num_cubes
        )
        return reward

    def get_permutation_variant_reward_from_goals(self, achieved_goal, target_goal):
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)
        reward = (
            jnp.sum(obj_target_pos_err < self._config.success_threshold).astype(float)
            - self._config.num_cubes
        )
        return reward

    def get_reward_from_goals(self, achieved_goal, target_goal):
        achieved_goal = achieved_goal.reshape((self._config.num_cubes, -1))
        target_goal = target_goal.reshape((self._config.num_cubes, -1))
        if self._config.permutation_invariant_reward:
            reward = self.get_permutation_invariant_reward_from_goals(
                achieved_goal, target_goal
            )
        else:
            reward = self.get_permutation_variant_reward_from_goals(
                achieved_goal, target_goal
            )
        return reward
