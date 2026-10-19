import jax
import jax.numpy as jnp
import optax
import numpy as np
import mujoco
import mujoco.mjx as mjx
from mujoco.mjx._src import math as mjx_math

from etils import epath
from ml_collections import config_dict
from typing import Optional

from builderbench.env_utils import State, mjx_make_data, mjx_step_data
from builderbench.constants import _CUSTOM_COLORS

def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        nconmax=16 * 1024, 
        njmax=32,
        num_cubes=1,
        action_scale = config_dict.create(
            xy_scale = 0.1,
            z_scale = 0.1,
            yaw_scale = 0.1,
            select_scale = np.pi,
            ),
        delta_control=False,
        episode_length=150,
        task_id=0,
        reward_sensitivity=5.0,
        success_threshold=0.02,
        easy_success_threshold=0.05,
        stacking_bottom_xy_threshold=0.04,
        stacking_z_threshold=0.01,
        env_early_termination=True,
        permutation_invariant_reward=False,
        permute_init_at_reset=True,   # shuffle _starts_data cube order each reset
        goal_noise_at_reset=True,     # add uniform target_pos noise to _target_goal_data
        impl='warp'
    )
    return config

class CreativeCube():
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        
        xml_path = (
            epath.Path(__file__).resolve().parent
            / "xmls"
            / "creative_scene.xml"
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
    def hybrid_action(self):
        """Force + one-hot cube index (vs continuous-only gripper)."""
        return True

    @property
    def force_dim(self):
        """Continuous force / ctrl dims (x, y, z, yaw)."""
        return 4

    @property
    def num_cubes(self):
        return self._config.num_cubes

    @property
    def action_size(self):
        """Force dims + one-hot block selection."""
        return self.force_dim + self._config.num_cubes
    
    @property
    def observation_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return obs.shape[-1]
    
    @property
    def goal_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        goal = abstract_state.info["achieved_goal"]
        return goal.shape[-1]

    @property
    def unwrapped(self):
        return self

    def _init(self, xml_path, config):

        # prepare spec and add objects to the spec
        spec = self._prepare_spec(xml_path, config)
        spec, self._object_names = self._add_objects(spec, config.num_cubes)
        self._spec = spec

        # compile spec and create mujoco model and data
        self._mj_model = self._spec.compile()
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)

        # get dimensions
        self._sensor_dim = self._mj_model.nsensordata
        self._pstate_dim = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._qpos_dim = self._mj_model.nq
        self._qvel_dim = self._mj_model.nv
        self._ctrl_dim = self._mj_model.nu

        # get mocap ids
        self._mocap_targets = np.array([
            self._mj_model.body( f"target_mocap_{i}" ).mocapid[0]
            for i in range( config.num_cubes )
        ])
        self._mocap_targets_geom = np.array([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"target_mocap_{i}")
            for i in range( config.num_cubes )
        ])

        # get object ids
        self._objs_ids = np.array([
            self._mj_model.body(obj_name).id
            for obj_name in self._object_names
        ])

        # get start indices in qpos and qvel
        self._objs_qposadr = np.array([
            self._mj_model.jnt_qposadr[ self._mj_model.body(obj_name).jntadr[0] ]
            for obj_name in self._object_names
        ])
        self._objs_qveladr = np.array([
            self._mj_model.jnt_dofadr[self._mj_model.body(obj_name).jntadr[0]]
            for obj_name in self._object_names
        ])

        # get object pos and quat indices in qpos
        self._objs_pos_qpos_idxs = np.concatenate([
            obj_adr + np.arange(3)
            for obj_adr in self._objs_qposadr
        ])
        self._objs_quat_qpos_idxs = np.concatenate([
            obj_adr + 3 + np.arange(4)
            for obj_adr in self._objs_qposadr
        ])

        # get start indices in actuators
        self._objs_actuator_adr = np.stack([
            np.array([
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in [f"x_block_{i}", f"y_block_{i}", f"z_block_{i}", f"yaw_block_{i}"]
            ])
            for i in range(config.num_cubes)
        ])
        self._objs_actuator_adr = jnp.array(self._objs_actuator_adr)

        # get constants
        _data = mujoco.MjData(self._mj_model)
        _data.ctrl = np.array([0, 0, 0, 0] * config.num_cubes)

        # get constants
        self._init_q = jnp.array(_data.qpos, dtype=jnp.float32)
        self._init_v = jnp.array(_data.qvel, dtype=jnp.float32) * 0
        self._init_ctrl = jnp.array(_data.ctrl, dtype=jnp.float32)
        
        # set action scale
        self._action_scale =  np.array([config.action_scale.xy_scale]*2 + [config.action_scale.z_scale] + [config.action_scale.yaw_scale] + [config.action_scale.select_scale])

        # set bounds       
        self._workspace_bounds = jnp.array([ [-0.05, -0.35, 0.00], [ 0.45,  0.35, 0.5 ] ])
        self._target_sampling_bounds = jnp.array([ [0.22, -0.1, 0.02], [0.32, +0.1, 0.02] ])
        self._ctrl_bounds = jnp.array( self._mj_model.actuator_ctrlrange.T )
        self._ctrl_median = (self._ctrl_bounds[1] + self._ctrl_bounds[0]) / 2
        self._ctrl_halfspan = (self._ctrl_bounds[1] - self._ctrl_bounds[0]) / 2

        # get task data
        task_data = np.load( epath.Path(__file__).resolve().parent / f'tasks/creative-{config.num_cubes}.npz')
        self._starts_data = jnp.array( task_data['starts'][config.task_id] )
        self._target_goal_data = jnp.array( task_data['goals'][config.task_id] )

    def _add_objects(self, spec, num_cubes):
        object_names = []
        for i in range(num_cubes):

            # stacked placement
            x = 0.1
            y = 0.0
            z = 0.02 * (2*i + 1)

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
                damping=0.1, 
                armature=0.001,
            )
            body.add_geom(
                name=f"block_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                contype=3,   
                conaffinity=1,
                solref=[0.01, 1],
                size=[0.02, 0.02, 0.02],
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)],
                density=1240,
            )

            # adding actuators for each cube
            actuator_x = spec.add_actuator(
                name=f"x_block_{i}",
                target=f"block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[1, 0, 0, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=(-0.1, 0.1),
            )
            actuator_x.set_to_motor()
            
            actuator_y = spec.add_actuator(
                name=f"y_block_{i}",
                target=f"block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[0, 1, 0, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=(-0.1, 0.1),
            )
            actuator_y.set_to_motor()

            actuator_z = spec.add_actuator(
                name=f"z_block_{i}",
                target=f"block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[0, 0, 1, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=(0.0, 1.0),
            )
            actuator_z.set_to_motor()

            actuator_yaw = spec.add_actuator(
                name=f"yaw_block_{i}",
                target=f"block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[0, 0, 0, 0, 0, 1],
                ctrllimited=True,
                ctrlrange=(-0.1, 0.1),
            )
            actuator_yaw.set_to_motor()

            # adding target position for cube
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
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)] - np.array([0.0, 0.0, 0.0, 0.8]),
                contype=0,
                conaffinity=0,
            )

            object_names.append(f"block_{i}")

        return spec, object_names
    
    def _prepare_spec(self, xml_path, config):
        spec = mujoco.MjSpec.from_file(str(xml_path))

        spec.option.timestep = config.sim_dt
        
        spec.stat.center = np.array([0.4, 0.0 , 0.4])
        spec.stat.extent = 0.8
        spec.visual.global_.elevation = -30.0
        spec.visual.global_.azimuth = 180

        return spec
    
    def reset(self, rng: jax.Array):
        rng, rng_box, rng_target, rng_starts, rng_select_action = jax.random.split(rng, 5) # key split for box, target, starts, select_action

        _starts_data = (
            jax.random.permutation(rng_starts, self._starts_data, axis=0)
            if self._config.permute_init_at_reset
            else self._starts_data
        )

        object_pos = (
            jax.random.uniform(
                rng_box,
                (self._config.num_cubes, 3),
                minval=_starts_data[:, 0],
                maxval=_starts_data[:, 1],
            )
        ) # random initial position for each cube
        achieved_goal = object_pos.reshape(-1)
        object_pos = object_pos.reshape(-1)
        object_quat = jnp.tile(jnp.array([1,0,0,0], dtype=jnp.float32), (self._config.num_cubes, 1)).reshape(-1)

        target_pos = (
            jax.random.uniform(
                rng_target,
                (1, 3),
                minval=self._target_sampling_bounds[0],
                maxval=self._target_sampling_bounds[1],
            )
        ) #random noise around the target goal position
        target_object_pos = (
            (target_pos + self._target_goal_data)
            if self._config.goal_noise_at_reset
            else self._target_goal_data
        )
        target_object_quat = jnp.tile(jnp.array([1,0,0,0], dtype=jnp.float32), (self._config.num_cubes, 1)) # target cube is always upright

        # set initial object position
        init_q = (
            self._init_q            
            .at[self._objs_pos_qpos_idxs]
            .set(object_pos)
        )
        init_q = (
            init_q              
            .at[self._objs_quat_qpos_idxs]
            .set(object_quat)
        )

        # set initial position and velocity of all joints and initial control of actuators
        data = mjx_make_data(
            self._mj_model,
            qpos=init_q,
            qvel=self._init_v,
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # set target mocap pos and quat
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_targets].set(target_object_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_targets].set(target_object_quat),
        )

        metrics = {
            "easy_success": jnp.array(0.0, dtype=float),
            "success": jnp.array(0.0, dtype=float),
            "stacking": jnp.array(0.0, dtype=float),
            "out_of_bounds": jnp.array(0.0, dtype=float),
            "obj_lifted": jnp.array(0.0, dtype=float),
            "obj_moved": jnp.array(0.0, dtype=float),
            "obj_goal_dist": jnp.array(0.0, dtype=float),
        }
        del rng_select_action  # kept in split for RNG continuity; index is in the action
        info = {
            "rng": rng,
            "achieved_goal": achieved_goal,
            "target_goal": target_object_pos.reshape(-1),
            "target_mocap_pos": target_object_pos,
            "target_mocap_quat": target_object_quat,
        }

        # calculate observation
        obs = self.get_obs(data, info)[0]
        
        # initial rewards and done
        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(data, info)
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(data, info)
        done, out_of_bounds = self.get_termination(data)
        metrics.update(
            out_of_bounds=out_of_bounds.astype(float), 
            **reward_info,
        )
    
        state = State(data, obs, reward, done, metrics, info)
        return state
    
    def get_obs(self, data, info):

        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos.reshape(-1,)
        obj_pos = obj_pos.reshape(-1,)
        obj_quat = data.qpos[(self._objs_qposadr + 3)[:, None] + np.arange(4)].reshape(-1,)
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)
        obj_angvel = data.qvel[(self._objs_qveladr + 3)[:, None] + np.arange(3)].reshape(-1,)

        # Block index is part of the action (one-hot), not the observation.
        if self._config.delta_control:
            obs = jnp.concatenate([
                obj_pos,
                obj_quat,
                obj_linvel,
                obj_angvel,
                data.ctrl,
            ])
        else:
            obs = jnp.concatenate([
                obj_pos,
                obj_quat,
                obj_linvel,
                obj_angvel,
            ])

        info.update({
            "achieved_goal": achieved_goal, 
        })

        return obs, info
    
    def _stacking_metric(self, obj_pos, target_goal):
        # obj_pos, target_goal: (num_cubes, 3), strict index pairing (i <-> i)
        bottom_idx = jnp.argmin(target_goal[:, 2])
        bottom_xy_err = jnp.linalg.norm(
            obj_pos[bottom_idx, :2] - target_goal[bottom_idx, :2]
        )
        bottom_ok = bottom_xy_err < self._config.stacking_bottom_xy_threshold
        z_err = jnp.abs(obj_pos[:, 2] - target_goal[:, 2])
        return (bottom_ok & jnp.all(z_err < self._config.stacking_z_threshold)).astype(float)

    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info

    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info

    def get_termination(self, data):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
    
        out_of_bounds = 1 - ( jnp.all((obj_pos >= self._workspace_bounds[0]) & jnp.all(obj_pos <= self._workspace_bounds[1])) )
        termination = out_of_bounds | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        termination = ( termination * self._config.env_early_termination ).astype(float)
        termination = termination.astype(float)
        return termination, out_of_bounds
    
    def step(self, state, action):
        # Immediate selection: action = [forces(force_dim), one_hot(num_cubes)].
        force_dim = self.force_dim
        selected_cube_idx = jnp.argmax(action[force_dim:])
        forces = action[:force_dim]
        new_ctrl = state.data.ctrl * 0

        if self._config.delta_control:
            new_ctrl = (
                new_ctrl            
                .at[ self._objs_actuator_adr[selected_cube_idx] ] 
                .set( forces * self._action_scale[:force_dim] + state.data.ctrl[ self._objs_actuator_adr[selected_cube_idx] ] )
            )
            new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])
        
        else:
            new_ctrl = (
                new_ctrl            
                .at[ self._objs_actuator_adr[selected_cube_idx] ] 
                .set( forces * self._ctrl_halfspan[:force_dim] + self._ctrl_median[:force_dim] )
            )
            new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])
            
        data = mjx_step_data(self._mjx_model, state.data, new_ctrl, self.n_substeps)
        
        obs, info = self.get_obs(data, state.info)
        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(data, info)
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(data, info)

        done, out_of_bounds = self.get_termination(data)
        state.metrics.update(
            out_of_bounds=out_of_bounds.astype(float), 
            **reward_info,
        )
        
        state = State(data, obs, reward, done, state.metrics, state.info)
        return state
        
    def render_from_info(
        self,
        qpos, qvel, mocap_pos, mocap_quat,
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
            d.mocap_pos[self._mocap_targets], d.mocap_quat[self._mocap_targets] = mocap_pos.reshape(self._config.num_cubes, 3), mocap_quat.reshape(self._config.num_cubes, 4)
            mujoco.mj_forward(self._mj_model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            return renderer.render()

        out = get_image(qpos, qvel, mocap_pos, mocap_quat)
        renderer.close()

        return out

class TruncatedRewardCreativeCube(CreativeCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        super().__init__(config=config)
    
    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh( self._config.reward_sensitivity * jnp.clip(obj_target_pos_err, min=self._config.success_threshold) )).astype(float)

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info

    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh( self._config.reward_sensitivity * jnp.clip(obj_target_pos_err, min=self._config.success_threshold) )).astype(float)

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info

class SparseCreativeCube(CreativeCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        super().__init__(config=config)
        assert self._config.env_early_termination == False, "sparse-creative envs requires env_early_termination to be False"
        
    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)

        dense_reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)).astype(float)
        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "dense_reward": dense_reward,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)

        dense_reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)).astype(float)
        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes
        
        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "stacking": self._stacking_metric(achieved_goal, target_goal),
            "dense_reward": dense_reward,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_permutation_invariant_reward_from_goals(self, achieved_goal, target_goal):
        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes
        return reward
        
    def get_permutation_variant_reward_from_goals(self, achieved_goal, target_goal):
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes        
        return reward

    def get_reward_from_goals(self, achieved_goal, target_goal):
        achieved_goal = achieved_goal.reshape((self._config.num_cubes, -1))
        target_goal = target_goal.reshape((self._config.num_cubes, -1))
        if self._config.permutation_invariant_reward:
            reward = self.get_permutation_invariant_reward_from_goals(achieved_goal, target_goal)
        else:
            reward = self.get_permutation_variant_reward_from_goals(achieved_goal, target_goal)
        return reward