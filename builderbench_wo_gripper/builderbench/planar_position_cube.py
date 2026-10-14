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
        halfgrid_size=3,
        action_scale = config_dict.create(
            xy_scale = 0.01,
            select_scale = np.pi,
            ),
        delta_control=True,     # direct position control would be too aggressive
        episode_length=150,
        reward_sensitivity=5.0,
        success_threshold=0.02,
        easy_success_threshold=0.05,
        env_early_termination=True,
        permutation_invariant_reward=True,
        impl='warp'
    )
    return config

class PlanarPositionCube():
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
    def action_size(self):
        """Size of the action space."""
        return 3
    
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
        
        # preprocess config
        config.scaled_halfgrid = config.halfgrid_size * 0.042
        config.workspace_bound = (config.halfgrid_size - 0.5) * 0.042

        # prepare spec and add objects to the spec
        spec = self._prepare_spec(xml_path, config)
        spec, self._object_names = self._add_objects(spec, config.num_cubes, config.workspace_bound)
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
            obj_adr + np.arange(2)
            for obj_adr in self._objs_qposadr
        ])

        # get start indices in actuators
        self._objs_actuator_adr = np.stack([
            np.array([
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in [f"x_block_{i}", f"y_block_{i}"]
            ])
            for i in range(config.num_cubes)
        ])
        self._objs_actuator_adr = jnp.array(self._objs_actuator_adr)

        # get constants
        _data = mujoco.MjData(self._mj_model)
        _data.ctrl = np.array([0, 0] * config.num_cubes)

        self._init_q = jnp.array(_data.qpos, dtype=jnp.float32)
        self._init_v = jnp.array(_data.qvel, dtype=jnp.float32) * 0
        self._init_ctrl = jnp.array(_data.ctrl, dtype=jnp.float32)
        
        # set action scale
        self._action_scale =  np.array([config.action_scale.xy_scale]*2 + [config.action_scale.select_scale])

        # set bounds
        self._workspace_bounds = jnp.array([ [ -config.workspace_bound-0.01, -config.workspace_bound-0.01, 0.03 ], [ config.workspace_bound+0.01, config.workspace_bound+0.01, 0.05 ] ])
        self._ctrl_bounds = jnp.array( self._mj_model.actuator_ctrlrange.T )
        self._ctrl_median = (self._ctrl_bounds[1] + self._ctrl_bounds[0]) / 2
        self._ctrl_halfspan = (self._ctrl_bounds[1] - self._ctrl_bounds[0]) / 2

        # calculate grid to sample target and start positions
        x_vals = jnp.arange(-config.halfgrid_size, config.halfgrid_size) + 0.5
        y_vals = jnp.arange(-config.halfgrid_size, config.halfgrid_size) + 0.5
        yy, xx = jnp.meshgrid(y_vals, x_vals, indexing='ij')
        self._grid_points = jnp.stack([yy.ravel(), xx.ravel()], axis=-1) * 0.042

    def _add_objects(self, spec, num_cubes, half_grid_size):
        object_names = []
        for i in range(num_cubes):

            # placement
            # this placement is important to ensure that the range of all cubes is the same. 
            # For some reason, the range of body joints are defined relative to the initial position of the body.
            x = 0.0
            y = 0.0
            z = 0.04

            body = spec.worldbody.add_body(
                name=f"block_{i}",
                pos=[x, y, z], 
            )

            mass = (0.04 ** 3) * 1240
            stiffness = 300.0
            damping = 2 * np.sqrt(mass * stiffness)

            body.add_joint(
                name=f"x_block_joint_{i}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(1, 0, 0),
                range=(-half_grid_size, half_grid_size),
                damping=damping,
            )
            body.add_joint(
                name=f"y_block_joint_{i}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(0, 1, 0),
                range=(-half_grid_size, half_grid_size),
                damping=damping,
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
                target=f"x_block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrllimited=True,
                ctrlrange=(-half_grid_size, half_grid_size), # Range matches workspace
            )
            actuator_x.set_to_position(kp=stiffness)

            actuator_y = spec.add_actuator(
                name=f"y_block_{i}",
                target=f"y_block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrllimited=True,
                ctrlrange=(-half_grid_size, half_grid_size), # Range matches workspace
            )
            actuator_y.set_to_position(kp=stiffness)

            # adding target position for cube
            body = spec.worldbody.add_body(
                name=f"target_mocap_{i}",
                mocap=True,
                pos=[x, y, z],
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
        
        spec.stat.center = np.array([0.0, 0.0 , 0.0])
        spec.stat.extent = 0.8
        spec.visual.global_.elevation = -45.0
        spec.visual.global_.azimuth = 180
        
        limit = config.scaled_halfgrid + 0.02
        thickness = 0.005 
        length = limit + thickness 
        color = [0.8, 0.3, 0.3, 0.4]         
        def add_visual_geom(name, pos, size):
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                rgba=color,
                contype=0,
                conaffinity=0,
                group=1,
            )

        add_visual_geom("border_top", [0, limit, 0.001], [length, thickness, 0.001])        
        add_visual_geom("border_bottom", [0, -limit, 0.001], [length, thickness, 0.001])        
        add_visual_geom("border_right", [limit, 0, 0.001], [thickness, length, 0.001])
        add_visual_geom("border_left", [-limit, 0, 0.001], [thickness, length, 0.001])

        return spec
    
    def reset(self, rng: jax.Array):
        rng, rng_starts, rng_starts_noise, rng_target, rng_target_noise, rng_select_action = jax.random.split(rng, 6)

        object_pos = jax.random.choice(rng_starts, self._grid_points, shape=(self._config.num_cubes,), axis=0, replace=False)
        object_pos += jax.random.normal(rng_starts_noise, shape=object_pos.shape) * 0.002    
        object_pos = jnp.clip(
            object_pos, 
            self._workspace_bounds[0][:2], 
            self._workspace_bounds[1][:2]
        )

        achieved_goal = object_pos.reshape(-1)
        object_pos = object_pos.reshape(-1)

        target_object_pos = jax.random.choice(rng_target, self._grid_points, shape=(self._config.num_cubes,), axis=0, replace=False)   
        target_object_pos += jax.random.normal(rng_target_noise, shape=target_object_pos.shape) * 0.002    
        target_object_pos = jnp.clip(
            target_object_pos, 
            self._workspace_bounds[0][:2], 
            self._workspace_bounds[1][:2]
        )
        target_object_quat = jnp.tile(jnp.array([1,0,0,0], dtype=jnp.float32), (self._config.num_cubes, 1))

        # set initial object position
        init_q = (
            self._init_q            
            .at[self._objs_pos_qpos_idxs]
            .set(object_pos)
        )

        # set initial position and velocity of all joints and initial control of actuators
        data = mjx_make_data(
            self._mj_model,
            qpos=init_q,
            qvel=self._init_v,
            ctrl=object_pos,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # set target mocap pos and quat
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_targets].set(jnp.concatenate([target_object_pos, jnp.array([[0.04]]*self._config.num_cubes)], axis=-1)),
            mocap_quat=data.mocap_quat.at[self._mocap_targets].set(target_object_quat),
        )

        metrics = {
            "easy_success": jnp.array(0.0, dtype=float),
            "success": jnp.array(0.0, dtype=float),
            "out_of_bounds": jnp.array(0.0, dtype=float),
            "obj_moved": jnp.array(0.0, dtype=float),
            "obj_goal_dist": jnp.array(0.0, dtype=float),
        }
        info = {
            "rng": rng, 
            "select_action": jax.random.uniform(rng_select_action, minval=-1, maxval=1),
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

        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]
        achieved_goal = obj_pos.reshape(-1,)
        obj_pos = obj_pos.reshape(-1,)
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(2)].reshape(-1,)

        select_action = info["select_action"]

        if self._config.delta_control:
            obs = jnp.concatenate([
                obj_pos,
                obj_linvel,
                data.ctrl,
                select_action[None],
            ])
        else:
            obs = jnp.concatenate([
                obj_pos,
                obj_linvel,
                select_action[None],
            ])

        info.update({
            "achieved_goal": achieved_goal, 
        })

        return obs, info
    
    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(2)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(2)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info

    def get_termination(self, data):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]

        out_of_bounds = 1 - ( jnp.all((obj_pos >= self._workspace_bounds[0][:2]) & jnp.all(obj_pos <= self._workspace_bounds[1][:2])) )
        termination = out_of_bounds | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        termination = ( termination * self._config.env_early_termination ).astype(float)
        termination = termination.astype(float)
        return termination, out_of_bounds
    
    def step(self, state, action):
        
        selected_cube_idx = jnp.digitize( ( self._action_scale[-1] * state.info["select_action"] + jnp.pi ), bins = jnp.arange(1, self._config.num_cubes+1) * 2 * jnp.pi / ( self._config.num_cubes ) )
        state.info.update(
            select_action = jnp.clip(action[-1], -1, 1),
        )
        new_ctrl = state.data.ctrl

        if self._config.delta_control:
            new_ctrl = (
                new_ctrl            
                .at[ self._objs_actuator_adr[selected_cube_idx] ] 
                .set( action[:-1] * self._action_scale[:-1] + state.data.ctrl[ self._objs_actuator_adr[selected_cube_idx] ] )
            )
            new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])
        
        else:
            new_ctrl = (
                new_ctrl            
                .at[ self._objs_actuator_adr[selected_cube_idx] ] 
                .set( action[:-1] * self._ctrl_halfspan[:2] + self._ctrl_median[:2] )
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
            d.mocap_pos[self._mocap_targets, :2] = mocap_pos.reshape(self._config.num_cubes, 2)
            d.mocap_pos[self._mocap_targets, -1] = 0.04
            d.mocap_quat[self._mocap_targets] = mocap_quat.reshape(self._config.num_cubes, 4)
            mujoco.mj_forward(self._mj_model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            return renderer.render()

        out = get_image(qpos, qvel, mocap_pos, mocap_quat)
        renderer.close()

        return out
            
class SparsePlanarPositionCube(PlanarPositionCube):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        super().__init__(config=config)
        assert self._config.env_early_termination == False, "sparse-creative envs requires env_early_termination to be False"
    
    def get_permutation_invariant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(2)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)

        dense_reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)).astype(float)
        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "dense_reward": dense_reward,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_permutation_variant_reward_from_obs(self, data, info):
        obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(2)]
        achieved_goal = obj_pos
        obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(2)].reshape(-1,)

        target_goal = info["target_goal"].reshape((self._config.num_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)

        dense_reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err)).astype(float)
        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)
        reward = jnp.sum( obj_target_pos_err < self._config.success_threshold ).astype(float) - self._config.num_cubes
        
        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "dense_reward": dense_reward,
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
