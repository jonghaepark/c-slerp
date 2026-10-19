import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from flax import struct
from typing import Any, Dict, Optional

from builderbench.env_utils import State

class Wrapper():
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, rng) -> State:
        return self.env.reset(rng)
            
    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size
    
    @property
    def goal_size(self) -> int:
        return self.env.goal_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self.env.dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self.env.sim_dt
    
    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return self.env.n_substeps

    @property
    def spec(self) -> mujoco.MjSpec:
        return self.env.spec

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model
    
    @property
    def xml_path(self) -> str:
        return self.env.xml_path


@jax.jit
def get_yaw_from_quat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return yaw
    
class PDWrapper(Wrapper):
    def __init__(self, env, duration: int = 1, kp_pos: float = 10.0, kd_pos: float = 2.0, kp_yaw: float = 10.0, kd_yaw: float = 0.5, delta_control: bool = False):
        super().__init__(env)
        
        self._kp_pos = kp_pos
        self._kd_pos = kd_pos
        self._kp_yaw = kp_yaw
        self._kd_yaw = kd_yaw
        self._duration = duration
        self._delta_control = delta_control
        
        # to do, port the following values from env config
        self._cube_mass = 0.07936
        self._gravity = 9.81
        self._gravity_comp = self._cube_mass * self._gravity

        self._workspace_median = (self.env._workspace_bounds[1] + self.env._workspace_bounds[0]) / 2
        self._workspace_halfspan = (self.env._workspace_bounds[1] - self.env._workspace_bounds[0]) / 2

        self._yaw_median = jnp.array( [0.0] )
        self._yaw_halfspan = jnp.array( [np.pi / 2] )

    def get_action(self, state, waypoint_pos, waypoint_yaw, cube_id):
        current_pos = state.data.qpos[self.env._objs_qposadr[:, None] + np.arange(3)][cube_id]
        current_quat = state.data.qpos[(self.env._objs_qposadr + 3)[:, None] + np.arange(4)][cube_id]
        current_linvel = state.data.qvel[self.env._objs_qveladr[:, None] + np.arange(3)][cube_id]
        current_angvel = state.data.qvel[(self.env._objs_qveladr + 3)[:, None] + np.arange(3)][cube_id]
        
        error_pos = waypoint_pos - current_pos
        output_pos = (self._kp_pos * error_pos) + (self._kd_pos * - current_linvel)
        output_pos = output_pos.at[2].add(self._gravity_comp)

        current_yaw = get_yaw_from_quat(current_quat)
        delta_yaw = waypoint_yaw - current_yaw
        error_yaw = jnp.arctan2(jnp.sin(delta_yaw), jnp.cos(delta_yaw))        
        output_yaw = (self._kp_yaw * error_yaw) + (self._kd_yaw * - current_angvel[-1])

        raw_ctrl_action = jnp.concatenate([output_pos, output_yaw], axis=0)
        ctrl_action = ( raw_ctrl_action - self.env._ctrl_median[:4] ) / self.env._ctrl_halfspan[:4]
        
        select_action = ( ( ( 2 * cube_id + 1) * jnp.pi / self.env._config.num_cubes ) - jnp.pi ) / ( jnp.pi )

        action =  jnp.concatenate([ctrl_action, select_action[None]], axis=0)
        action = jnp.clip(action, -1, 1)
        return action
    
    def step(self, state, action):

        state.info.update(
            select_action = jnp.clip(action[-1], -1, 1),
        )
        cube_id = jnp.digitize( ( self.env._action_scale[-1] * action[-1] + jnp.pi ), bins = jnp.arange(1, self.env._config.num_cubes+1) * 2 * jnp.pi / ( self.env._config.num_cubes ) )

        if self._delta_control:
            raise NotImplementedError
        else:
            waypoint_pos = action[:3] * self._workspace_halfspan + self._workspace_median
            waypoint_yaw = action[3] * self._yaw_halfspan + self._yaw_median

        def f(carry, _):
            state, prev_done = carry
            action = self.get_action(state, waypoint_pos, waypoint_yaw, cube_id)
            state = self.env.step(state, action)
            done = jnp.maximum(state.done, prev_done)
            
            return (state, done), (state.reward, prev_done, state.metrics)
        
        (state, final_done), (rewards, dones, metrics) = jax.lax.scan(f, (state, state.done), (), self._duration)

        state = state.replace(reward = jnp.sum(rewards * (1-dones)))
        state = state.replace(metrics = jax.tree_util.tree_map(lambda m: jnp.sum( m * (1-dones) ), metrics))
        state = state.replace(done = final_done)
        return state

class AutoResetWrapper(Wrapper):
    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_data'] = state.data
        state.info['first_obs'] = state.obs
        state.info['first_reward'] = state.reward
        state.info['first_metrics'] = state.metrics
        self._select_action_exist = False
        if 'select_action' in state.info:
            state.info['first_select_action'] = state.info['select_action']
            self._select_action_exist = True
        self._active_cube_id_exist = False
        if 'active_cube_id' in state.info:
            state.info['first_active_cube_id'] = state.info['active_cube_id']
            self._active_cube_id_exist = True
        state.info['first_achieved_goal'] = state.info['achieved_goal']
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])

        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return x
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        data = jax.tree.map(where_done, state.info['first_data'], state.data)
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        reward = jax.tree.map(where_done, state.info['first_reward'], state.reward)
        metrics = jax.tree.map(where_done, state.info['first_metrics'], state.metrics)

        state.info['achieved_goal'] = jax.tree.map(where_done, state.info['first_achieved_goal'], state.info['achieved_goal'])
        if self._select_action_exist:
            state.info['select_action'] = jax.tree.map(where_done, state.info['first_select_action'], state.info['select_action'])
        if self._active_cube_id_exist:
            state.info['active_cube_id'] = jax.tree.map(where_done, state.info['first_active_cube_id'], state.info['active_cube_id'])
        state.info['traj_id'] = jnp.where(state.done, state.info['traj_id'] + 1, state.info['traj_id'])

        return state.replace(data=data, obs=obs, reward=reward, metrics=metrics, info=state.info)
        
class VmapWrapper(Wrapper):
    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)
    
    def step(self, state, action):
        return jax.vmap(self.env.step)(state, action)

class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        assert action_repeat == 1
        self.action_repeat = action_repeat  # unsued right now

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        
        steps = state.info['steps'] + 1

        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps        
        return state.replace(done=done)

@struct.dataclass
class EvalMetrics:
    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array

class EvalWrapper(Wrapper):
    def reset(self, rng: jax.Array):
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward

        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state
        
    def step(self, state, action):
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']

        state  = self.env.step(state, action)

        state.metrics['reward'] = state.reward

        episode_steps = jnp.where(
            state_metrics.active_episodes,
            state.info['steps'],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            state.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - state.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        state.info['eval_metrics'] = eval_metrics

        return state

def wrap_env(
    env,
    episode_length: int = 150,
    action_repeat: int = 1,
) -> Wrapper:

    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env