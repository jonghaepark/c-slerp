import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


class WaypointWrapper(Wrapper):
    """Initializes wp_counter and current_waypoint in state.info.

    Step is pass-through — actor_step handles all runtime logic
    (counter increment, waypoint sampling, episode-boundary reset).
    """

    def __init__(self, env, repr_dim: int, control_horizon: int, goal_indices: tuple,
                 state_size: int = 0):
        super().__init__(env)
        self.repr_dim = repr_dim
        self.control_horizon = control_horizon
        self.goal_indices = goal_indices
        self.state_size = state_size

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["wp_counter"] = jnp.zeros(rng.shape[:-1], dtype=jnp.int32)
        state.info["current_waypoint"] = jnp.zeros((*rng.shape[:-1], self.repr_dim))
        state.info["current_planning_k"] = jnp.zeros(rng.shape[:-1], dtype=jnp.int32)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)
