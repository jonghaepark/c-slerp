import jax
import flax
import functools
import jax.numpy as jnp

from jax import flatten_util

# Adapted from https://github.com/MichalBortkiewicz/JaxGCRL/blob/master/jaxgcrl/utils/replay_buffer.py

@flax.struct.dataclass
class ReplayBufferState:
    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: jax.random.PRNGKey

class TrajectoryUniformSamplingQueue():
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        sequence_length: int,
    ):

        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self.num_envs = num_envs
        self.sequence_length = sequence_length

    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state, samples):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples) # updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data # shape = (max_replay_size, num_envs, data_size)

        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        shape = self.num_envs

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.sequence_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.sequence_length,
            sample_key,
        )

        # sample the environment after constructing the transition batch to avoid reallocating the memory for full environment permutations
        # The previous approach was permute environment -> construct transition batch, which needed full environment x full buffer stored length memory intermediate
        # This new approach is sample transition first -> permute environment, which only needs sampled sequence length x environment memory intermediate
        batch = create_batch_vmaped(buffer_state.data, matrix)
        envs_idxs = jax.random.permutation(shuffle_key, self.num_envs)
        batch = batch[envs_idxs]
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_crl_fn(buffer_config, transition, sample_key):

        gamma, = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
        probs = is_future_mask * discount  
        
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
        )

        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))

        goal_obs = jnp.take(transition.achieved_goal, goal_index[:-1], axis=0) # the last goal_index cannot be considered as there is no future.  

        extras = {
            "state_extras": {
                "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            },
            "future_goal": goal_obs,
        }

        return transition._replace(
            observation=transition.observation[:-1],
            achieved_goal=transition.achieved_goal[:-1],
            action=jnp.squeeze(transition.action[:-1]),
            extras=extras,
        )
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_cslerp_sg_fn(buffer_config, transition, sample_key):
        import jax.scipy as jsp

        gamma, max_waypoint_depth = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        traj_id = transition.extras["state_extras"]["traj_id"]

        is_boundary = jnp.concatenate(
            [traj_id[:-1] != traj_id[1:], jnp.array([True])]
        )
        boundary_idx = jnp.where(is_boundary, arrangement, seq_len)
        last_valid = jnp.flip(
            jax.lax.associative_scan(jnp.minimum, jnp.flip(boundary_idx))
        )
        remaining = (last_valid - arrangement).astype(jnp.int32)

        eff_interval = jnp.maximum(
            1, jnp.round(1.0 / (1.0 - gamma)).astype(jnp.int32)
        )
        max_scaled = jnp.round(remaining / eff_interval).astype(jnp.int32)
        feasible_k_max = jnp.maximum(max_scaled // 2, 1)
        # feasible_k_max = jnp.maximum(jnp.floor(max_scaled / 1.5), 1) #reducing the conservativeness of the k_target
        feasible_k_max = jnp.clip(feasible_k_max[:-1], 1, max_waypoint_depth)

        key_k, key_goal = jax.random.split(sample_key)
        k_target = jax.random.randint(
            key_k,
            shape=feasible_k_max.shape,
            minval=1,
            maxval=feasible_k_max + 1,
        )
        # k_target is the NegBin count parameter (K=1 -> one geometric draw,
        # which maps to internal k_star=0 in the planner).
        k_valid_mask = (remaining[:-1] > 0).astype(jnp.float32)
        k_target = jnp.minimum(k_target, remaining[:-1])
        k_target = jnp.maximum(k_target, 1)

        t_indices = arrangement[:-1]
        tau = arrangement[None, :] - t_indices[:, None]

        same_traj = traj_id[None, :] == traj_id[:-1, None]
        valid = same_traj & (tau > 0) & (tau >= k_target[:, None])

        tau_f = tau.astype(jnp.float32)
        k_f = k_target[:, None].astype(jnp.float32)
        tau_safe = jnp.maximum(tau_f, k_f)

        log_comb = (
            jsp.special.gammaln(tau_safe)
            - jsp.special.gammaln(k_f)
            - jsp.special.gammaln(tau_safe - k_f + 1.0)
        )
        log_pmf = (
            log_comb
            + k_f * jnp.log(jnp.maximum(1.0 - gamma, 1e-12))
            + (tau_safe - k_f) * jnp.log(jnp.maximum(gamma, 1e-12))
        )

        main_log_probs = jnp.where(valid, log_pmf, -1e10)
        row_has_valid = jnp.any(valid, axis=-1, keepdims=True)
        fallback_mask = tau == remaining[:-1, None]
        fallback_log_probs = jnp.where(fallback_mask, 0.0, -1e10)
        safe_log_probs = jnp.where(row_has_valid, main_log_probs, fallback_log_probs)

        goal_index = jax.random.categorical(key_goal, safe_log_probs, axis=-1)
        sg_goal = jnp.take(transition.achieved_goal, goal_index, axis=0)
        sampled_interval = (goal_index - t_indices).astype(jnp.int32)

        extras = {
            "state_extras": {
                "traj_id": transition.extras["state_extras"]["traj_id"][:-1],
            },
            "sg_goal": sg_goal,
            "k_target": k_target,
            "k_valid_mask": k_valid_mask,
            "negbin_interval": sampled_interval,
        }

        return transition._replace(
            observation=transition.observation[:-1],
            achieved_goal=transition.achieved_goal[:-1],
            action=jnp.squeeze(transition.action[:-1]),
            extras=extras,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_crl_pick_fn(buffer_config, transition, sample_key):
        """Pick-timestep-first variant of `flatten_crl_fn`."""
        gamma, = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        traj_id = transition.extras["state_extras"]["traj_id"]

        pick_key, goal_key = jax.random.split(sample_key)
        # Uniform t in [0, seq_len - 2]; the last timestep has no future (matches the
        # old `_pick_one` maxval == action.shape[1] == seq_len - 1).
        t = jax.random.randint(pick_key, (), minval=0, maxval=seq_len - 1)

        is_future = (arrangement > t).astype(jnp.float32)
        discount_row = gamma ** (arrangement - t).astype(jnp.float32)
        same_traj = jnp.equal(traj_id, traj_id[t]).astype(jnp.float32)
        probs_row = is_future * discount_row * same_traj
        probs_row = probs_row.at[t].add(1e-5)  # self-mass regularizer (old +eye*1e-5)

        goal_index = jax.random.categorical(goal_key, jnp.log(probs_row))
        future_obs = jnp.take(transition.observation, goal_index, axis=0)  # full state at the SW future
        goal_obs = jnp.take(transition.achieved_goal, goal_index, axis=0)

        extras = {
            "state_extras": {
                "traj_id": transition.extras["state_extras"]["traj_id"][t],
            },
            "future_goal": goal_obs,
            "future_obs": future_obs,  # full state at the SW future (for full-state psi)
        }

        return transition._replace(
            observation=transition.observation[t],
            achieved_goal=transition.achieved_goal[t],
            action=transition.action[t],
            extras=extras,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_cslerp_sg_pick_fn(buffer_config, transition, sample_key):
        import jax.scipy as jsp

        gamma, max_waypoint_depth = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        traj_id = transition.extras["state_extras"]["traj_id"]

        is_boundary = jnp.concatenate(
            [traj_id[:-1] != traj_id[1:], jnp.array([True])]
        )
        boundary_idx = jnp.where(is_boundary, arrangement, seq_len)
        last_valid = jnp.flip(
            jax.lax.associative_scan(jnp.minimum, jnp.flip(boundary_idx))
        )
        remaining = (last_valid - arrangement).astype(jnp.int32)

        eff_interval = jnp.maximum(
            1, jnp.round(1.0 / (1.0 - gamma)).astype(jnp.int32)
        )
        max_scaled = jnp.round(remaining / eff_interval).astype(jnp.int32)
        feasible_k_max = jnp.maximum(max_scaled // 2, 1)
        feasible_k_max = jnp.clip(feasible_k_max[:-1], 1, max_waypoint_depth)
        # until here, the code is the same as the flatten_cslerp_sg_fn.
        pick_key, key_k, key_goal = jax.random.split(sample_key, 3)
        # Uniform t in [0, seq_len - 2] (t_indices == arrangement[:-1] in the dense fn).
        t = jax.random.randint(pick_key, (), minval=0, maxval=seq_len - 1)

        remaining_t = remaining[t]
        feasible_k_max_t = feasible_k_max[t]
        k_target = jax.random.randint(key_k, (), minval=1, maxval=feasible_k_max_t + 1)
        k_valid_mask = (remaining_t > 0).astype(jnp.float32)
        k_target = jnp.minimum(k_target, remaining_t)
        k_target = jnp.maximum(k_target, 1)

        tau = arrangement - t                       # [seq_len], tau[j] = j - t
        same_traj = traj_id == traj_id[t]
        valid = same_traj & (tau > 0) & (tau >= k_target)

        tau_f = tau.astype(jnp.float32)
        k_f = k_target.astype(jnp.float32)
        tau_safe = jnp.maximum(tau_f, k_f)

        log_comb = (
            jsp.special.gammaln(tau_safe)
            - jsp.special.gammaln(k_f)
            - jsp.special.gammaln(tau_safe - k_f + 1.0)
        )
        log_pmf = (
            log_comb
            + k_f * jnp.log(jnp.maximum(1.0 - gamma, 1e-12))
            + (tau_safe - k_f) * jnp.log(jnp.maximum(gamma, 1e-12))
        )

        main_log_probs = jnp.where(valid, log_pmf, -1e10)
        row_has_valid = jnp.any(valid)
        fallback_mask = tau == remaining_t
        fallback_log_probs = jnp.where(fallback_mask, 0.0, -1e10)
        safe_log_probs = jnp.where(row_has_valid, main_log_probs, fallback_log_probs)

        goal_index = jax.random.categorical(key_goal, safe_log_probs)
        sg_goal = jnp.take(transition.achieved_goal, goal_index, axis=0)
        sampled_interval = (goal_index - t).astype(jnp.int32)

        extras = {
            "state_extras": {
                "traj_id": transition.extras["state_extras"]["traj_id"][t],
            },
            "sg_goal": sg_goal,
            "k_target": k_target,
            "k_valid_mask": k_valid_mask,
            "negbin_interval": sampled_interval,
        }

        return transition._replace(
            observation=transition.observation[t],
            achieved_goal=transition.achieved_goal[t],
            action=transition.action[t],
            extras=extras,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_sac_her_fn(buffer_config, transition, sample_key):

        gamma, = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
        probs = is_future_mask * discount  
        
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
        )

        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))

        goal_obs = jnp.take(transition.achieved_goal, goal_index[:-1], axis=0) 

        extras = {
            "state_extras": {
                "truncation": transition.extras["state_extras"]["truncation"][:-1],
                "traj_id": transition.extras["state_extras"]["traj_id"][:-1],
            },
            "future_goals": goal_obs,
        }

        return transition._replace(
            observation=transition.observation[:-1],
            commanded_goal=transition.commanded_goal[:-1],
            achieved_goal=transition.achieved_goal[:-1],
            action=transition.action[:-1],
            done=transition.done[:-1],
            next_observation=transition.next_observation[:-1],
            extras=extras,
        )

    def size(self, buffer_state):
        return (buffer_state.insert_position - buffer_state.sample_position)