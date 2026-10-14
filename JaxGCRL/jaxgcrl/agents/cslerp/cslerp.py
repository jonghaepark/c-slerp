import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper, WaypointWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import (
    _cayley_rotation,
    energy_fn,
    update_critics,
    update_low_actor_and_alpha,
)
from .networks import CayleyTransform, LowActor, NormalizedEncoder
Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    low_actor_state: TrainState
    critic_state: TrainState
    alpha_low_state: TrainState
    alpha_high_state: TrainState


class Transition(NamedTuple):
    """Container for a transition."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def _normalize_latent(x: jnp.ndarray) -> jnp.ndarray:
    return x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-6)


def _rotation_powers(rotation: jnp.ndarray, num_powers: int) -> jnp.ndarray:
    """Returns [I, rotation, ..., rotation^(num_powers - 1)]."""
    powers = []
    current = jnp.eye(rotation.shape[0], dtype=rotation.dtype)
    for _ in range(num_powers):
        powers.append(current)
        current = current @ rotation
    return jnp.stack(powers, axis=0)


def analytic_k_logits(
    state_latent: jnp.ndarray,
    goal_latent: jnp.ndarray,
    rotation: jnp.ndarray,
    max_waypoint_depth: int,
) -> jnp.ndarray:
    """Returns alignment logits for zero-based planning depths.

    Logit ``k`` is ``psi(s)^T R^(k+1) psi(g)`` for
    ``k in {0, ..., max_waypoint_depth - 1}``.  With row-vector latents this is
    computed as ``psi(g) @ (R.T)^(k+1)`` before taking the dot product.
    """
    state_latent = _normalize_latent(state_latent)
    goal_latent = _normalize_latent(goal_latent)
    rt_powers = _rotation_powers(rotation.T, max_waypoint_depth + 1)[1:]
    goal_t_all = jnp.einsum("bd,kde->bke", goal_latent, rt_powers)
    return jnp.sum(state_latent[:, None, :] * goal_t_all, axis=-1)


def select_analytic_k(
    state_latent: jnp.ndarray,
    goal_latent: jnp.ndarray,
    rotation: jnp.ndarray,
    max_waypoint_depth: int,
    *,
    key: Optional[jax.Array] = None,
    selection: Literal["sample", "argmax"] = "argmax",
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Selects zero-based K from analytic alignment logits."""
    logits = analytic_k_logits(
        state_latent,
        goal_latent,
        rotation,
        max_waypoint_depth,
    )
    if selection == "sample":
        if key is None:
            raise ValueError("Analytic K sampling requires a PRNG key")
        temperature = jnp.asarray(temperature, dtype=logits.dtype)
        temperature = jnp.maximum(temperature, jnp.asarray(1e-6, dtype=logits.dtype))
        k_star = jax.random.categorical(key, logits / temperature, axis=-1)
    elif selection == "argmax":
        k_star = jnp.argmax(logits, axis=-1)
    else:
        raise ValueError(f"Unknown analytic K selection mode: {selection}")
    return k_star.astype(jnp.int32), logits


def _orthogonal_unit(x: jnp.ndarray) -> jnp.ndarray:
    basis_idx = jnp.argmin(jnp.abs(x), axis=-1)
    basis = jax.nn.one_hot(basis_idx, x.shape[-1], dtype=x.dtype)
    basis = basis - jnp.sum(basis * x, axis=-1, keepdims=True) * x
    return _normalize_latent(basis)


def _slerp(a: jnp.ndarray, b: jnp.ndarray, frac: jnp.ndarray) -> jnp.ndarray:
    raw_dot = jnp.sum(a * b, axis=-1, keepdims=True)
    dot = jnp.clip(raw_dot, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    safe_sin_theta = jnp.maximum(sin_theta, 1e-6)

    spherical = (
        jnp.sin((1.0 - frac) * theta) / safe_sin_theta * a
        + jnp.sin(frac * theta) / safe_sin_theta * b
    )
    linear = _normalize_latent((1.0 - frac) * a + frac * b)
    anti = _normalize_latent(
        jnp.cos(jnp.pi * frac) * a + jnp.sin(jnp.pi * frac) * _orthogonal_unit(a)
    )

    fallback = jnp.where(raw_dot < -1.0 + 1e-4, anti, linear)
    out = jnp.where(sin_theta > 1e-4, spherical, fallback)
    out = jnp.where(frac <= 1e-6, a, out)
    out = jnp.where(frac >= 1.0 - 1e-6, b, out)
    return _normalize_latent(out)


def midpoint_planning_i(k_star: jnp.ndarray) -> jnp.ndarray:
    """One-based midpoint index i = ceil(K/2) for one-based depth K = k_star + 1."""
    k_star = k_star.astype(jnp.int32)
    return (k_star + 2) // 2


def closed_form_high_waypoint(
    state_latent: jnp.ndarray,
    goal_latent: jnp.ndarray,
    rotation: jnp.ndarray,
    max_waypoint_depth: int,
    alpha_high: jnp.ndarray,
    k_star: jnp.ndarray,
    waypoint_index: int = 1,
    use_midpoint: bool = False,
    min_logstd: float = -8.0,
    max_logstd: float = -2.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Closed-form high-level policy with online K selection."""
    state_latent = _normalize_latent(state_latent)
    goal_latent = _normalize_latent(goal_latent)

    # Row-vector convention: (R^(K+1) goal_col)^T == goal_row @ (R.T)^(K+1).
    rt_powers = _rotation_powers(rotation.T, max_waypoint_depth + 1)[1:]
    r_powers = _rotation_powers(rotation, max_waypoint_depth + 1)
    goal_t_all = jnp.einsum("bd,kde->bke", goal_latent, rt_powers)
    alignments = jnp.sum(state_latent[:, None, :] * goal_t_all, axis=-1)

    k_star = jnp.clip(
        k_star.astype(jnp.int32),
        0,
        max_waypoint_depth - 1,
    )
    goal_t = jnp.take_along_axis(goal_t_all, k_star[:, None, None], axis=1)[:, 0, :]
    selected_alignment = jnp.take_along_axis(alignments, k_star[:, None], axis=1)[:, 0]

    if use_midpoint:
        i_star = midpoint_planning_i(k_star)
        direct_goal = i_star >= (k_star + 1)
    else:
        fixed_i = jnp.asarray(waypoint_index, dtype=jnp.int32)
        direct_goal = k_star < fixed_i
        i_star = jnp.where(direct_goal, k_star + 1, fixed_i)
    k_float = k_star.astype(state_latent.dtype)
    i_float = i_star.astype(state_latent.dtype)
    frac = (i_float / (k_float + 1.0))[:, None]
    t_i = _slerp(state_latent, goal_t, frac)
    r_i = jnp.take(r_powers, i_star, axis=0)
    mean = _normalize_latent(jnp.einsum("bd,bde->be", t_i, r_i))
    direct_goal = direct_goal[:, None]
    mean = jnp.where(direct_goal, goal_latent, mean)

    theta = jnp.arccos(jnp.clip(selected_alignment, -1.0 + 1e-6, 1.0 - 1e-6))
    cos_step = jnp.cos(theta / (k_float + 1.0))
    alpha = jnp.asarray(alpha_high, dtype=state_latent.dtype)
    sigma_sq = alpha / jnp.maximum(2.0 * cos_step, 1e-6)
    logstd = 0.5 * jnp.log(sigma_sq)
    logstd = jnp.clip(
        logstd,
        jnp.asarray(min_logstd, dtype=state_latent.dtype),
        jnp.asarray(max_logstd, dtype=state_latent.dtype),
    )
    sigma_sq = jnp.exp(2.0 * logstd)

    waypoint = mean
    if key is not None:
        eps = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        eps_tan = eps - jnp.sum(eps * mean, axis=-1, keepdims=True) * mean
        noisy_waypoint = _normalize_latent(mean + jnp.sqrt(sigma_sq)[:, None] * eps_tan)
        waypoint = jnp.where(direct_goal, mean, noisy_waypoint)

    return waypoint, mean, k_star, selected_alignment, sigma_sq


def add_tangent_noise_to_waypoint(
    waypoint: jnp.ndarray,
    sigma_sq: jnp.ndarray,
    direct_goal_mask: jnp.ndarray,
    goal_latent: jnp.ndarray,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Adds tangent-space Gaussian noise while keeping direct goals exact."""
    if key is None:
        return waypoint

    eps = jax.random.normal(key, shape=waypoint.shape, dtype=waypoint.dtype)
    eps_tan = eps - jnp.sum(eps * waypoint, axis=-1, keepdims=True) * waypoint
    sigma = jnp.sqrt(sigma_sq).reshape(waypoint.shape[:-1] + (1,))
    noisy_waypoint = _normalize_latent(waypoint + sigma * eps_tan)
    direct_goal_mask = direct_goal_mask.reshape(waypoint.shape[:-1] + (1,))
    return jnp.where(direct_goal_mask, goal_latent, noisy_waypoint)


def closed_form_high_waypoint_candidates(
    state_latent: jnp.ndarray,
    goal_latent: jnp.ndarray,
    rotation: jnp.ndarray,
    max_waypoint_depth: int,
    k_star: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Builds all one-based closed-form waypoint candidates for each K*.

    Returns candidates with shape [B, max_waypoint_depth, D].  Valid candidates
    are i in {1, ..., K* + 1}; the final valid candidate K* + 1 is the direct
    goal.  Invalid slots are filled with the direct goal and masked out by the
    returned valid_mask.
    """
    state_latent = _normalize_latent(state_latent)
    goal_latent = _normalize_latent(goal_latent)

    rt_powers = _rotation_powers(rotation.T, max_waypoint_depth + 1)[1:]
    r_powers = _rotation_powers(rotation, max_waypoint_depth + 1)
    goal_t_all = jnp.einsum("bd,kde->bke", goal_latent, rt_powers)
    alignments = jnp.sum(state_latent[:, None, :] * goal_t_all, axis=-1)

    k_star = jnp.clip(
        k_star.astype(jnp.int32),
        0,
        max_waypoint_depth - 1,
    )
    goal_t = jnp.take_along_axis(goal_t_all, k_star[:, None, None], axis=1)[:, 0, :]
    selected_alignment = jnp.take_along_axis(alignments, k_star[:, None], axis=1)[:, 0]

    candidate_i = jnp.arange(1, max_waypoint_depth + 1, dtype=jnp.int32)
    candidate_i_batch = candidate_i[None, :]
    final_i = k_star[:, None] + 1
    valid_mask = candidate_i_batch <= final_i
    direct_goal_mask = candidate_i_batch == final_i

    k_float = k_star.astype(state_latent.dtype)
    frac = (
        candidate_i.astype(state_latent.dtype)[None, :]
        / (k_float[:, None] + 1.0)
    )[:, :, None]

    t_i = _slerp(state_latent[:, None, :], goal_t[:, None, :], frac)
    r_i = jnp.take(r_powers, candidate_i, axis=0)
    chain_candidates = _normalize_latent(jnp.einsum("bmd,mde->bme", t_i, r_i))
    goal_candidates = jnp.broadcast_to(
        goal_latent[:, None, :], chain_candidates.shape
    )
    candidates = jnp.where(direct_goal_mask[:, :, None], goal_candidates, chain_candidates)
    candidates = jnp.where(valid_mask[:, :, None], candidates, goal_candidates)

    return candidates, valid_mask, candidate_i, k_star, selected_alignment


def select_adaptive_waypoint_by_low_reachability(
    state: jnp.ndarray,
    candidate_waypoints: jnp.ndarray,
    valid_mask: jnp.ndarray,
    candidate_i: jnp.ndarray,
    k_star: jnp.ndarray,
    low_actor,
    low_actor_params,
    sa_encoder,
    sa_encoder_params,
    energy_name: str,
    threshold: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Selects the farthest reachable candidate under the current low critic."""
    batch_size, num_candidates = candidate_waypoints.shape[:2]
    state_tiled = jnp.broadcast_to(
        state[:, None, :],
        (batch_size, num_candidates, state.shape[-1]),
    )

    low_input = jnp.concatenate([state_tiled, candidate_waypoints], axis=-1)
    means, _ = low_actor.apply(
        low_actor_params,
        low_input.reshape(batch_size * num_candidates, -1),
    )
    actions = nn.tanh(means).reshape(batch_size, num_candidates, -1)

    sa_input = jnp.concatenate([state_tiled, actions], axis=-1)
    sa_repr = sa_encoder.apply(
        sa_encoder_params,
        sa_input.reshape(batch_size * num_candidates, -1),
    ).reshape(batch_size, num_candidates, -1)

    scores = energy_fn(energy_name, sa_repr, candidate_waypoints)

    threshold = jnp.asarray(threshold, dtype=scores.dtype)
    reachable = valid_mask & (scores >= threshold)
    candidate_i_batch = jnp.broadcast_to(candidate_i[None, :], reachable.shape)
    selected_i = jnp.max(jnp.where(reachable, candidate_i_batch, 0), axis=-1)
    has_reachable = selected_i > 0
    selected_i = jnp.where(has_reachable, selected_i, jnp.ones_like(selected_i))
    selected_index = selected_i - 1

    selected_waypoint = jnp.take_along_axis(
        candidate_waypoints,
        selected_index[:, None, None],
        axis=1,
    )[:, 0, :]
    selected_score = jnp.take_along_axis(
        scores,
        selected_index[:, None],
        axis=1,
    )[:, 0]
    direct_goal = selected_i == (k_star + 1)
    fallback = ~has_reachable

    return selected_waypoint, selected_i, selected_score, direct_goal, fallback


# ---------------------------------------------------------------------------
# Relabeling helpers
# ---------------------------------------------------------------------------

def flatten_batch_sw(buffer_config, transition, sample_key):
    """Geometric-horizon relabeling for low-level (critics + low policy).

    Returns transitions with extras["future_state"]: raw future state for
    encoding into latent at loss-computation time.
    """
    gamma, state_size, goal_indices = buffer_config
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)

    is_future_mask = (arrangement[:, None] < arrangement[None]).astype(jnp.float32)
    discount = gamma ** (arrangement[None] - arrangement[:, None]).astype(jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T)
    probs = probs + jnp.eye(seq_len) * 1e-5

    safe_log_probs = jnp.where(probs > 0, jnp.log(jnp.maximum(probs, 1e-30)), -1e10)
    future_index = jax.random.categorical(sample_key, safe_log_probs)
    future_obs = jnp.take(transition.observation, future_index[:-1], axis=0)
    future_state = future_obs[:, :state_size]

    state = transition.observation[:-1, :state_size]

    extras = {
        "state_extras": {
            "truncation": transition.extras["state_extras"]["truncation"][:-1],
            "traj_id": transition.extras["state_extras"]["traj_id"][:-1],
        },
        "state": state,
        "future_state": future_state,
    }

    return transition._replace(
        observation=transition.observation[:-1],
        action=transition.action[:-1],
        reward=transition.reward[:-1],
        discount=transition.discount[:-1],
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# ---------------------------------------------------------------------------
# Agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class C_slerp:

    # Learning rates
    low_policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_low_lr: float = 3e-4
    alpha_high_lr: float = 3e-3


    # Batch and buffer
    batch_size: int = 512
    max_replay_size: int = 10000
    min_replay_size: int = 2000
    # Number of env steps collected per get_experience call.
    # Each call produces a trajectory chunk of (unroll_length, num_envs, ...).
    # Must be < episode_length.
    unroll_length: int = 62
    # Extra gradient phases per env collection (each phase resamples the replay buffer).
    # Total update passes per outer step = train_step_multiplier × (num_envs×(episode_length−1) / batch_size).
    train_step_multiplier: int = 1
    sw_critic_repetition_factor: int = 1

    # Architecture
    h_dim: int = 512
    n_hidden_high: int = 9
    n_hidden_low: int = 9

    skip_connections_high: int = 4
    skip_connections_low: int = 4
    use_relu: bool = False
    use_ln: bool = True
    repr_dim: int = 128

    # Contrastive
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "dot"
    low_contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce"]  = "bwd_infonce"
    high_contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce"] = "sym_infonce"
    normalize_encoder: bool = True

    critic_temperature : float = 1e-1
    cayley_fro_reg_coef: float = 0.0
    learn_cayley: bool = True
    cayley_init_std: float = 0.0
    high_critic_loss_coef: float = 4.0
    actor_also_temperature: bool = False

    # Hierarchical
    discounting: float = 0.99
    control_horizon: int = 10
    max_waypoint_depth: int = 5
    waypoint_index: int = 1
    analytic_waypoint_min_logstd: float = -8.0
    analytic_waypoint_max_logstd: float = -2.0

    # K-selection for online closed-form high-level planning.
    analytic_k_sample_temperature: float = 0.1
    analytic_k_eval_sample_temperature: float = 0.1
    # Ablation: force a constant zero-based planning depth k_star instead of analytic K selection.
    fixed_planning_k: Optional[int] = None

    # I-selection for online closed-form high-level planning.
    adaptive_waypoint_selection: bool = True
    adaptive_waypoint_threshold: float = 0.4
    midpoint_waypoint_selection: bool = False

    # Shared encoder gradient config
    train_shared_in_low: bool = True

    # Entropy / closed-form high-policy temperature.
    alpha_low_init: float = 1.0
    alpha_high_init: float = 0.01
    
    target_entropy_low_coef: float = -1.0
    target_entropy_high_coef: float = -2.5

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        assert self.train_step_multiplier >= 1, "train_step_multiplier must be >= 1"
        assert self.sw_critic_repetition_factor >= 1, "sw_critic_repetition_factor must be >= 1"
        assert config.num_envs % self.sw_critic_repetition_factor == 0, (
            "num_envs must be divisible by sw_critic_repetition_factor"
        )
        assert self.alpha_high_init > 0, "alpha_high_init must be > 0"
        assert np.isfinite(self.analytic_k_sample_temperature) and (
            self.analytic_k_sample_temperature > 0
        ), (
            "analytic_k_sample_temperature must be finite and > 0"
        )
        assert np.isfinite(self.analytic_k_eval_sample_temperature) and (
            self.analytic_k_eval_sample_temperature > 0
        ), (
            "analytic_k_eval_sample_temperature must be finite and > 0"
        )
        assert self.high_critic_loss_coef >= 0, "high_critic_loss_coef must be >= 0"
        assert self.max_waypoint_depth >= 1, "max_waypoint_depth must be >= 1"
        assert self.waypoint_index >= 1, "waypoint_index must be >= 1"
        assert not (
            self.adaptive_waypoint_selection and self.midpoint_waypoint_selection
        ), "adaptive_waypoint_selection and midpoint_waypoint_selection are mutually exclusive"
        if self.fixed_planning_k is not None:
            assert (
                0 <= self.fixed_planning_k < self.max_waypoint_depth
            ), (
                "fixed_planning_k must be in "
                f"[0, max_waypoint_depth-1] = [0, {self.max_waypoint_depth - 1}], "
                f"got {self.fixed_planning_k}"
            )
        assert np.isfinite(
            self.adaptive_waypoint_threshold
        ), "adaptive_waypoint_threshold must be finite"
        assert (
            self.analytic_waypoint_min_logstd <= self.analytic_waypoint_max_logstd
        ), "analytic_waypoint_min_logstd must be <= analytic_waypoint_max_logstd"
        assert self.target_entropy_low_coef < 0, "target_entropy_low_coef must be < 0"
        # assert self.target_entropy_high_coef < 0, "target_entropy_high_coef must be < 0"

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        vis_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        unwrapped_vis_env = vis_env if vis_env is not None else eval_env
        eval_episode_length = (
            config.episode_length
            if config.eval_episode_length is None
            else config.eval_episode_length
        )

        # Wrap environments
        train_env = TrajectoryIdWrapper(train_env)
        train_env = WaypointWrapper(
            train_env,
            repr_dim=self.repr_dim,
            control_horizon=self.control_horizon,
            goal_indices=tuple(unwrapped_env.goal_indices),
            state_size=unwrapped_env.state_dim,
        )
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = WaypointWrapper(
            eval_env,
            repr_dim=self.repr_dim,
            control_horizon=self.control_horizon,
            goal_indices=tuple(unwrapped_env.goal_indices),
            state_size=unwrapped_env.state_dim,
        )
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=eval_episode_length,
            action_repeat=config.action_repeat,
        )

        # Timing
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )
        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        # K_max for online closed-form high-level planning.
        max_waypoint_depth = int(self.max_waypoint_depth)
        waypoint_index = int(self.waypoint_index)
        adaptive_waypoint_selection = bool(self.adaptive_waypoint_selection)
        adaptive_waypoint_threshold = float(self.adaptive_waypoint_threshold)
        midpoint_waypoint_selection = bool(self.midpoint_waypoint_selection)
        analytic_k_sample_temperature = float(self.analytic_k_sample_temperature)
        analytic_k_eval_sample_temperature = float(
            self.analytic_k_eval_sample_temperature
        )
        fixed_planning_k = self.fixed_planning_k
        analytic_waypoint_min_logstd = float(self.analytic_waypoint_min_logstd)
        analytic_waypoint_max_logstd = float(self.analytic_waypoint_max_logstd)

        def resolve_planning_k(
            state_latent,
            goal_latent,
            rotation,
            *,
            key=None,
            selection="argmax",
            temperature=1.0,
        ):
            """Zero-based K*: fixed ablation override, else analytic selection."""
            if fixed_planning_k is not None:
                batch = state_latent.shape[0]
                k_star = jnp.full((batch,), int(fixed_planning_k), dtype=jnp.int32)
                logits = analytic_k_logits(
                    state_latent,
                    goal_latent,
                    rotation,
                    max_waypoint_depth,
                )
                return k_star, logits
            return select_analytic_k(
                state_latent,
                goal_latent,
                rotation,
                max_waypoint_depth,
                key=key,
                selection=selection,
                temperature=temperature,
            )

        def high_waypoint_entropy(sigma_sq):
            tangent_dim = jnp.asarray(self.repr_dim - 1, dtype=sigma_sq.dtype)
            log_two_pi_e = jnp.asarray(np.log(2.0 * np.pi) + 1.0, dtype=sigma_sq.dtype)
            return 0.5 * tangent_dim * (
                log_two_pi_e + jnp.log(jnp.maximum(sigma_sq, 1e-8))
            )

        # Dimensions
        action_size = unwrapped_env.action_size
        state_size = unwrapped_env.state_dim
        goal_size = len(unwrapped_env.goal_indices)
        goal_indices = jnp.array(unwrapped_env.goal_indices)
        obs_size = state_size + goal_size
        control_horizon = self.control_horizon
        batch_size = self.batch_size

        buffer_config = (
            float(self.discounting),
            int(state_size),
            tuple(int(i) for i in unwrapped_env.goal_indices),
        )

        logging.info("state_size=%d, goal_size=%d, action_size=%d, obs_size=%d",
                      state_size, goal_size, action_size, obs_size)
        logging.info(
            "max_waypoint_depth=%d, waypoint_index=%d, control_horizon=%d",
            max_waypoint_depth,
            waypoint_index,
            control_horizon,
        )
        logging.info(
            "adaptive_waypoint_selection=%s, adaptive_waypoint_threshold=%.4f, "
            "midpoint_waypoint_selection=%s",
            adaptive_waypoint_selection,
            adaptive_waypoint_threshold,
            midpoint_waypoint_selection,
        )
        if fixed_planning_k is not None:
            logging.info(
                "FIXED k_star=%d (one-based K=%d); analytic K temperatures ignored",
                fixed_planning_k,
                fixed_planning_k + 1,
            )
        else:
            logging.info(
                "analytic_k_sample_temperature=%.4f (training=sample)",
                analytic_k_sample_temperature,
            )
            logging.info(
                "analytic_k_eval_sample_temperature=%.4f (inference/render=sample)",
                analytic_k_eval_sample_temperature,
            )

        # Seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key = jax.random.split(key, 4)
        key, sa_key, g_key, cayley_key, low_key = jax.random.split(key, 5)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # ---------------------------------------------------------------
        # Network setup
        # ---------------------------------------------------------------
        high_enc_kwargs = dict(
            network_width=self.h_dim,
            network_depth=self.n_hidden_high,
            skip_connections=self.skip_connections_high,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        low_enc_kwargs = dict(
            network_width=self.h_dim,
            network_depth=self.n_hidden_low,
            skip_connections=self.skip_connections_low,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        # Encoders
        sa_encoder = NormalizedEncoder(
            repr_dim=self.repr_dim, normalize=self.normalize_encoder, **low_enc_kwargs
        )
        sa_encoder_params = sa_encoder.init(sa_key, jnp.ones([1, state_size + action_size]))

        shared_encoder = NormalizedEncoder(
            repr_dim=self.repr_dim, normalize=self.normalize_encoder, **high_enc_kwargs
        )
        shared_encoder_params = shared_encoder.init(g_key, jnp.ones([1, goal_size]))
        
        cayley_v = CayleyTransform(dim=self.repr_dim, init_std=self.cayley_init_std)
        cayley_v_params = cayley_v.init(cayley_key, jnp.ones([1, self.repr_dim]))

        # Gradient clipping prevents NaN from L2-normalization gradients
        # in the encoder and waypoint actor during long mini-batch scans.
        max_grad_norm = 1.0

        critic_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), optax.adam(self.critic_lr)
        )
        if not self.learn_cayley:
            critic_tx = optax.chain(
                critic_tx,
                optax.masked(
                    optax.set_to_zero(),
                    {"sa_encoder": False, "shared_encoder": False, "cayley_v": True},
                ),
            )

        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder_params,
                "shared_encoder": shared_encoder_params,
                "cayley_v": cayley_v_params,
            },
            tx=critic_tx,
        )

        # Low actor
        low_actor = LowActor(action_size=action_size, **low_enc_kwargs)
        low_actor_state = TrainState.create(
            apply_fn=low_actor.apply,
            params=low_actor.init(low_key, jnp.ones([1, state_size + self.repr_dim])),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(self.low_policy_lr)),
        )

        # Alpha states (no clipping needed for scalar params).
        alpha_low_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.log(jnp.float32(self.alpha_low_init))},
            tx=optax.adam(self.alpha_low_lr),
        )
        alpha_high_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.log(jnp.float32(self.alpha_high_init))},
            tx=optax.adam(self.alpha_high_lr),
        )

        # Target entropies
        target_entropy_low = self.target_entropy_low_coef * action_size
        target_entropy_high = self.target_entropy_high_coef * (self.repr_dim - 1)

        # Training state
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            low_actor_state=low_actor_state,
            critic_state=critic_state,
            alpha_low_state=alpha_low_state,
            alpha_high_state=alpha_high_state,
        )

        # ---------------------------------------------------------------
        # Replay buffer
        # ---------------------------------------------------------------
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            buffer.sample_internal_repeat = jax.jit(buffer.sample_internal_repeat, static_argnames=("repetition_factor",),)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # ---------------------------------------------------------------
        # Context and networks dicts for loss functions
        # ---------------------------------------------------------------
        context = {
            **vars(self),
            "state_size": state_size,
            "action_size": action_size,
            "goal_size": goal_size,
            "obs_size": obs_size,
            "goal_indices": tuple(unwrapped_env.goal_indices),
            "target_entropy_low": target_entropy_low,
            "target_entropy_high": target_entropy_high,
            "max_waypoint_depth": max_waypoint_depth,
        }

        networks = {
            "sa_encoder": sa_encoder,
            "shared_encoder": shared_encoder,
            "cayley_v": cayley_v,
            "low_actor": low_actor,
        }

        def fixed_planning_i(k_star):
            if midpoint_waypoint_selection:
                return midpoint_planning_i(k_star)
            fixed_i = jnp.asarray(waypoint_index, dtype=jnp.int32)
            return jnp.where(k_star < fixed_i, k_star + 1, fixed_i)

        def fixed_direct_goal(k_star):
            if midpoint_waypoint_selection:
                return midpoint_planning_i(k_star) >= (k_star + 1)
            return k_star < waypoint_index

        def adaptive_high_waypoint_from_params(
            critic_params,
            low_actor_params,
            state,
            state_latent,
            goal_latent,
            rotation,
            alpha_high,
            k_star,
        ):
            candidates, valid_mask, candidate_i, k_star, selected_alignment = (
                closed_form_high_waypoint_candidates(
                    state_latent,
                    goal_latent,
                    rotation,
                    max_waypoint_depth,
                    k_star,
                )
            )
            waypoint, selected_i, reach_score, direct_goal, fallback = (
                select_adaptive_waypoint_by_low_reachability(
                    state,
                    candidates,
                    valid_mask,
                    candidate_i,
                    k_star,
                    low_actor,
                    low_actor_params,
                    sa_encoder,
                    critic_params["sa_encoder"],
                    self.energy_fn,
                    adaptive_waypoint_threshold,
                )
            )
            k_float = k_star.astype(state_latent.dtype)
            theta = jnp.arccos(
                jnp.clip(selected_alignment, -1.0 + 1e-6, 1.0 - 1e-6)
            )
            cos_step = jnp.cos(theta / (k_float + 1.0))
            alpha = jnp.asarray(alpha_high, dtype=state_latent.dtype)
            sigma_sq = alpha / jnp.maximum(2.0 * cos_step, 1e-6)
            logstd = 0.5 * jnp.log(sigma_sq)
            logstd = jnp.clip(
                logstd,
                jnp.asarray(analytic_waypoint_min_logstd, dtype=state_latent.dtype),
                jnp.asarray(analytic_waypoint_max_logstd, dtype=state_latent.dtype),
            )
            sigma_sq = jnp.exp(2.0 * logstd)
            return (
                waypoint,
                k_star,
                selected_i,
                reach_score,
                direct_goal,
                fallback,
                sigma_sq,
            )

        # ---------------------------------------------------------------
        # Actor steps
        # ---------------------------------------------------------------
        def evaluation_actor_step(
            training_state, env, env_state, extra_fields, key
        ):
            """Evaluation actor step with stochastic analytic-K selection."""
            obs = env_state.obs
            state = obs[:, :state_size]
            goal_values = obs[:, state_size:]

            critic_params = training_state.critic_state.params
            state_latent = shared_encoder.apply(critic_params["shared_encoder"], state[:, goal_indices])
            goal_latent = shared_encoder.apply(critic_params["shared_encoder"], goal_values)
            R = _cayley_rotation(critic_params["cayley_v"])

            wp_counter = env_state.info["wp_counter"]
            current_wp = env_state.info["current_waypoint"]
            current_planning_k = env_state.info["current_planning_k"]
            alpha_high = jnp.exp(training_state.alpha_high_state.params["log_alpha"])
            k_star, _ = resolve_planning_k(
                state_latent,
                goal_latent,
                R,
                key=key,
                selection="sample",
                temperature=analytic_k_eval_sample_temperature,
            )
            if adaptive_waypoint_selection:
                means_wp, k_star, _, _, _, _, _ = (
                    adaptive_high_waypoint_from_params(
                        critic_params,
                        training_state.low_actor_state.params,
                        state,
                        state_latent,
                        goal_latent,
                        R,
                        alpha_high,
                        k_star,
                    )
                )
            else:
                means_wp, _, k_star, _, _ = closed_form_high_waypoint(
                    state_latent,
                    goal_latent,
                    R,
                    max_waypoint_depth,
                    alpha_high,
                    waypoint_index=waypoint_index,
                    use_midpoint=midpoint_waypoint_selection,
                    min_logstd=analytic_waypoint_min_logstd,
                    max_logstd=analytic_waypoint_max_logstd,
                    k_star=k_star,
                )

            needs_new = (wp_counter % control_horizon == 0)[:, None]
            current_wp = jnp.where(needs_new, means_wp, current_wp)
            current_planning_k = jnp.where(needs_new[:, 0], k_star, current_planning_k)

            low_input = jnp.concatenate([state, current_wp], axis=-1)
            means, _ = low_actor.apply(training_state.low_actor_state.params, low_input)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)

            done_mask = nstate.done.astype(jnp.bool_)
            new_counter = jnp.where(done_mask, 0, wp_counter + 1)
            new_wp_val = jnp.where(done_mask[:, None], jnp.zeros_like(current_wp), current_wp)
            new_planning_k = jnp.where(done_mask, jnp.zeros_like(current_planning_k), current_planning_k)
            nstate_info = {
                **nstate.info,
                "wp_counter": new_counter,
                "current_waypoint": new_wp_val,
                "current_planning_k": new_planning_k,
            }
            nstate = nstate.replace(info=nstate_info)

            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(training_state, env, env_state, key, extra_fields):
            """Stochastic actor step for training rollouts."""
            key, k_sample_key, wp_key, act_key = jax.random.split(key, 4)

            wp_counter = env_state.info["wp_counter"]
            current_wp = env_state.info["current_waypoint"]
            current_planning_k = env_state.info["current_planning_k"]

            obs = env_state.obs
            state = obs[:, :state_size]
            goal_values = obs[:, state_size:]

            critic_params = training_state.critic_state.params
            state_latent = shared_encoder.apply(critic_params["shared_encoder"], state[:, goal_indices])
            goal_latent = shared_encoder.apply(critic_params["shared_encoder"], goal_values)
            R = _cayley_rotation(critic_params["cayley_v"])
            alpha_high = jnp.exp(training_state.alpha_high_state.params["log_alpha"])
            k_star, _ = resolve_planning_k(
                state_latent,
                goal_latent,
                R,
                key=k_sample_key,
                selection="sample",
                temperature=analytic_k_sample_temperature,
            )
            if adaptive_waypoint_selection:
                clean_wp, k_star, planning_i, reach_score, direct_goal_mask, fallback, sigma_sq = (
                    adaptive_high_waypoint_from_params(
                        critic_params,
                        training_state.low_actor_state.params,
                        state,
                        state_latent,
                        goal_latent,
                        R,
                        alpha_high,
                        k_star,
                    )
                )
                new_wp = add_tangent_noise_to_waypoint(
                    clean_wp,
                    sigma_sq,
                    direct_goal_mask,
                    goal_latent,
                    wp_key,
                )
                high_entropy = high_waypoint_entropy(sigma_sq)
                adaptive_direct_goal = direct_goal_mask
                adaptive_fallback = fallback
            else:
                new_wp, _, k_star, _, sigma_sq = closed_form_high_waypoint(
                    state_latent,
                    goal_latent,
                    R,
                    max_waypoint_depth,
                    alpha_high,
                    waypoint_index=waypoint_index,
                    use_midpoint=midpoint_waypoint_selection,
                    min_logstd=analytic_waypoint_min_logstd,
                    max_logstd=analytic_waypoint_max_logstd,
                    key=wp_key,
                    k_star=k_star,
                )
                high_entropy = high_waypoint_entropy(sigma_sq)
                direct_goal_mask = fixed_direct_goal(k_star)
                planning_i = fixed_planning_i(k_star)
                reach_score = jnp.zeros_like(high_entropy)
                adaptive_direct_goal = jnp.zeros_like(direct_goal_mask)
                adaptive_fallback = jnp.zeros_like(direct_goal_mask)

            needs_new = (wp_counter % control_horizon == 0)[:, None]
            high_planning_mask = needs_new[:, 0].astype(high_entropy.dtype)
            high_update_mask = high_planning_mask * (~direct_goal_mask).astype(high_entropy.dtype)
            planning_k = k_star.astype(high_entropy.dtype)
            planning_i = planning_i.astype(high_entropy.dtype)
            current_wp = jnp.where(needs_new, new_wp, current_wp)
            current_planning_k = jnp.where(needs_new[:, 0], k_star, current_planning_k)

            low_input = jnp.concatenate([state, current_wp], axis=-1)
            means, log_stds = low_actor.apply(
                training_state.low_actor_state.params, low_input
            )
            stds = jnp.exp(log_stds)
            action = nn.tanh(
                means + stds * jax.random.normal(act_key, shape=means.shape)
            )

            nstate = env.step(env_state, action)

            done_mask = nstate.done.astype(jnp.bool_)
            new_counter = jnp.where(done_mask, 0, wp_counter + 1)
            new_wp_val = jnp.where(done_mask[:, None], jnp.zeros_like(current_wp), current_wp)
            new_planning_k = jnp.where(done_mask, jnp.zeros_like(current_planning_k), current_planning_k)
            nstate_info = {
                **nstate.info,
                "wp_counter": new_counter,
                "current_waypoint": new_wp_val,
                "current_planning_k": new_planning_k,
            }
            nstate = nstate.replace(info=nstate_info)

            state_extras = {x: nstate.info[x] for x in extra_fields}
            transition = Transition(
                observation=env_state.obs,
                action=action,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )
            return nstate, (
                transition,
                high_entropy,
                high_update_mask,
                planning_k,
                planning_i,
                high_planning_mask,
                reach_score,
                adaptive_direct_goal.astype(high_entropy.dtype),
                adaptive_fallback.astype(high_entropy.dtype),
            )

        # ---------------------------------------------------------------
        # Experience collection
        # ---------------------------------------------------------------
        def update_high_alpha_from_rollout(
            training_state,
            high_entropy,
            high_update_mask,
        ):
            num_high_updates = jnp.sum(high_update_mask)
            denom = jnp.maximum(num_high_updates, 1.0)
            mean_high_entropy = jnp.sum(high_entropy * high_update_mask) / denom
            has_high_updates = num_high_updates > 0

            def alpha_high_loss_fn(alpha_params):
                entropy_gap = jax.lax.stop_gradient(
                    mean_high_entropy - target_entropy_high
                )
                return alpha_params["log_alpha"] * entropy_gap

            alpha_high_loss, alpha_high_grad = jax.value_and_grad(alpha_high_loss_fn)(
                training_state.alpha_high_state.params
            )
            alpha_high_loss = jnp.where(
                has_high_updates, alpha_high_loss, jnp.zeros_like(alpha_high_loss)
            )
            new_alpha_high_state = jax.lax.cond(
                has_high_updates,
                lambda alpha_state: alpha_state.apply_gradients(grads=alpha_high_grad),
                lambda alpha_state: alpha_state,
                training_state.alpha_high_state,
            )
            training_state = training_state.replace(alpha_high_state=new_alpha_high_state)

            log_alpha_high = training_state.alpha_high_state.params["log_alpha"]
            return training_state, {
                "alpha_high_loss": alpha_high_loss,
                "high_entropy": mean_high_entropy,
                "target_entropy_high": jnp.asarray(
                    target_entropy_high, dtype=mean_high_entropy.dtype
                ),
                "log_alpha_high": log_alpha_high,
                "alpha_high": jnp.exp(log_alpha_high),
            }

        def summarize_planning_from_rollout(
            high_planning_mask,
            planning_k,
            planning_i,
            adaptive_reach_score,
            adaptive_direct_goal,
            adaptive_fallback,
        ):
            denom = jnp.maximum(jnp.sum(high_planning_mask), 1.0)
            mean_planning_k = jnp.sum(planning_k * high_planning_mask) / denom
            planning_i_masked = jnp.where(
                high_planning_mask > 0,
                planning_i,
                jnp.zeros_like(planning_i),
            )
            metrics = {
                "planning_k_mean": mean_planning_k,
                "planning_K_one_based_mean": jnp.sum(
                    (planning_k + 1.0) * high_planning_mask
                ) / denom,
                "planning_i_mean": jnp.sum(planning_i * high_planning_mask) / denom,
                "planning_i_max": jnp.max(planning_i_masked),
            }
            metrics["adaptive_reach_score_mean"] = jnp.sum(
                adaptive_reach_score * high_planning_mask
            ) / denom
            metrics["adaptive_direct_goal_fraction"] = jnp.sum(
                adaptive_direct_goal * high_planning_mask
            ) / denom
            metrics["adaptive_fallback_fraction"] = jnp.sum(
                adaptive_fallback * high_planning_mask
            ) / denom
            return metrics

        @jax.jit
        def get_experience(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, high_data = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, next_key), high_data

            (
                (env_state, _),
                (
                    data,
                    high_entropy,
                    high_update_mask,
                    planning_k,
                    planning_i,
                    high_planning_mask,
                    adaptive_reach_score,
                    adaptive_direct_goal,
                    adaptive_fallback,
                ),
            ) = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            training_state, high_metrics = update_high_alpha_from_rollout(
                training_state, high_entropy, high_update_mask
            )
            planning_metrics = summarize_planning_from_rollout(
                high_planning_mask,
                planning_k,
                planning_i,
                adaptive_reach_score,
                adaptive_direct_goal,
                adaptive_fallback,
            )
            high_metrics = {**high_metrics, **planning_metrics}
            return training_state, env_state, buffer_state, high_metrics

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                training_state, env_state, buffer_state, _ = get_experience(
                    training_state, env_state, buffer_state, key
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ---------------------------------------------------------------
        # Update step
        # ---------------------------------------------------------------
        # def _has_nan(tree):
        #     return jnp.any(
        #         jnp.array([jnp.any(jnp.isnan(x)) for x in jax.tree_util.tree_leaves(tree)])
        #     )

        @jax.jit
        def update_networks(carry, batch_pair):
            training_state, key = carry
            (
                sw_low_critic_batch,
                sw_high_critic_batch,
                sw_low_actor_batch,
            ) = batch_pair
            key, c_key, la_key = jax.random.split(key, 3)

            training_state, critic_metrics = update_critics(
                context,
                networks,
                sw_low_critic_batch,
                sw_high_critic_batch,
                training_state,
                c_key,
            )
            # jax.debug.callback(
            #     lambda s, v: logging.warning("NaN in critic params at step %d", s) if v else None,
            #     training_state.gradient_steps,
            #     _has_nan(training_state.critic_state.params),
            # )

            training_state, low_metrics = update_low_actor_and_alpha(
                context, networks, sw_low_actor_batch, training_state, la_key
            )
            # jax.debug.callback(
            #     lambda s, v: logging.warning("NaN in low_actor params at step %d", s) if v else None,
            #     training_state.gradient_steps,
            #     _has_nan(training_state.low_actor_state.params),
            # )

            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )

            metrics = {}
            metrics.update(critic_metrics)
            metrics.update(low_metrics)
            metrics["critic_loss"] = critic_metrics["low_critic_loss"] + critic_metrics["high_critic_loss"]
            metrics["actor_loss"] = low_metrics["low_actor_loss"]
            metrics["alpha_loss"] = low_metrics["alpha_low_loss"]
            metrics["entropy"] = low_metrics["low_entropy"]
            metrics["log_alpha"] = low_metrics["log_alpha_low"]
            return (training_state, key), metrics

        def train_steps(training_state, buffer_state, key):
            """Sample replay buffer and run full mini-batch scan over update_networks."""
            sample_key, train_key, psi_key = jax.random.split(key, 3)
            sample_keys = jax.random.split(sample_key, 2)

            def build_sw_transitions(raw_transitions, sw_key):
                sw_keys = jax.random.split(sw_key, raw_transitions.observation.shape[0])
                sw_transitions = jax.vmap(flatten_batch_sw, in_axes=(None, 0, 0))(
                    buffer_config, raw_transitions, sw_keys
                )
                return jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                    sw_transitions,
                )

            # Draw one replay sample, then decouple branches by independent permutations.
            if self.sw_critic_repetition_factor > 1:
                buffer_state, critic_raw_transitions = replay_buffer.sample_repeat(
                    buffer_state, self.sw_critic_repetition_factor
                )
                sw_critic_transitions = build_sw_transitions(critic_raw_transitions, sample_keys[0])
                buffer_state, raw_transitions = replay_buffer.sample(buffer_state)
            else:
                buffer_state, raw_transitions = replay_buffer.sample(buffer_state)
                sw_critic_transitions = build_sw_transitions(raw_transitions, sample_keys[0])

            sw_actor_transitions = build_sw_transitions(raw_transitions, sample_keys[1])

            # Permute and mini-batch
            (
                perm_key_sw_low_critic,
                perm_key_sw_high_critic,
                perm_key_sw_low_actor,
            ) = jax.random.split(train_key, 3)

            def permute_and_batch(transitions, perm_key):
                perm = jax.random.permutation(perm_key, len(transitions.observation))
                transitions = jax.tree_util.tree_map(lambda x: x[perm], transitions)
                return jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
                    transitions,
                )

            sw_low_critic_batched = permute_and_batch(sw_critic_transitions, perm_key_sw_low_critic)
            sw_high_critic_batched = permute_and_batch(sw_critic_transitions, perm_key_sw_high_critic)
            sw_low_actor_batched = permute_and_batch(sw_actor_transitions, perm_key_sw_low_actor)

            ((training_state, _), metrics) = jax.lax.scan(
                update_networks,
                (training_state, train_key),
                (
                    sw_low_critic_batched,
                    sw_high_critic_batched,
                    sw_low_actor_batched,
                ),
            )

            return training_state, buffer_state, metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            exp_key, train_key = jax.random.split(key, 2)

            training_state, env_state, buffer_state, high_metrics = get_experience(
                training_state, env_state, buffer_state, exp_key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step
            )

            training_state, buffer_state, metrics = train_steps(
                training_state, buffer_state, train_key
            )
            metrics = {**metrics, **high_metrics}
            return training_state, env_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, unused_t):
                del unused_t
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, new_key)
                return (ts, bs, update_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        train_step_mult = self.train_step_multiplier

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit 
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, bs, metrics = training_step(ts, es, bs, k)
                (ts, bs, _), _ = scan_train_steps(train_step_mult - 1, ts, bs, update_key)
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        # ---------------------------------------------------------------
        # Prefill
        # ---------------------------------------------------------------
        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # ---------------------------------------------------------------
        # Evaluator
        # ---------------------------------------------------------------
        evaluator = ActorEvaluator(
            evaluation_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=eval_episode_length,
            key=eval_env_key,
        )

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        def make_policy(params):
            """Hierarchical policy for environment rendering.

            Carries ``(wp_counter, current_waypoint)`` so the high-level planner
            only fires every *control_horizon* steps, matching training.
            """
            def policy(obs, rng, init_state):
                wp_counter, current_wp = init_state

                state = obs[:, :state_size]
                goal_values = obs[:, state_size:]

                state_latent = shared_encoder.apply(params["critic"]["shared_encoder"], state[:, goal_indices])
                goal_latent = shared_encoder.apply(params["critic"]["shared_encoder"], goal_values)
                R = _cayley_rotation(params["critic"]["cayley_v"])
                k_star, _ = resolve_planning_k(
                    state_latent,
                    goal_latent,
                    R,
                    key=rng,
                    selection="sample",
                    temperature=analytic_k_eval_sample_temperature,
                )
                if adaptive_waypoint_selection:
                    alpha_high = jnp.exp(params["alpha_high"]["log_alpha"])
                    means_wp, _, _, _, _, _, _ = adaptive_high_waypoint_from_params(
                        params["critic"],
                        params["low_actor"],
                        state,
                        state_latent,
                        goal_latent,
                        R,
                        alpha_high,
                        k_star,
                    )
                else:
                    alpha_high = jnp.exp(params["alpha_high"]["log_alpha"])
                    means_wp, _, _, _, _ = closed_form_high_waypoint(
                        state_latent,
                        goal_latent,
                        R,
                        max_waypoint_depth,
                        alpha_high,
                        waypoint_index=waypoint_index,
                        use_midpoint=midpoint_waypoint_selection,
                        min_logstd=analytic_waypoint_min_logstd,
                        max_logstd=analytic_waypoint_max_logstd,
                        k_star=k_star,
                    )

                needs_new = (wp_counter % control_horizon == 0)[:, None]
                current_wp = jnp.where(needs_new, means_wp, current_wp)

                low_input = jnp.concatenate([state, current_wp], axis=-1)
                means, _ = low_actor.apply(params["low_actor"], low_input)
                action = nn.tanh(means)
                return action, (wp_counter + 1, current_wp)

            policy.init_state = (
                jnp.zeros(1, dtype=jnp.int32),
                jnp.zeros((1, self.repr_dim)),
            )
            return policy

        training_walltime = 0
        logging.info("starting C_slerp training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            formatted = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
            }
            alpha_high_metric_names = {
                "alpha_high_loss",
                "high_entropy",
                "target_entropy_high",
                "log_alpha_high",
                "alpha_high",
            }
            for metric_name, value in metrics.items():
                if metric_name in alpha_high_metric_names:
                    prefix = "alpha_high"
                    short_name = metric_name
                elif metric_name.startswith("recursive_"):
                    prefix = "recursive"
                    short_name = metric_name[len("recursive_"):]
                else:
                    prefix = "training"
                    short_name = metric_name
                formatted[f"{prefix}/{short_name}"] = jnp.mean(value).item()
            metrics = formatted
            current_step = int(training_state.env_steps.item())

            do_render = ne % config.visualization_interval == 0
            metrics = evaluator.run_evaluation(
                training_state,
                metrics,
            )

            logging.info("step: %d", current_step)

            render_params = {
                "low_actor": training_state.low_actor_state.params,
                "critic": training_state.critic_state.params,
                "alpha_high": training_state.alpha_high_state.params,
            }
            progress_fn(
                current_step,
                metrics,
                make_policy,
                render_params,
                unwrapped_vis_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                ckpt_params = {
                    "low_actor": training_state.low_actor_state.params,
                    "critic": training_state.critic_state.params,
                    "alpha_low": training_state.alpha_low_state.params,
                    "alpha_high": training_state.alpha_high_state.params,
                }
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, ckpt_params)

        total_steps = current_step
        if total_steps < config.total_env_steps:
            logging.warning(
                "Completed %d env steps, target was %d. "
                "Adjust num_evals or total_env_steps for exact coverage.",
                total_steps, config.total_env_steps,
            )

        logging.info("total steps: %s", total_steps)

        params = {
            "low_actor": training_state.low_actor_state.params,
            "critic": training_state.critic_state.params,
            "alpha_low": training_state.alpha_low_state.params,
            "alpha_high": training_state.alpha_high_state.params,
        }
        return make_policy, params, metrics
