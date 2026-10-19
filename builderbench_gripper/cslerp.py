import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax
JAX_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_max_size", 10 * 1024**3)  

import time
import json
import tyro
import numpy as np
import pprint
import wandb
import wandb_osh

import jax
import flax
import optax
import flax.linen as nn
import jax.numpy as jnp

from flax.training.train_state import TrainState
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.wrapper import wrap_env, PDWrapper, EvalWrapper
from utils.networks import save_params
from utils.jax import count_parameters
from builderbench.env_utils import make_env
from utils.buffer import TrajectoryUniformSamplingQueue
from utils.cslerp_networks import NormalizedEncoder, CayleyTransform, LowActor
from utils.cslerp_losses import (
    energy_fn,
    _cayley_rotation,
    update_critics,
    update_low_actor_and_alpha,
    goal_to_obs_shape,
)
from utils.hybrid_actions import sample_action


# ---------------------------------------------------------------------------
# Analytic high-level planner helpers
# ---------------------------------------------------------------------------

def _normalize_latent(x):
    return x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-6)


def _rotation_powers(rotation, num_powers):
    """Returns [I, rotation, ..., rotation^(num_powers - 1)]."""
    powers = []
    current = jnp.eye(rotation.shape[0], dtype=rotation.dtype)
    for _ in range(num_powers):
        powers.append(current)
        current = current @ rotation
    return jnp.stack(powers, axis=0)


def analytic_k_logits(state_latent, goal_latent, rotation, max_waypoint_depth):
    """Alignment logits for zero-based planning depths.

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
    state_latent,
    goal_latent,
    rotation,
    max_waypoint_depth,
    *,
    key: Optional[jax.Array] = None,
    selection: Literal["sample", "argmax"] = "argmax",
    temperature: float = 1.0,
):
    """Selects zero-based K from analytic alignment logits."""
    logits = analytic_k_logits(
        state_latent, goal_latent, rotation, max_waypoint_depth,
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


def _orthogonal_unit(x):
    basis_idx = jnp.argmin(jnp.abs(x), axis=-1)
    basis = jax.nn.one_hot(basis_idx, x.shape[-1], dtype=x.dtype)
    basis = basis - jnp.sum(basis * x, axis=-1, keepdims=True) * x
    return _normalize_latent(basis)


def _slerp(a, b, frac):
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


def closed_form_high_waypoint(
    state_latent, goal_latent, rotation, max_waypoint_depth, alpha_high, k_star,
    waypoint_index=1, min_logstd=-8.0, max_logstd=-2.0, key=None,
):
    """Closed-form high-level waypoint (mean) with online K selection."""
    state_latent = _normalize_latent(state_latent)
    goal_latent = _normalize_latent(goal_latent)

    # Row-vector convention: (R^(K+1) goal_col)^T == goal_row @ (R.T)^(K+1).
    rt_powers = _rotation_powers(rotation.T, max_waypoint_depth + 1)[1:]
    r_powers = _rotation_powers(rotation, max_waypoint_depth + 1)
    goal_t_all = jnp.einsum("bd,kde->bke", goal_latent, rt_powers)
    alignments = jnp.sum(state_latent[:, None, :] * goal_t_all, axis=-1)

    k_star = jnp.clip(k_star.astype(jnp.int32), 0, max_waypoint_depth - 1)
    goal_t = jnp.take_along_axis(goal_t_all, k_star[:, None, None], axis=1)[:, 0, :]
    selected_alignment = jnp.take_along_axis(alignments, k_star[:, None], axis=1)[:, 0]

    fixed_i = jnp.asarray(waypoint_index, dtype=jnp.int32)
    direct_goal = k_star < fixed_i
    i_star = jnp.where(direct_goal, k_star + 1, fixed_i)
    k_float = k_star.astype(state_latent.dtype)
    i_float = i_star.astype(state_latent.dtype)
    frac = (i_float / (k_float + 1.0))[:, None]
    t_i = _slerp(state_latent, goal_t, frac)
    r_i = jnp.take(r_powers, i_star, axis=0)
    mean = _normalize_latent(jnp.einsum("bd,bde->be", t_i, r_i))
    direct_goal_col = direct_goal[:, None]
    mean = jnp.where(direct_goal_col, goal_latent, mean)

    theta = jnp.arccos(jnp.clip(selected_alignment, -1.0 + 1e-6, 1.0 - 1e-6))
    cos_step = jnp.cos(theta / (k_float + 1.0))
    alpha = jnp.asarray(alpha_high, dtype=state_latent.dtype)
    sigma_sq = alpha / jnp.maximum(2.0 * cos_step, 1e-6)
    logstd = jnp.clip(0.5 * jnp.log(sigma_sq), min_logstd, max_logstd)
    sigma_sq = jnp.exp(2.0 * logstd)

    waypoint = mean
    if key is not None:
        eps = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        eps_tan = eps - jnp.sum(eps * mean, axis=-1, keepdims=True) * mean
        noisy = _normalize_latent(mean + jnp.sqrt(sigma_sq)[:, None] * eps_tan)
        waypoint = jnp.where(direct_goal_col, mean, noisy)

    return waypoint, mean, k_star, selected_alignment, sigma_sq


def closed_form_high_waypoint_candidates(
    state_latent, goal_latent, rotation, max_waypoint_depth, k_star,
):
    """All one-based closed-form candidates per zero-based K*.

    Returns candidates with shape [B, max_waypoint_depth, D]. Valid candidates
    are i in {1, ..., K* + 1}; the final valid candidate K* + 1 is the direct
    goal. Also returns (valid_mask, candidate_i, k_star, selected_alignment).
    """
    state_latent = _normalize_latent(state_latent)
    goal_latent = _normalize_latent(goal_latent)

    rt_powers = _rotation_powers(rotation.T, max_waypoint_depth + 1)[1:]
    r_powers = _rotation_powers(rotation, max_waypoint_depth + 1)
    goal_t_all = jnp.einsum("bd,kde->bke", goal_latent, rt_powers)
    alignments = jnp.sum(state_latent[:, None, :] * goal_t_all, axis=-1)

    k_star = jnp.clip(k_star.astype(jnp.int32), 0, max_waypoint_depth - 1)
    goal_t = jnp.take_along_axis(goal_t_all, k_star[:, None, None], axis=1)[:, 0, :]
    selected_alignment = jnp.take_along_axis(alignments, k_star[:, None], axis=1)[:, 0]

    candidate_i = jnp.arange(1, max_waypoint_depth + 1, dtype=jnp.int32)
    candidate_i_batch = candidate_i[None, :]
    final_i = k_star[:, None] + 1
    valid_mask = candidate_i_batch <= final_i
    direct_goal_mask = candidate_i_batch == final_i

    k_float = k_star.astype(state_latent.dtype)
    frac = (candidate_i.astype(state_latent.dtype)[None, :] / (k_float[:, None] + 1.0))[:, :, None]
    t_i = _slerp(state_latent[:, None, :], goal_t[:, None, :], frac)
    r_i = jnp.take(r_powers, candidate_i, axis=0)
    chain_candidates = _normalize_latent(jnp.einsum("bmd,mde->bme", t_i, r_i))
    goal_candidates = jnp.broadcast_to(goal_latent[:, None, :], chain_candidates.shape)
    candidates = jnp.where(direct_goal_mask[:, :, None], goal_candidates, chain_candidates)
    candidates = jnp.where(valid_mask[:, :, None], candidates, goal_candidates)

    return candidates, valid_mask, candidate_i, k_star, selected_alignment


def add_tangent_noise_to_waypoint(waypoint, sigma_sq, direct_goal_mask, goal_latent, key):
    if key is None:
        return waypoint
    eps = jax.random.normal(key, shape=waypoint.shape, dtype=waypoint.dtype)
    eps_tan = eps - jnp.sum(eps * waypoint, axis=-1, keepdims=True) * waypoint
    sigma = jnp.sqrt(sigma_sq).reshape(waypoint.shape[:-1] + (1,))
    noisy = _normalize_latent(waypoint + sigma * eps_tan)
    mask = direct_goal_mask.reshape(waypoint.shape[:-1] + (1,))
    return jnp.where(mask, goal_latent, noisy)


def select_adaptive_waypoint_by_low_reachability(
    state, candidate_waypoints, valid_mask, candidate_i, k_star,
    low_actor, low_actor_params, sa_encoder, sa_encoder_params, energy_name, threshold,
):
    """Selects the farthest reachable candidate under the current low critic."""
    batch_size, num_candidates = candidate_waypoints.shape[:2]
    state_tiled = jnp.broadcast_to(
        state[:, None, :], (batch_size, num_candidates, state.shape[-1])
    )
    low_input = jnp.concatenate([state_tiled, candidate_waypoints], axis=-1)
    means, _ = low_actor.apply(low_actor_params, low_input.reshape(batch_size * num_candidates, -1))
    actions, _, _ = sample_action(means, jnp.zeros_like(means), key=None, mode="eval")
    actions = actions.reshape(batch_size, num_candidates, -1)

    sa_input = jnp.concatenate([state_tiled, actions], axis=-1)
    sa_repr = sa_encoder.apply(
        sa_encoder_params, sa_input.reshape(batch_size * num_candidates, -1)
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
        candidate_waypoints, selected_index[:, None, None], axis=1
    )[:, 0, :]
    selected_score = jnp.take_along_axis(scores, selected_index[:, None], axis=1)[:, 0]
    direct_goal = selected_i == (k_star + 1)
    fallback = ~has_reachable
    return selected_waypoint, selected_i, selected_score, direct_goal, fallback


# ---------------------------------------------------------------------------
# Args / training state / transition
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # experiment
    agent: str = "cslerp"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    # logging and checkpointing
    track: bool = True
    wandb_project_name: str = "sub_test"
    wandb_entity: str = "RHCRL"
    wandb_mode: str = "online"
    wandb_dir: str = "./"
    wandb_group: str = "default"
    wandb_name_tag: str = ""

    num_eval_steps: int = 800
    num_reset_steps: int = 800
    save_checkpoint: bool = True

    record_videos: bool = True
    video_every_n_evals: int = 5
    video_fps: int = 10

    # environment
    env_id: str = "gripper-creative-4-task2"
    num_envs: int = 1024
    num_eval_envs: int = 128
    env_early_termination: bool = False
    env_episode_length: int = 500 
    eval_episode_length: Optional[int] = 1000
    permutation_invariant_reward: bool = False
    permute_init_at_reset: bool = True        
    goal_noise_at_reset: bool = True          
    xyz_scale: Optional[float] = 0.005

    # control mode
    use_pd: bool = False
    duration: int = 5

    # core training
    num_timesteps: int = 2_000_000_000
    rollout_length: int = 64
    discount: float = 0.99
    max_replay_size: int = 10000
    min_replay_size: int = 2000
    repetition_factor: int = 12       # CRTR on SW critic relabels

    # learning rates
    low_policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_force_lr: float = 3e-4
    alpha_high_lr: float = 3e-3

    # architecture
    rep_size: int = 512
    h_dim: int = 1024
    n_hidden_low: int = 17
    n_hidden_actor: int = 17
    skip_connections_low: int = 4
    use_relu: bool = False
    use_ln: bool = True
    g_encoder_full_state: bool = False

    # contrastive
    low_energy_fn: Literal["dot", "l2"] = "dot"
    low_contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce"] = "bwd_infonce"
    high_contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce"] = "sym_infonce"
    critic_temperature: float = 0.1
    cayley_fro_reg_coef: float = 1e-4
    high_critic_loss_coef: float = 2.0
    actor_also_temperature: bool = False
    train_shared_in_low: bool = True

    # hierarchical / planning
    control_horizon: int = 10
    max_waypoint_depth: int = 4
    waypoint_index: int = 1
    adaptive_waypoint_selection: bool = True
    adaptive_waypoint_threshold: float = 0.4
    analytic_waypoint_min_logstd: float = -8.0
    analytic_waypoint_max_logstd: float = -2.0
    # Temperature for categorical sampling over analytic K logits during training rollouts.
    analytic_k_sample_temperature: float = 0.1
    # Eval / video analytic-K selection.
    analytic_k_eval_selection: Literal["sample", "argmax"] = "sample"
    analytic_k_eval_temperature: float = 0.1

    # entropy temperatures
    alpha_force_init: float = 0.01
    alpha_high_init: float = 0.01
    target_entropy_force_coef: float = -1.0   
    target_entropy_high_coef: float = -3.0


class CriticTrainState(TrainState):
    """Critic TrainState that also stores a lagged g_encoder (psi) target."""
    g_encoder_target_params: Any


@flax.struct.dataclass
class C_slerpTrainingState:
    env_steps: np.ndarray
    gradient_steps: np.ndarray
    low_actor_state: TrainState
    critic_state: CriticTrainState
    alpha_force_state: TrainState
    alpha_high_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()


def main(args: Args):

    if not np.isfinite(args.analytic_k_sample_temperature) or args.analytic_k_sample_temperature <= 0:
        raise ValueError("analytic_k_sample_temperature must be finite and > 0")
    if args.analytic_k_eval_selection not in ("sample", "argmax"):
        raise ValueError("analytic_k_eval_selection must be 'sample' or 'argmax'")
    if args.analytic_k_eval_selection == "sample" and (
        not np.isfinite(args.analytic_k_eval_temperature) or args.analytic_k_eval_temperature <= 0
    ):
        raise ValueError(
            "analytic_k_eval_temperature must be finite and > 0 when "
            "analytic_k_eval_selection='sample'"
        )
    if args.max_waypoint_depth < 1:
        raise ValueError("max_waypoint_depth must be >= 1")
    if args.waypoint_index < 1:
        raise ValueError("waypoint_index must be >= 1")

    args.num_training_step = args.num_timesteps // (args.num_envs * args.rollout_length)
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Eval every {args.num_training_steps_per_eval} training steps ({args.num_eval_steps} evals total)")

    # If wandb_name_tag is set, use it as the exact run/checkpoint name; otherwise
    # fall back to the env/seed/timestamp default.
    args.exp_name = (
        args.wandb_name_tag
        if args.wandb_name_tag != ""
        else (
            f"{args.env_id}__{args.seed}__"
            f"{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"
        )
    )

    if args.track:
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity, mode=args.wandb_mode,
            dir=args.wandb_dir, group=args.wandb_group, name=args.exp_name,
            config=vars(args), save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    (key, key_buffer, key_env, key_eval, key_actor, key_sa, key_g, key_cayley) = jax.random.split(key, 8)

    # Environment (raw or PD waypoint control)
    env_class, default_config = make_env(args)
    raw_eval_episode_length = (
        args.eval_episode_length
        if args.eval_episode_length is not None
        else default_config.episode_length
    )
    if args.use_pd:
        assert default_config.episode_length % args.duration == 0, (
            "Environment episode length must be divisible by duration"
        )
        assert raw_eval_episode_length % args.duration == 0, (
            "Eval episode length must be divisible by duration"
        )
        episode_length = default_config.episode_length // args.duration
        eval_episode_length = raw_eval_episode_length // args.duration
        env = wrap_env(PDWrapper(env_class(config=default_config), duration=args.duration), episode_length)
        eval_env = wrap_env(PDWrapper(env_class(config=default_config), duration=args.duration), eval_episode_length)
        print(f"Control mode: PD waypoint controls (duration={args.duration})")
    else:
        episode_length = default_config.episode_length
        eval_episode_length = raw_eval_episode_length
        env = wrap_env(env_class(config=default_config), episode_length)
        eval_env = wrap_env(env_class(config=default_config), eval_episode_length)
        print("Control mode: raw controls")
    print(f"Episode length: train={episode_length}, eval={eval_episode_length}")

    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    force_dim = env.unwrapped.force_dim
    num_cubes = env.unwrapped.num_cubes
    goal_size = env.goal_size
    rep_size = args.rep_size
    assert action_size == force_dim, (
        f"gripper action_size={action_size} != force_dim({force_dim})"
    )
    log_data_metric_keys = []
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # -----------------------------------------------------------------
    # Networks
    # -----------------------------------------------------------------
    enc_kwargs = dict(
        network_width=args.h_dim, network_depth=args.n_hidden_low,
        skip_connections=args.skip_connections_low, use_relu=args.use_relu, use_ln=args.use_ln,
    )
    sa_encoder = NormalizedEncoder(repr_dim=rep_size, normalize=(args.low_energy_fn == "dot"), **enc_kwargs)
    g_encoder = NormalizedEncoder(repr_dim=rep_size, normalize=True, **enc_kwargs)  # shared psi, always unit-norm
    # When g_encoder_full_state is set, psi encodes FULL STATES: training feeds real
    # observations (and future observations) and queries augment goals via goal_to_obs_shape.
    g_enc_input_dim = obs_size if args.g_encoder_full_state else goal_size
    cayley_v = CayleyTransform(dim=rep_size)
    low_actor = LowActor(force_dim=force_dim, network_width=args.h_dim, network_depth=args.n_hidden_actor, skip_connections=args.skip_connections_low, use_relu=args.use_relu, use_ln=args.use_ln)

    sa_encoder_params = sa_encoder.init(key_sa, np.ones([1, obs_size + action_size]))
    g_encoder_params = g_encoder.init(key_g, np.ones([1, g_enc_input_dim], dtype=np.float32))
    cayley_v_params = cayley_v.init(key_cayley, np.ones([1, rep_size]))
    low_actor_params = low_actor.init(key_actor, np.ones([1, obs_size + rep_size]))

    max_grad_norm = 1.0
    critic_state = CriticTrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params, "cayley_v": cayley_v_params},
        tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(args.critic_lr)),
        g_encoder_target_params=jax.tree_util.tree_map(jnp.array, g_encoder_params),
    )
    low_actor_state = TrainState.create(
        apply_fn=low_actor.apply, params=low_actor_params,
        tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(args.low_policy_lr)),
    )
    alpha_force_state = TrainState.create(
        apply_fn=None, params={"log_alpha": jnp.log(jnp.float32(args.alpha_force_init))},
        tx=optax.adam(args.alpha_force_lr),
    )
    alpha_high_state = TrainState.create(
        apply_fn=None, params={"log_alpha": jnp.log(jnp.float32(args.alpha_high_init))},
        tx=optax.adam(args.alpha_high_lr),
    )

    target_entropy_force = args.target_entropy_force_coef * force_dim
    target_entropy_high = args.target_entropy_high_coef * (rep_size - 1)

    training_state = C_slerpTrainingState(
        env_steps=np.zeros((), dtype=np.float64),
        gradient_steps=np.zeros((), dtype=np.float64),
        low_actor_state=low_actor_state,
        critic_state=critic_state,
        alpha_force_state=alpha_force_state,
        alpha_high_state=alpha_high_state,
    )

    print(
        f"\nParams: low_actor={count_parameters(low_actor_state.params)}, "
        f"critic={count_parameters(critic_state.params)}\n"
        f"Continuous gripper action: force_dim={force_dim}, num_cubes={num_cubes}, "
        f"action_size={action_size}; H*_force={target_entropy_force}; "
        f"xyz_scale={getattr(default_config.action_scale, 'xyz_scale', None)}\n"
    )
    eval_k_desc = (
        f"sample(T={args.analytic_k_eval_temperature:g})"
        if args.analytic_k_eval_selection == "sample"
        else "argmax"
    )
    print(
        f"Analytic K: max_waypoint_depth={args.max_waypoint_depth}, "
        f"waypoint_index={args.waypoint_index}, "
        f"analytic_k_sample_temperature={args.analytic_k_sample_temperature} "
        f"(training=sample, eval/video={eval_k_desc})"
    )

    # -----------------------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------------------
    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,)),
        achieved_goal=jnp.zeros((goal_size,)),
        action=jnp.zeros((action_size,)),
        extras={"state_extras": {"traj_id": 0.0}},
    )

    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size, dummy_data_sample=dummy_transition,
            sample_batch_size=args.num_envs, num_envs=args.num_envs,
            sequence_length=episode_length + 1,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(key_buffer)

    # -----------------------------------------------------------------
    # Loss config + networks dict
    # -----------------------------------------------------------------
    networks = {"sa_encoder": sa_encoder, "g_encoder": g_encoder, "low_actor": low_actor}
    loss_config = {
        "low_energy_fn": args.low_energy_fn,
        "low_contrastive_loss_fn": args.low_contrastive_loss_fn,
        "high_contrastive_loss_fn": args.high_contrastive_loss_fn,
        "critic_temperature": args.critic_temperature,
        "cayley_fro_reg_coef": args.cayley_fro_reg_coef,
        "high_critic_loss_coef": args.high_critic_loss_coef,
        "actor_also_temperature": args.actor_also_temperature,
        "train_shared_in_low": args.train_shared_in_low,
        "target_entropy_force": target_entropy_force,
        "g_encoder_full_state": args.g_encoder_full_state,
    }

    control_horizon = args.control_horizon
    max_waypoint_depth = args.max_waypoint_depth

    # -----------------------------------------------------------------
    # Planning helpers (closures over modules / config)
    # -----------------------------------------------------------------
    def _psi_state(g_params, obs, goal_space_state):
        """psi of a STATE. When g_encoder_full_state, psi is trained on full observations,
        so feed the real obs; otherwise feed the goal-space achieved_goal."""
        return g_encoder.apply(g_params, obs if args.g_encoder_full_state else goal_space_state)

    def _psi_goal(g_params, goal):
        """psi of a bare GOAL (query). When g_encoder_full_state, augment the goal to the
        observation layout via goal_to_obs_shape before encoding; else encode goal-space."""
        return g_encoder.apply(g_params,
                               goal_to_obs_shape(goal, obs_size) if args.g_encoder_full_state else goal)

    def _high_entropy(sigma_sq):
        tangent_dim = jnp.asarray(rep_size - 1, dtype=sigma_sq.dtype)
        log_two_pi_e = jnp.asarray(np.log(2.0 * np.pi) + 1.0, dtype=sigma_sq.dtype)
        return 0.5 * tangent_dim * (log_two_pi_e + jnp.log(jnp.maximum(sigma_sq, 1e-8)))

    def adaptive_high_waypoint(critic_params, low_actor_params, obs, psi_s, psi_g, R, alpha_high, k_star):
        candidates, valid_mask, candidate_i, k_star, selected_alignment = (
            closed_form_high_waypoint_candidates(psi_s, psi_g, R, max_waypoint_depth, k_star)
        )
        waypoint, selected_i, reach_score, direct_goal, fallback = (
            select_adaptive_waypoint_by_low_reachability(
                obs, candidates, valid_mask, candidate_i, k_star,
                low_actor, low_actor_params, sa_encoder, critic_params["sa_encoder"],
                args.low_energy_fn, args.adaptive_waypoint_threshold,
            )
        )
        k_float = k_star.astype(psi_s.dtype)
        theta = jnp.arccos(jnp.clip(selected_alignment, -1.0 + 1e-6, 1.0 - 1e-6))
        cos_step = jnp.cos(theta / (k_float + 1.0))
        sigma_sq = jnp.asarray(alpha_high, dtype=psi_s.dtype) / jnp.maximum(2.0 * cos_step, 1e-6)
        logstd = jnp.clip(0.5 * jnp.log(sigma_sq),
                          args.analytic_waypoint_min_logstd, args.analytic_waypoint_max_logstd)
        sigma_sq = jnp.exp(2.0 * logstd)
        return waypoint, k_star, selected_i, reach_score, direct_goal, fallback, sigma_sq

    def compute_high_waypoint(critic_params, low_actor_params, alpha_high,
                              obs, achieved_goal, target_goal, k_star, noise_key=None,
                              psi_s=None, psi_g=None, R=None):
        """Returns (clean_wp, used_wp, k_star, selected_i, direct_goal_mask, sigma_sq).

        ``k_star`` is zero-based. ``selected_i`` is the one-based waypoint index
        i in {1, ..., k_star + 1} that the planner committed to (adaptive: chosen
        by low reachability; fixed: the configured waypoint_index, or k_star + 1
        once it falls back to direct goal).

        ``psi_s`` / ``psi_g`` / ``R`` may be supplied by callers that already
        computed them, to avoid a duplicate g_encoder forward over the batch.
        """
        if psi_s is None:
            psi_s = _psi_state(critic_params["g_encoder"], obs, achieved_goal)
        if psi_g is None:
            psi_g = _psi_goal(critic_params["g_encoder"], target_goal)
        if R is None:
            R = _cayley_rotation(critic_params["cayley_v"])
        if args.adaptive_waypoint_selection:
            clean_wp, k_star, selected_i, _, direct_goal, _, sigma_sq = adaptive_high_waypoint(
                critic_params, low_actor_params, obs, psi_s, psi_g, R, alpha_high, k_star
            )
            if noise_key is None:
                used_wp = clean_wp
            else:
                used_wp = add_tangent_noise_to_waypoint(clean_wp, sigma_sq, direct_goal, psi_g, noise_key)
        else:
            used_wp, clean_wp, k_star, _, sigma_sq = closed_form_high_waypoint(
                psi_s, psi_g, R, max_waypoint_depth, alpha_high, k_star,
                waypoint_index=args.waypoint_index,
                min_logstd=args.analytic_waypoint_min_logstd,
                max_logstd=args.analytic_waypoint_max_logstd, key=noise_key,
            )
            direct_goal = k_star < args.waypoint_index
            fixed_i = jnp.asarray(args.waypoint_index, dtype=jnp.int32)
            selected_i = jnp.where(direct_goal, k_star + 1, fixed_i)
        return clean_wp, used_wp, k_star, selected_i, direct_goal, sigma_sq

    def init_waypoint_state(n):
        return (jnp.zeros((n,), dtype=jnp.int32),
                jnp.zeros((n, rep_size)),
                jnp.zeros((n,), dtype=jnp.int32))

    # -----------------------------------------------------------------
    # Rollout actor step (stochastic) with waypoint carry
    # -----------------------------------------------------------------
    def actor_step(training_state, env_state, waypoint_state, key, extra_fields, metrics_fields):
        key, k_sample_key, wp_key, act_key = jax.random.split(key, 4)
        wp_counter, current_wp, current_planning_k = waypoint_state

        obs = env_state.obs
        achieved_goal = env_state.info["achieved_goal"]
        target_goal = env_state.info["target_goal"]

        critic_params = training_state.critic_state.params
        low_actor_params = training_state.low_actor_state.params
        alpha_high = jnp.exp(training_state.alpha_high_state.params["log_alpha"])

        needs_new = (wp_counter % control_horizon == 0)

        # The planner output is only consumed on refresh steps, so skip it entirely
        # when no env needs a new waypoint.
        def refresh_plan(_):
            psi_s = _psi_state(critic_params["g_encoder"], obs, achieved_goal)
            psi_g = _psi_goal(critic_params["g_encoder"], target_goal)
            R = _cayley_rotation(critic_params["cayley_v"])
            k_star, _ = select_analytic_k(
                psi_s, psi_g, R, max_waypoint_depth,
                key=k_sample_key, selection="sample",
                temperature=args.analytic_k_sample_temperature,
            )

            _, used_wp, k_star, _, direct_goal, sigma_sq = compute_high_waypoint(
                critic_params, low_actor_params, alpha_high,
                obs, achieved_goal, target_goal, k_star, noise_key=wp_key,
                psi_s=psi_s, psi_g=psi_g, R=R,
            )

            return used_wp, k_star, direct_goal, sigma_sq

        def reuse_plan(_):
            # Masked out downstream by needs_new, so the placeholder values cannot
            # reach any metric: high_update_mask is 0 and the jnp.where below keeps
            # the carried waypoint.
            return (current_wp,
                    current_planning_k,
                    jnp.zeros_like(current_planning_k, dtype=jnp.bool_),
                    jnp.zeros((obs.shape[0],), dtype=current_wp.dtype))

        used_wp, k_star, direct_goal, sigma_sq = jax.lax.cond(
            jnp.any(needs_new), refresh_plan, reuse_plan, operand=None
        )

        current_wp = jnp.where(needs_new[:, None], used_wp, current_wp)
        current_planning_k = jnp.where(needs_new, k_star, current_planning_k)

        low_input = jnp.concatenate([obs, current_wp], axis=-1)
        means, log_stds = low_actor.apply(low_actor_params, low_input)
        action, _, _ = sample_action(means, log_stds, act_key, mode="train")

        next_env_state = env.step(env_state, action)

        done_mask = next_env_state.done.astype(jnp.bool_)
        new_counter = jnp.where(done_mask, 0, wp_counter + 1)
        new_wp = jnp.where(done_mask[:, None], jnp.zeros_like(current_wp), current_wp)
        new_planning_k = jnp.where(done_mask, jnp.zeros_like(current_planning_k), current_planning_k)
        new_waypoint_state = (new_counter, new_wp, new_planning_k)

        high_entropy = _high_entropy(sigma_sq)
        high_update_mask = needs_new.astype(high_entropy.dtype) * (~direct_goal).astype(high_entropy.dtype)

        state_extras = {x: next_env_state.info[x] for x in extra_fields}
        metrics = {x: next_env_state.metrics[x] for x in metrics_fields}
        transition = Transition(
            observation=env_state.obs,
            achieved_goal=env_state.info["achieved_goal"],
            action=action,
            extras={"state_extras": state_extras},
        )
        aux = (high_entropy, high_update_mask, k_star.astype(jnp.float32), needs_new.astype(jnp.float32))
        return next_env_state, new_waypoint_state, transition, metrics, aux

    def update_high_alpha(training_state, high_entropy, high_update_mask):
        num_high = jnp.sum(high_update_mask)
        denom = jnp.maximum(num_high, 1.0)
        mean_high_entropy = jnp.sum(high_entropy * high_update_mask) / denom
        has_high = num_high > 0

        def alpha_high_loss_fn(alpha_params):
            gap = jax.lax.stop_gradient(mean_high_entropy - target_entropy_high)
            return alpha_params["log_alpha"] * gap

        _, grad = jax.value_and_grad(alpha_high_loss_fn)(training_state.alpha_high_state.params)
        new_alpha_high_state = jax.lax.cond(
            has_high,
            lambda s: s.apply_gradients(grads=grad),
            lambda s: s,
            training_state.alpha_high_state,
        )
        training_state = training_state.replace(alpha_high_state=new_alpha_high_state)
        log_alpha_high = training_state.alpha_high_state.params["log_alpha"]
        return training_state, {
            "high_entropy": mean_high_entropy,
            "target_entropy_high": jnp.asarray(target_entropy_high, dtype=mean_high_entropy.dtype),
            "alpha_high": jnp.exp(log_alpha_high),
            "log_alpha_high": log_alpha_high,
        }

    @jax.jit
    def data_collect_step(training_state, env_state, waypoint_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, waypoint_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, waypoint_state, transition, metrics, aux = actor_step(
                training_state, env_state, waypoint_state, current_key,
                extra_fields=("traj_id",), metrics_fields=log_data_metric_keys,
            )
            return (env_state, waypoint_state, next_key), (transition, metrics, aux)

        (env_state, waypoint_state, _), (data, metrics, aux) = jax.lax.scan(
            f, (env_state, waypoint_state, key), (), length=args.rollout_length
        )
        high_entropy, high_update_mask, planning_k, planning_mask = aux

        buffer_state = replay_buffer.insert(buffer_state, data)
        training_state = training_state.replace(
            env_steps=training_state.env_steps + (args.num_envs * args.rollout_length),
        )
        training_state, high_metrics = update_high_alpha(training_state, high_entropy, high_update_mask)

        denom = jnp.maximum(jnp.sum(planning_mask), 1.0)
        high_metrics["planning_k_mean"] = jnp.sum(planning_k * planning_mask) / denom
        high_metrics["planning_K_one_based_mean"] = jnp.sum((planning_k + 1.0) * planning_mask) / denom
        # Fraction of env-steps that re-planned. Expected to be ~1/control_horizon;
        # a higher value means the envs have desynced and the planner gate is no
        # longer skipping work.
        high_metrics["planning_refresh_fraction"] = jnp.mean(planning_mask)
        metrics = {**metrics, **high_metrics}
        return training_state, env_state, waypoint_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, waypoint_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, waypoint_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            training_state, env_state, waypoint_state, buffer_state, _ = data_collect_step(
                training_state, env_state, waypoint_state, buffer_state, key
            )
            return (training_state, env_state, waypoint_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f, (training_state, env_state, waypoint_state, buffer_state, key), (),
            length=int(np.ceil(args.min_replay_size / args.rollout_length)),
        )[0]

    # -----------------------------------------------------------------
    # Relabeling helpers (builderbench-style: one transition per trajectory)
    # -----------------------------------------------------------------
    # Pick-timestep-first relabels: each per-trajectory flatten samples ONE timestep and
    # one future for it (O(seq_len)).
    def sw_relabel(trajs, key):
        batch_keys = jax.random.split(key, trajs.observation.shape[0])
        return jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_pick_fn, in_axes=(None, 0, 0)
        )((args.discount,), trajs, batch_keys)

    def crtr_repeat(trajs, factor):
        n_unique = trajs.observation.shape[0] // factor
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:n_unique], factor, axis=0), trajs
        )

    @jax.jit
    def sgd_step(carry, unused_t):
        del unused_t
        training_state, buffer_state, key = carry
        key, k_lc, k_hc, k_act, k_actor_upd, k_critic = jax.random.split(key, 6)

        buffer_state, trajs = replay_buffer.sample(buffer_state)

        critic_trajs = crtr_repeat(trajs, args.repetition_factor) if args.repetition_factor > 1 else trajs
        low_c = sw_relabel(critic_trajs, k_lc)
        high_c = sw_relabel(critic_trajs, k_hc)
        actor_b = sw_relabel(trajs, k_act)

        training_state, critic_metrics = update_critics(
            loss_config, networks, low_c, high_c, training_state, k_critic
        )
        training_state, low_metrics = update_low_actor_and_alpha(
            loss_config, networks, actor_b, training_state, k_actor_upd
        )
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(critic_metrics)
        metrics.update(low_metrics)
        metrics["critic_loss"] = critic_metrics["low_critic_loss"] + critic_metrics["high_critic_loss"]
        metrics["actor_loss"] = low_metrics["low_actor_loss"]
        return (training_state, buffer_state, key), metrics

    @jax.jit
    def learn_step(training_state, buffer_state, key):
        num_sgd_steps = args.rollout_length
        (training_state, buffer_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, buffer_state, key), (), length=num_sgd_steps
        )
        return training_state, buffer_state, metrics

    # -----------------------------------------------------------------
    # C_slerp waypoints evaluation (threads the waypoint carry)
    # -----------------------------------------------------------------
    eval_env_wrapped = EvalWrapper(eval_env)

    def eval_action(training_state, env_state, waypoint_state, key, command_direct_goal=False):
        wp_counter, current_wp, current_planning_k = waypoint_state
        obs = env_state.obs
        achieved_goal = env_state.info["achieved_goal"]
        target_goal = env_state.info["target_goal"]

        critic_params = training_state.critic_state.params
        low_actor_params = training_state.low_actor_state.params
        alpha_high = jnp.exp(training_state.alpha_high_state.params["log_alpha"])

        # Direct-goal eval: bypass the planner and command psi(target_goal) directly.
        if command_direct_goal:
            current_wp = _psi_goal(critic_params["g_encoder"], target_goal)
            low_input = jnp.concatenate([obs, current_wp], axis=-1)
            means, log_stds = low_actor.apply(low_actor_params, low_input)
            action, _, _ = sample_action(means, log_stds, key=None, mode="eval")
            return action, (current_wp, current_planning_k)

        needs_new = (wp_counter % control_horizon == 0)

        def refresh_plan(_):
            psi_s = _psi_state(critic_params["g_encoder"], obs, achieved_goal)
            psi_g = _psi_goal(critic_params["g_encoder"], target_goal)
            R = _cayley_rotation(critic_params["cayley_v"])
            k_star, _ = select_analytic_k(
                psi_s, psi_g, R, max_waypoint_depth,
                key=key,
                selection=args.analytic_k_eval_selection,
                temperature=args.analytic_k_eval_temperature,
            )
            _, used_wp, k_star, _, _, _ = compute_high_waypoint(
                critic_params, low_actor_params, alpha_high,
                obs, achieved_goal, target_goal, k_star, noise_key=None,
                psi_s=psi_s, psi_g=psi_g, R=R,
            )
            return used_wp, k_star

        def reuse_plan(_):
            return current_wp, current_planning_k

        used_wp, k_star = jax.lax.cond(
            jnp.any(needs_new), refresh_plan, reuse_plan, operand=None
        )

        current_wp = jnp.where(needs_new[:, None], used_wp, current_wp)
        current_planning_k = jnp.where(needs_new, k_star, current_planning_k)

        low_input = jnp.concatenate([obs, current_wp], axis=-1)
        means, log_stds = low_actor.apply(low_actor_params, low_input)
        action, _, _ = sample_action(means, log_stds, key=None, mode="eval")
        return action, (current_wp, current_planning_k)

    @partial(jax.jit, static_argnums=(2,))
    def run_eval_unroll(training_state, key, command_direct_goal=False):
        reset_keys = jax.random.split(key, args.num_eval_envs)
        env_state = eval_env_wrapped.reset(reset_keys)
        waypoint_state = init_waypoint_state(args.num_eval_envs)

        def body(i, carry):
            env_state, waypoint_state, key = carry
            key, step_key = jax.random.split(key)
            action, (current_wp, current_planning_k) = eval_action(
                training_state, env_state, waypoint_state, step_key, command_direct_goal
            )
            next_env_state = eval_env_wrapped.step(env_state, action)
            done_mask = next_env_state.done.astype(jnp.bool_)
            wp_counter = waypoint_state[0]
            new_counter = jnp.where(done_mask, 0, wp_counter + 1)
            new_wp = jnp.where(done_mask[:, None], jnp.zeros_like(current_wp), current_wp)
            new_planning_k = jnp.where(done_mask, jnp.zeros_like(current_planning_k), current_planning_k)
            return (next_env_state, (new_counter, new_wp, new_planning_k), key)

        final_state, _, _ = jax.lax.fori_loop(0, eval_episode_length, body, (env_state, waypoint_state, key))
        return final_state

    def _collect_eval_metrics(eval_state, prefix, eval_time):
        eval_metrics = eval_state.info["eval_metrics"]
        metrics = {}
        for name, value in eval_metrics.episode_metrics.items():
            metrics[f"{prefix}/episode_{name}"] = float(np.mean(value))
            if name in ["success", "easy_success", "hard_success", "stacking"]:
                metrics[f"{prefix}/episode_{name}_rate"] = float(np.mean(np.asarray(value) > 0))
        metrics[f"{prefix}/avg_episode_length"] = float(np.mean(eval_metrics.episode_steps))
        metrics[f"{prefix}/epoch_eval_time"] = eval_time
        return metrics

    def run_evaluation(training_state, training_metrics, key):
        key_wp, key_dg = jax.random.split(key)

        # Primary eval: planner-derived waypoints (eval/ prefix).
        t = time.time()
        eval_state = run_eval_unroll(training_state, key_wp, False)
        eval_state.info["eval_metrics"].active_episodes.block_until_ready()
        metrics = _collect_eval_metrics(eval_state, "eval", time.time() - t)

        eval_state_dg = run_eval_unroll(training_state, key_dg, True)
        eval_state_dg.info["eval_metrics"].active_episodes.block_until_ready()

        return {**training_metrics, **metrics}

    # -----------------------------------------------------------------
    # video logging
    # -----------------------------------------------------------------
    @jax.jit
    def _video_rollout(training_state, key):
        env_key, key = jax.random.split(key)
        first_state = eval_env.reset(jax.random.split(env_key, 1))
        waypoint_state = init_waypoint_state(1)

        def f(carry, unused_t):
            env_state, waypoint_state, key = carry
            key, step_key = jax.random.split(key)
            action, (current_wp, current_planning_k) = eval_action(
                training_state, env_state, waypoint_state, step_key
            )
            next_env_state = eval_env.step(env_state, action)
            done_mask = next_env_state.done.astype(jnp.bool_)
            wp_counter = waypoint_state[0]
            new_counter = jnp.where(done_mask, 0, wp_counter + 1)
            new_wp = jnp.where(done_mask[:, None], jnp.zeros_like(current_wp), current_wp)
            new_planning_k = jnp.where(
                done_mask, jnp.zeros_like(current_planning_k), current_planning_k
            )
            render_fields = {
                "qpos": next_env_state.data.qpos,
                "qvel": next_env_state.data.qvel,
                "target_mocap_pos": next_env_state.info["target_mocap_pos"],
                "target_mocap_quat": next_env_state.info["target_mocap_quat"],
            }
            return ((next_env_state, (new_counter, new_wp, new_planning_k), key), render_fields)

        _, render_traj = jax.lax.scan(
            f, (first_state, waypoint_state, key), (), length=eval_episode_length,
        )
        return render_traj

    def get_cslerp_video(training_state, key):
        """Roll out one eval episode and render env frames."""
        render_traj = jax.tree_util.tree_map(np.asarray, _video_rollout(training_state, key))
        video_images = []
        for i in range(eval_episode_length):
            if i % 2 == 0:
                video_images.append(eval_env.render_from_info(
                    render_traj["qpos"][i][0],
                    render_traj["qvel"][i][0],
                    render_traj["target_mocap_pos"][i][0],
                    render_traj["target_mocap_quat"][i][0],
                ))
        return video_images

    def save_gif(frames, path, fps=10):
        from PIL import Image
        imgs = [Image.fromarray(np.asarray(fr).astype(np.uint8)) for fr in frames]
        duration_ms = max(1, int(round(1000 / fps)))
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=duration_ms, loop=0, optimize=False)

    # -----------------------------------------------------------------
    # Prefill + training loop
    # -----------------------------------------------------------------
    waypoint_state = init_waypoint_state(args.num_envs)

    print("prefilling replay buffer....")
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, waypoint_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, waypoint_state, buffer_state, prefill_key
    )

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None

    for ts in range(1, args.num_training_step + 1): #num_timesteps / (num_envs * rollout_length)

        if ts == 1 or ts % max(1, args.num_training_step // 100) == 0:
            print(f"[{ts}/{args.num_training_step}] {100 * ts / args.num_training_step:.1f}%", flush=True)

        key_sgd, key_rollout, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, waypoint_state, buffer_state, data_metrics = data_collect_step(
            training_state, env_state, waypoint_state, buffer_state, key_rollout
        ) #data collection for rollout_length steps
        data_collect_step_time += time.time() - data_collect_start

        learn_step_start = time.time()
        training_state, buffer_state, training_metrics = learn_step(training_state, buffer_state, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if os.environ.get("C_slerp_MEM_PROBE") and ts <= 2:
            jax.block_until_ready((training_state, buffer_state, training_metrics))
            ms = jax.devices()[0].memory_stats()
            print(f"[MEM_PROBE ts={ts}] after learn_step: "
                  f"peak={ms['peak_bytes_in_use']/1e9:.2f}GB "
                  f"in_use={ms['bytes_in_use']/1e9:.2f}GB "
                  f"limit={ms['bytes_limit']/1e9:.2f}GB", flush=True)

        if metrics is None:
            metrics = data_metrics | training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, (data_metrics | training_metrics)
            )

        if args.num_reset_steps > 0 and ts % args.num_training_steps_per_real_reset == 0: #hard reset 
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs) #hard reset the environment
            waypoint_state = init_waypoint_state(args.num_envs)

        if ts % args.num_training_steps_per_eval == 0: #training_step / num_eval_steps
            es = ts // args.num_training_steps_per_eval

            metrics = jax.tree_util.tree_map(lambda x: x / args.num_training_steps_per_eval, metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt
            training_walltime += training_step_time
            sps = (args.num_training_steps_per_eval * args.num_envs * args.rollout_length) / training_step_time

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/data_collection_time_fraction": data_collect_step_time / training_step_time,
                "training/learning_time_fraction": learn_step_time / training_step_time,
                "training/env_steps": training_state.env_steps,
                **{f"training/{name}": value for name, value in metrics.items()},
                "buffer_current_size": replay_buffer.size(buffer_state),
            }

            key, eval_key = jax.random.split(key, 2)
            metrics = run_evaluation(training_state, metrics, eval_key)

            if os.environ.get("C_slerp_MEM_PROBE"):
                ms = jax.devices()[0].memory_stats()
                print(f"[MEM_PROBE eval es={es}] after run_evaluation: "
                      f"peak={ms['peak_bytes_in_use']/1e9:.2f}GB "
                      f"in_use={ms['bytes_in_use']/1e9:.2f}GB "
                      f"limit={ms['bytes_limit']/1e9:.2f}GB", flush=True)

            video_path_file = None
            if args.record_videos and es % max(1, args.video_every_n_evals) == 0:
                try:
                    key, video_key = jax.random.split(key, 2)
                    video_frames = get_cslerp_video(training_state, video_key)
                    if args.save_checkpoint:
                        video_dir = f"{save_path}/videos"
                        os.makedirs(video_dir, exist_ok=True)
                        video_path_file = f"{video_dir}/eval_{es}.gif"
                        save_gif(video_frames, video_path_file, fps=args.video_fps)
                        print(f"Saved eval video to {video_path_file}")
                except Exception as e:
                    print(f"Video recording failed at eval {es}: {e}")
                    video_path_file = None

            print(f"\nEvaluation step {es}:\n")
            pprint.pprint(metrics)
            if args.track:
                wandb_log = dict(metrics)
                if video_path_file is not None and os.path.exists(video_path_file):
                    wandb_log["eval/video"] = wandb.Video(video_path_file, fps=args.video_fps, format="gif")
                wandb.log(wandb_log, step=int(training_state.env_steps))
                if args.wandb_mode == "offline":
                    trigger_sync()

            if args.save_checkpoint:
                # Overwrite a single checkpoint so only the latest params are kept on disk.
                save_params(
                    f"{save_path}/params.pkl",
                    params=(
                        training_state.low_actor_state.params,
                        training_state.critic_state.params,
                        training_state.critic_state.g_encoder_target_params,
                        training_state.alpha_force_state.params,
                        training_state.alpha_high_state.params,
                    ),
                )

                def _to_jsonable(v):
                    if isinstance(v, (jnp.ndarray, np.ndarray)):
                        return v.item() if v.ndim == 0 else v.tolist()
                    if isinstance(v, (np.floating, np.integer, np.bool_)):
                        return v.item()
                    return v

                log_entry = {"eval_step": es, **{k: _to_jsonable(v) for k, v in metrics.items()}}
                with open(f"{save_path}/eval_log.jsonl", "a") as fh:
                    fh.write(json.dumps(log_entry) + "\n")

            metrics = None
            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)



