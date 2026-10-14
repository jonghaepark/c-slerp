"""C_slerp loss functions for builderbench: low/high critics, low actor/alpha"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.cslerp_networks import goal_to_obs_shape  # re-exported for cslerp.py


# ---------------------------------------------------------------------------
# Energy / contrastive helpers
# ---------------------------------------------------------------------------

def energy_fn(name, x, y):
    if name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "l2":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        return -0.5 * jnp.mean(
            2 * jnp.diag(logits)
            - jax.nn.logsumexp(logits, axis=1)
            - jax.nn.logsumexp(logits, axis=0)
        )
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")


def _cayley_rotation(cayley_params, learnable=True):
    """Pre-compute the Cayley rotation matrix R = (I + K)^{-1}(I - K) from params.
    """
    K_raw = cayley_params["params"]["K"]
    if not learnable:
        K_raw = jax.lax.stop_gradient(K_raw)
    K = K_raw - K_raw.T
    dim = K.shape[0]
    I = jnp.eye(dim)
    return jnp.linalg.solve(I + K, I - K)


# ---------------------------------------------------------------------------
# Critic update (low + high, joint)
# ---------------------------------------------------------------------------

def update_critics(config, networks, low_transitions, high_transitions, training_state, key):
    """Joint critic update; each critic loss uses its own SW-relabeled batch."""
    key_smooth, key_diag = jax.random.split(key)
    low_energy = config["low_energy_fn"]
    critic_temperature = config["critic_temperature"]
    full_state = config.get("g_encoder_full_state", False)

    def critic_loss(critic_params, g_encoder_target_params):
        sa_enc_params = critic_params["sa_encoder"]
        shared_enc_params = critic_params["g_encoder"]

        if config["train_shared_in_low"]:
            shared_enc_for_low = shared_enc_params
        else:
            shared_enc_for_low = jax.lax.stop_gradient(shared_enc_params)

        # ---- Low critic: f_L(s, a, w) = energy(phi_sa, psi(w)) ----
        low_state = low_transitions.observation
        low_action = low_transitions.action
        low_future_goal = (low_transitions.extras["future_obs"] if full_state
                           else low_transitions.extras["future_goal"])

        sa_repr = networks["sa_encoder"].apply(
            sa_enc_params, jnp.concatenate([low_state, low_action], axis=-1)
        )
        w_repr_low = networks["g_encoder"].apply(shared_enc_for_low, low_future_goal)

        if low_energy == "dot":
            low_logits = energy_fn("dot", sa_repr[:, None, :], w_repr_low[None, :, :])
            low_logits = low_logits / critic_temperature
        else:  # "l2"
            low_logits = energy_fn("l2", sa_repr[:, None, :], w_repr_low[None, :, :])
        low_loss = contrastive_loss_fn(config["low_contrastive_loss_fn"], low_logits)

        # ---- High critic: f_H(s, w) = <psi(s) R, psi(w)> ----
        high_ach = (high_transitions.observation if full_state
                    else high_transitions.achieved_goal)
        high_future_goal = (high_transitions.extras["future_obs"] if full_state
                            else high_transitions.extras["future_goal"])

        s_repr = networks["g_encoder"].apply(shared_enc_params, high_ach)
        w_repr_high = jax.lax.stop_gradient(networks["g_encoder"].apply(shared_enc_params, high_future_goal))

        learn_cayley = config.get("learn_cayley", True)
        cayley_k_raw = critic_params["cayley_v"]["params"]["K"]
        if not learn_cayley:
            cayley_k_raw = jax.lax.stop_gradient(cayley_k_raw)
        cayley_a = cayley_k_raw - cayley_k_raw.T
        cayley_fro_sq = jnp.sum(jnp.square(cayley_a)) / cayley_a.size
        cayley_reg_loss = config["cayley_fro_reg_coef"] * cayley_fro_sq

        R = _cayley_rotation(critic_params["cayley_v"], learn_cayley)
        s_repr_R = s_repr @ R

        high_logits = energy_fn("dot", s_repr_R[:, None, :], w_repr_high[None, :, :])
        high_logits = high_logits / critic_temperature
        high_loss = contrastive_loss_fn(config["high_contrastive_loss_fn"], high_logits)

        smooth_reg_loss = jnp.asarray(0.0, dtype=s_repr.dtype)

        I = jnp.eye(low_logits.shape[0])
        low_correct = jnp.argmax(low_logits, axis=1) == jnp.argmax(I, axis=1)
        high_correct = jnp.argmax(high_logits, axis=1) == jnp.argmax(I, axis=1)

        total_loss = (
            low_loss + config["high_critic_loss_coef"] * high_loss
            + cayley_reg_loss
        )
        return total_loss, {
            "low_critic_loss": low_loss,
            "high_critic_loss": high_loss,
            "cayley_skew_frobenius_norm": jnp.linalg.norm(cayley_a, ord="fro"),
            "low_categorical_accuracy": jnp.mean(low_correct),
            "high_categorical_accuracy": jnp.mean(high_correct),
        }
    (loss, aux), grad = jax.value_and_grad(critic_loss, has_aux=True)(
        training_state.critic_state.params,
        training_state.critic_state.g_encoder_target_params,
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    low_future_goal = (low_transitions.extras["future_obs"] if full_state
                       else low_transitions.extras["future_goal"])
    z = networks["g_encoder"].apply(
        new_critic_state.params["g_encoder"], low_future_goal
    )
    cos_sim = jnp.matmul(z, z.T)
    B = cos_sim.shape[0]
    mask = 1.0 - jnp.eye(B)
    aux["embed_uniformity"] = jnp.sum(jnp.abs(cos_sim) * mask) / jnp.sum(mask)

    R = _cayley_rotation(new_critic_state.params["cayley_v"])
    aux["cayley_identity_fro_norm"] = jnp.linalg.norm(
        R - jnp.eye(R.shape[0], dtype=R.dtype), ord="fro"
    )

    return training_state, aux


# ---------------------------------------------------------------------------
# Low actor + alpha_low update
# ---------------------------------------------------------------------------

def update_low_actor_and_alpha(config, networks, transitions, training_state, key):
    low_energy = config["low_energy_fn"]
    full_state = config.get("g_encoder_full_state", False)

    def low_actor_loss(actor_params, critic_params, log_alpha, key):
        key, sample_key = jax.random.split(key)

        state = transitions.observation
        future_goal = (transitions.extras["future_obs"] if full_state
                       else transitions.extras["future_goal"])

        shared_enc_params = jax.lax.stop_gradient(critic_params["g_encoder"])
        waypoint = networks["g_encoder"].apply(shared_enc_params, future_goal)

        obs = jnp.concatenate([state, waypoint], axis=-1)
        means, log_stds = networks["low_actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(sample_key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)

        sa_enc_params = jax.lax.stop_gradient(critic_params["sa_encoder"])
        sa_repr = networks["sa_encoder"].apply(
            sa_enc_params, jnp.concatenate([state, action], axis=-1)
        )
        w_repr = networks["g_encoder"].apply(shared_enc_params, future_goal)

        qf_pi = energy_fn(low_energy, sa_repr, w_repr)
        if low_energy == "dot" and config["actor_also_temperature"]:
            qf_pi = qf_pi / config["critic_temperature"]

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return actor_loss, (log_prob, jnp.mean(qf_pi))

    def alpha_loss_fn(alpha_params, log_prob):
        entropy_gap = jax.lax.stop_gradient(-log_prob - config["target_entropy_low"])
        return jnp.mean(alpha_params["log_alpha"] * entropy_gap)

    (actor_loss_val, (log_prob, low_policy_q)), actor_grad = jax.value_and_grad(
        low_actor_loss, has_aux=True
    )(
        training_state.low_actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_low_state.params["log_alpha"],
        key,
    )
    new_low_actor_state = training_state.low_actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
        training_state.alpha_low_state.params, log_prob
    )
    new_alpha_low_state = training_state.alpha_low_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(
        low_actor_state=new_low_actor_state,
        alpha_low_state=new_alpha_low_state,
    )
    log_alpha_low = training_state.alpha_low_state.params["log_alpha"]
    metrics = {
        "low_actor_loss": actor_loss_val,
        "alpha_low_loss": alpha_loss_val,
        "low_entropy": -log_prob.mean(),
        "log_alpha_low": log_alpha_low,
        "alpha_low": jnp.exp(log_alpha_low),
        "low_policy_q": low_policy_q,
    }
    return training_state, metrics



def update_k_predictor(config, networks, batch, training_state, key):
    del key
    critic_params = training_state.critic_state.params
    g_encoder = networks["g_encoder"]
    k_predictor = networks["k_predictor"]

    cur_goal = batch.achieved_goal
    sg_goal = batch.extras["sg_goal"]
    obs_size = batch.observation.shape[-1]

    if config["k_predictor_with_psi"]:
        if config.get("g_encoder_full_state", False):
            state_k = g_encoder.apply(critic_params["g_encoder"], batch.observation)
            goal_k = g_encoder.apply(critic_params["g_encoder"], goal_to_obs_shape(sg_goal, obs_size))
        else:
            state_k = g_encoder.apply(critic_params["g_encoder"], cur_goal)
            goal_k = g_encoder.apply(critic_params["g_encoder"], sg_goal)
    else:
        state_k = batch.observation if config.get("k_predictor_full_state", False) else cur_goal
        goal_k = (goal_to_obs_shape(sg_goal, obs_size)
                  if config.get("k_predictor_goal_full_state", False) else sg_goal)
    state_k = jax.lax.stop_gradient(state_k)
    goal_k = jax.lax.stop_gradient(goal_k)

    x_k = jnp.concatenate([state_k, goal_k], axis=-1)
    target = (batch.extras["k_target"] - 1).astype(jnp.int32)
    mask = batch.extras["k_valid_mask"].astype(jnp.float32)
    mask_denom = jnp.maximum(jnp.sum(mask), 1.0)

    def loss_fn(k_predictor_params):
        logits = k_predictor.apply(k_predictor_params, x_k)
        per_example_loss = optax_softmax_ce(logits, target)
        k_loss = jnp.sum(per_example_loss * mask) / mask_denom
        total_loss = config["k_predictor_loss_coef"] * k_loss
        return total_loss, (k_loss, logits)

    (total_loss, (k_loss, logits)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        training_state.k_predictor_state.params
    )
    new_k_predictor_state = training_state.k_predictor_state.apply_gradients(grads=grad)
    training_state = training_state.replace(k_predictor_state=new_k_predictor_state)

    pred = jnp.argmax(logits, axis=-1)
    correct = (pred == target).astype(jnp.float32)
    pred_k_f = (pred + 1).astype(jnp.float32)
    target_k_f = (target + 1).astype(jnp.float32)
    return training_state, {
        "k_predictor_loss": k_loss,
        "k_predictor_total_loss": total_loss,
        "k_predictor_accuracy": jnp.sum(correct * mask) / mask_denom,
        "k_target_mean": jnp.sum(target_k_f * mask) / mask_denom,
        "k_pred_mean": jnp.sum(pred_k_f * mask) / mask_denom,
        "k_abs_error": jnp.sum(jnp.abs(pred_k_f - target_k_f) * mask) / mask_denom,
        "k_valid_fraction": jnp.mean(mask),
    }


def optax_softmax_ce(logits, integer_labels):
    """Softmax cross-entropy with integer labels (avoids an optax import here)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    labels_onehot = jax.nn.one_hot(integer_labels, logits.shape[-1], dtype=logits.dtype)
    return -jnp.sum(labels_onehot * log_probs, axis=-1)
