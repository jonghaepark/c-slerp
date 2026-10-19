"""C_slerp loss functions: low/high critics plus low actor/alpha."""

import flax.linen as nn
import jax
import jax.numpy as jnp


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -0.5 * jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def _cayley_rotation(cayley_params, learnable=True):
    """Pre-compute the Cayley rotation matrix R from parameters."""
    K_raw = cayley_params["params"]["K"]
    if not learnable:
        K_raw = jax.lax.stop_gradient(K_raw)
    K = K_raw - K_raw.T
    dim = K.shape[0]
    I = jnp.eye(dim)
    return jnp.linalg.solve(I + K, I - K)


def update_critics(
    config,
    networks,
    low_transitions,
    high_transitions,
    training_state,
    key,
):
    """Joint critic update where each critic loss uses its own SW batch."""

    def critic_loss(critic_params, low_transitions, high_transitions, key):
        sa_enc_params = critic_params["sa_encoder"]
        shared_enc_params = critic_params["shared_encoder"]
        goal_indices = jnp.array(config["goal_indices"])

        if config["train_shared_in_low"]:
            shared_enc_for_low = shared_enc_params
        else:
            shared_enc_for_low = jax.lax.stop_gradient(shared_enc_params)

        low_state = low_transitions.extras["state"]
        low_action = low_transitions.action
        low_future_state = low_transitions.extras["future_state"][:, goal_indices]

        # Low-level: f_L(s, a, w) — plain dot, no Cayley
        sa_repr = networks["sa_encoder"].apply(
            sa_enc_params, jnp.concatenate([low_state, low_action], axis=-1)
        )
        w_repr_low = networks["shared_encoder"].apply(shared_enc_for_low, low_future_state)

        low_logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], w_repr_low[None, :, :])
        low_logits = low_logits / config["critic_temperature"]
        low_loss = contrastive_loss_fn(config["low_contrastive_loss_fn"], low_logits)

        high_state = high_transitions.extras["state"][:, goal_indices]
        high_future_state = high_transitions.extras["future_state"][:, goal_indices]

        # High-level: f_H(s, w) — Cayley + dot.
        s_repr = networks["shared_encoder"].apply(shared_enc_params, high_state)
        s_repr = jax.lax.stop_gradient(s_repr)

        w_repr_high = networks["shared_encoder"].apply(shared_enc_params, high_future_state)

        cayley_k_raw = critic_params["cayley_v"]["params"]["K"]
        if not config["learn_cayley"]:
            cayley_k_raw = jax.lax.stop_gradient(cayley_k_raw)
        cayley_a = cayley_k_raw - cayley_k_raw.T
        cayley_fro_sq = jnp.sum(jnp.square(cayley_a)) / cayley_a.size
        cayley_reg_loss = config["cayley_fro_reg_coef"] * cayley_fro_sq

        R = _cayley_rotation(critic_params["cayley_v"], config["learn_cayley"])
        s_repr_R = s_repr @ R

        high_logits = energy_fn(config["energy_fn"], s_repr_R[:, None, :], w_repr_high[None, :, :])
        high_logits = high_logits / config["critic_temperature"]
        high_loss = contrastive_loss_fn(config["high_contrastive_loss_fn"], high_logits)

        # Diagnostics
        I = jnp.eye(low_logits.shape[0])
        low_correct = jnp.argmax(low_logits, axis=1) == jnp.argmax(I, axis=1)
        high_correct = jnp.argmax(high_logits, axis=1) == jnp.argmax(I, axis=1)

        high_critic_loss_coef = config.get("high_critic_loss_coef", 4.0)
        total_loss = low_loss + high_critic_loss_coef * high_loss + cayley_reg_loss
        return total_loss, {
            "low_critic_loss": low_loss,
            "high_critic_loss": high_loss,
            "cayley_skew_frobenius_norm": jnp.linalg.norm(cayley_a, ord="fro"),
            "cayley_skew_mean_square": cayley_fro_sq,
            "low_categorical_accuracy": jnp.mean(low_correct),
            "high_categorical_accuracy": jnp.mean(high_correct),
        }

    (loss, aux), grad = jax.value_and_grad(critic_loss, has_aux=True)(
        training_state.critic_state.params,
        low_transitions,
        high_transitions,
        key,
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    goal_indices = jnp.array(config["goal_indices"])
    low_future_state = low_transitions.extras["future_state"][:, goal_indices]
    z = networks["shared_encoder"].apply(
        new_critic_state.params["shared_encoder"], low_future_state
    )
    cos_sim = jnp.matmul(z, z.T)
    B = cos_sim.shape[0]
    mask = 1.0 - jnp.eye(B)
    aux["embed_uniformity"] = jnp.sum(jnp.abs(cos_sim) * mask) / jnp.sum(mask)

    R = _cayley_rotation(new_critic_state.params["cayley_v"])
    aux["cayley_identity_fro_norm"] = jnp.linalg.norm(R - jnp.eye(R.shape[0], dtype=R.dtype), ord="fro")

    return training_state, aux


def update_low_actor_and_alpha(config, networks, transitions, training_state, key):
    """Update low-level actor and its entropy coefficient."""

    def low_actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        key, sample_key = jax.random.split(key)

        state = transitions.extras["state"]
        future_state = transitions.extras["future_state"]
        future_goal = future_state[:, jnp.array(config["goal_indices"])]

        shared_enc_params = jax.lax.stop_gradient(critic_params["shared_encoder"])
        waypoint = networks["shared_encoder"].apply(shared_enc_params, future_goal)

        obs = jnp.concatenate([state, waypoint], axis=-1)
        means, log_stds = networks["low_actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(sample_key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)

        sa_enc_params = jax.lax.stop_gradient(critic_params["sa_encoder"])
        sa_repr = networks["sa_encoder"].apply(
            sa_enc_params, jnp.concatenate([state, action], axis=-1)
        )
        w_repr = networks["shared_encoder"].apply(shared_enc_params, future_goal)
        qf_pi = energy_fn(config["energy_fn"], sa_repr, w_repr)
        if config["actor_also_temperature"]:
            qf_pi = qf_pi / config["critic_temperature"]

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
        return actor_loss, (log_prob, jnp.mean(qf_pi))

    def alpha_loss_fn(alpha_params, log_prob):
        # Optimize log_alpha directly so the temperature can recover after collapsing.
        entropy_gap = jax.lax.stop_gradient(-log_prob - config["target_entropy_low"])
        return jnp.mean(alpha_params["log_alpha"] * entropy_gap)

    (actor_loss_val, (log_prob, low_policy_q)), actor_grad = jax.value_and_grad(low_actor_loss, has_aux=True)(
        training_state.low_actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_low_state.params["log_alpha"],
        transitions,
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
