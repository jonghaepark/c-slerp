"""Continuous tanh-Gaussian actions for the gripper branch.

Modes:
  train / actor_update — reparameterized tanh-Gaussian sample
  eval                 — tanh(mean)
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp


Mode = Literal["train", "eval", "actor_update"]


def sample_action(
    means: jnp.ndarray,
    log_stds: jnp.ndarray,
    key: jax.Array | None,
    *,
    mode: Mode,
):
    """Sample a continuous action in (-1, 1).

    Returns:
      action: tanh-squashed action  [B, A]
      log_prob: sum of tanh-Gaussian log probs over action dims  [B]
      aux: dict with raw pre-tanh sample
    """
    stds = jnp.exp(log_stds)

    if mode == "eval":
        x_ts = means
        action = nn.tanh(means)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)
        return action, log_prob, {"raw": x_ts, "action": action}

    assert key is not None, f"key required for mode={mode}"
    if mode not in ("train", "actor_update"):
        raise ValueError(f"Unknown mode: {mode}")

    x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
    action = nn.tanh(x_ts)
    log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
    log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
    log_prob = log_prob.sum(-1)
    return action, log_prob, {"raw": x_ts, "action": action}
