"""Procedural goal layouts with configurable lateral spacing."""

from __future__ import annotations

import jax.numpy as jnp


def build_goal_layout(kind: str, s, z_step: float = 0.04):
    """Build cube goal positions (num_cubes, 3) for a named layout.

    Args:
        kind: one of ``pyramid_211``.
        s: lateral spacing (adjacent centers for 2-wide base).
        z_step: vertical story height (default 0.04).

    Index order is layer-major, left-to-right within each layer. Template x = 0.
    """
    s = jnp.asarray(s, dtype=jnp.float32)
    z = jnp.asarray(z_step, dtype=jnp.float32)

    if kind == "pyramid_211":
        # 2-1-1 pyramid; base centers at +-s/2 so adjacent distance is s
        layers = [
            (0.0, jnp.array([-0.5, 0.5], dtype=jnp.float32) * s),
            (z, jnp.array([0.0], dtype=jnp.float32)),
            (2.0 * z, jnp.array([0.0], dtype=jnp.float32)),
        ]
        goals = []
        for z_val, ys in layers:
            for y in ys:
                goals.append(jnp.array([0.0, y, z_val], dtype=jnp.float32))
        return jnp.stack(goals, axis=0)

    raise ValueError(f"Unknown goal layout kind: {kind!r}")


def nominal_goal_layout_numpy(kind: str, s: float = 0.052, z_step: float = 0.04):
    """NumPy copy of ``build_goal_layout`` for task-npz generation."""
    import numpy as np

    if kind == "pyramid_211":
        layers = [
            (0.0, [-0.5 * s, 0.5 * s]),
            (z_step, [0.0]),
            (2 * z_step, [0.0]),
        ]
        goals = []
        for z, ys in layers:
            for y in ys:
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    raise ValueError(f"Unknown goal layout kind: {kind!r}")
