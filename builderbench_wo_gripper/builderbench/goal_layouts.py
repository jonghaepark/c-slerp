"""Procedural goal layouts with configurable lateral spacing."""

from __future__ import annotations

import jax.numpy as jnp


def build_goal_layout(kind: str, s, z_step: float = 0.04):
    """Build cube goal positions (num_cubes, 3) for a named layout.

    Args:
        kind: one of ``towers_3x3``, ``towers_2x5``, ``towers_2x4``,
            ``pyramid_4321``, ``pyramid_3221``.
        s: lateral spacing (adjacent tower centers for towers; pyramid unit step).
        z_step: vertical story height (default 0.04).

    Index order is layer-major, left-to-right within each layer. Template x = 0.
    """
    s = jnp.asarray(s, dtype=jnp.float32)
    z = jnp.asarray(z_step, dtype=jnp.float32)

    if kind == "towers_3x3":
        # 3 towers x 3 stories; centers at (-s, 0, +s)
        ys = jnp.array([-s, 0.0, s], dtype=jnp.float32)
        zs = jnp.array([0.0, z, 2.0 * z], dtype=jnp.float32)
        goals = []
        for zi in range(3):
            for yi in range(3):
                goals.append(jnp.array([0.0, ys[yi], zs[zi]], dtype=jnp.float32))
        return jnp.stack(goals, axis=0)

    if kind == "towers_2x5":
        # 2 towers x 5 stories; centers at (+-s/2) so adjacent distance is s
        ys = jnp.array([-0.5 * s, 0.5 * s], dtype=jnp.float32)
        zs = jnp.arange(5, dtype=jnp.float32) * z
        goals = []
        for zi in range(5):
            for yi in range(2):
                goals.append(jnp.array([0.0, ys[yi], zs[zi]], dtype=jnp.float32))
        return jnp.stack(goals, axis=0)

    if kind == "towers_2x4":
        # 2 towers x 4 stories; centers at (+-s/2) so adjacent distance is s
        ys = jnp.array([-0.5 * s, 0.5 * s], dtype=jnp.float32)
        zs = jnp.arange(4, dtype=jnp.float32) * z
        goals = []
        for zi in range(4):
            for yi in range(2):
                goals.append(jnp.array([0.0, ys[yi], zs[zi]], dtype=jnp.float32))
        return jnp.stack(goals, axis=0)

    if kind == "pyramid_4321":
        # 4-3-2-1 pyramid; lateral unit = s
        layers = [
            (0.0, jnp.array([-1.5, -0.5, 0.5, 1.5], dtype=jnp.float32) * s),
            (z, jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32) * s),
            (2.0 * z, jnp.array([-0.5, 0.5], dtype=jnp.float32) * s),
            (3.0 * z, jnp.array([0.0], dtype=jnp.float32)),
        ]
        goals = []
        for z_val, ys in layers:
            for y in ys:
                goals.append(jnp.array([0.0, y, z_val], dtype=jnp.float32))
        return jnp.stack(goals, axis=0)

    if kind == "pyramid_3221":
        # 3-2-2-1 pyramid from bottom; lateral unit = s
        layers = [
            (0.0, jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32) * s),
            (z, jnp.array([-0.5, 0.5], dtype=jnp.float32) * s),
            (2.0 * z, jnp.array([-0.5, 0.5], dtype=jnp.float32) * s),
            (3.0 * z, jnp.array([0.0], dtype=jnp.float32)),
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

    if kind == "towers_3x3":
        goals = []
        for zi, z in enumerate([0.0, z_step, 2 * z_step]):
            for y in (-s, 0.0, s):
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    if kind == "towers_2x5":
        goals = []
        for zi in range(5):
            z = zi * z_step
            for y in (-0.5 * s, 0.5 * s):
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    if kind == "towers_2x4":
        goals = []
        for zi in range(4):
            z = zi * z_step
            for y in (-0.5 * s, 0.5 * s):
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    if kind == "pyramid_4321":
        layers = [
            (0.0, [-1.5 * s, -0.5 * s, 0.5 * s, 1.5 * s]),
            (z_step, [-s, 0.0, s]),
            (2 * z_step, [-0.5 * s, 0.5 * s]),
            (3 * z_step, [0.0]),
        ]
        goals = []
        for z, ys in layers:
            for y in ys:
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    if kind == "pyramid_3221":
        layers = [
            (0.0, [-s, 0.0, s]),
            (z_step, [-0.5 * s, 0.5 * s]),
            (2 * z_step, [-0.5 * s, 0.5 * s]),
            (3 * z_step, [0.0]),
        ]
        goals = []
        for z, ys in layers:
            for y in ys:
                goals.append([0.0, y, z])
        return np.asarray(goals, dtype=np.float32)

    raise ValueError(f"Unknown goal layout kind: {kind!r}")
