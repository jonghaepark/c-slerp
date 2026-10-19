"""C_slerp network modules."""

import math

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


def goal_to_obs_shape(goal, obs_size):
    """Embed a goal-space vector (per-cube xyz, shape [..., 3*nc]) into the full
    observation layout so a goal can be fed to a full-state encoder / predictor.

    Observation layout is [obj_pos(3nc), obj_quat(4nc), obj_linvel(3nc),
    obj_angvel(3nc), ...]. We fill obj_pos with the goal positions, set each cube's
    quaternion to the upright (1,0,0,0) (matching the target cubes), and zero
    everything afterwards (velocities / ctrl / select_action).
    """
    nc = goal.shape[-1] // 3
    batch = goal.shape[:-1]
    quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=goal.dtype), nc)
    quat = jnp.broadcast_to(quat, batch + (4 * nc,))
    rest = obs_size - goal.shape[-1] - 4 * nc
    zeros = jnp.zeros(batch + (rest,), dtype=goal.dtype)
    return jnp.concatenate([goal, quat, zeros], axis=-1)


class ResidualBlock(nn.Module):
    """4x (Dense -> norm -> activation) wrapped in an identity shortcut."""

    hidden_dim: int
    use_ln: bool = True
    use_relu: bool = False
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        residual = x
        for _ in range(4):
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = activation(x)
        return residual + self.residual_scale * x


class NormalizedEncoder(nn.Module):
    """Encoder with optional L2-normalization on output (residual trunk)."""

    repr_dim: int = 64
    network_width: int = 256       # residual hidden dim
    network_depth: int = 9         # num residual blocks = max(1, network_depth // 4)
    skip_connections: int = 4      
    use_relu: bool = False         # swish when False
    use_ln: bool = True
    normalize: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray):
    
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        hidden_dim = self.network_width
        num_blocks = max(1, self.network_depth // 4)
        branch_scale = num_blocks ** -0.5 if num_blocks > 1 else 1.0

        x = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(data)
        if self.use_ln:
            x = nn.LayerNorm()(x)
        x = activation(x)

        for _ in range(num_blocks):
            x = ResidualBlock(
                hidden_dim=hidden_dim,
                use_ln=self.use_ln,
                use_relu=self.use_relu,
                residual_scale=branch_scale,
            )(x)

        x = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        if self.normalize:
            x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-6)
        return x



class CayleyTransform(nn.Module):
    """Orthogonal matrix via Cayley transform: R = (I + K)^{-1}(I - K)."""

    dim: int
    init_std: float = 0.0

    @nn.compact
    def __call__(self, x):
        if self.init_std > 0:
            K_init = nn.initializers.normal(stddev=self.init_std / math.sqrt(2.0))
        else:
            K_init = nn.initializers.zeros
        K_raw = self.param("K", K_init, (self.dim, self.dim))
        K = K_raw - K_raw.T
        I = jnp.eye(self.dim)
        R = jnp.linalg.solve(I + K, I - K)
        return x @ R


class KPredictor(nn.Module):

    num_classes: int
    network_width: int = 256       # residual hidden dim
    network_depth: int = 4         # num residual blocks = max(1, network_depth // 4)
    skip_connections: int = 4
    use_relu: bool = False         # swish when False
    use_ln: bool = True
    use_residual_trunk: bool = True # for compatibility with the original C_slerp

    @nn.compact
    def __call__(self, x):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        if not self.use_residual_trunk:
            for _ in range(self.network_depth):
                x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
                if self.use_ln:
                    x = nn.LayerNorm()(x)
                x = activation(x)
            return nn.Dense(self.num_classes, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        hidden_dim = self.network_width
        num_blocks = max(1, self.network_depth // 4)
        branch_scale = num_blocks ** -0.5 if num_blocks > 1 else 1.0

        x = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        if self.use_ln:
            x = nn.LayerNorm()(x)
        x = activation(x)

        for _ in range(num_blocks):
            x = ResidualBlock(
                hidden_dim=hidden_dim,
                use_ln=self.use_ln,
                use_relu=self.use_relu,
                residual_scale=branch_scale,
            )(x)

        return nn.Dense(self.num_classes, kernel_init=lecun_uniform, bias_init=bias_init)(x)


class LowActor(nn.Module):
    """Low-level actor (residual trunk): outputs (mean, log_std).

    Takes a single concatenated input ``[state, latent_waypoint]``.
    """

    action_size: int
    network_width: int = 256       # residual hidden dim
    network_depth: int = 9         # num residual blocks = max(1, network_depth // 4)
    skip_connections: int = 4      
    use_relu: bool = False         # swish when False
    use_ln: bool = True
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -6.0

    @nn.compact
    def __call__(self, x):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        hidden_dim = self.network_width
        num_blocks = max(1, self.network_depth // 4)
        branch_scale = num_blocks ** -0.5 if num_blocks > 1 else 1.0

        # ---- residual trunk (crl-block.py style) ----
        x = nn.Dense(hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        if self.use_ln:
            x = nn.LayerNorm()(x)
        x = activation(x)

        for _ in range(num_blocks):
            x = ResidualBlock(
                hidden_dim=hidden_dim,
                use_ln=self.use_ln,
                use_relu=self.use_relu,
                residual_scale=branch_scale,
            )(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

