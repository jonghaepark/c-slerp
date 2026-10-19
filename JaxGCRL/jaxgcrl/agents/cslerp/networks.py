"""C_slerp network modules."""

import math

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class NormalizedEncoder(nn.Module):
    """Encoder with optional L2-normalization on output."""

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 9
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = True
    normalize: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        if self.normalize:
            # x = nn.LayerNorm()(x)
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


class LowActor(nn.Module):
    """Low-level actor: outputs (mean, log_std) for tanh-squashed actions."""

    action_size: int
    network_width: int = 256
    network_depth: int = 9
    skip_connections: int = 4
    use_relu: bool = False
    use_ln: bool = True
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -6.0

    @nn.compact
    def __call__(self, x):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
