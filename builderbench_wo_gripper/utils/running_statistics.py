from typing import Tuple

from flax import struct
import jax
import jax.numpy as jnp

# Adapted from https://github.com/google/brax/blob/main/brax/training/acme/running_statistics.py

@struct.dataclass
class RunningStatisticsState:
	"""Full state of running statistics computation."""
	mean: jnp.ndarray
	std: jnp.ndarray
	count: jnp.ndarray
	summed_variance: jnp.ndarray


def init_state(shape: Tuple[int, ...]) -> RunningStatisticsState:
	"""Initializes the running statistics for the given shape."""
	dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

	return RunningStatisticsState(
		count = jnp.zeros((), dtype=dtype),
		mean = jnp.zeros(shape, dtype=dtype),
		summed_variance = jnp.zeros(shape, dtype=dtype),
		std = jnp.ones(shape, dtype=dtype)
		)


def _validate_batch_shapes(batch: jnp.ndarray,
                           reference_sample: jnp.ndarray,
                           batch_dims: Tuple[int, ...]) -> None:
	"""Verifies shapes of batch against the reference sample.

	Checks that batch dimensions are the same in all leaves in the batch.
	Checks that non-batch dimensions for all leaves in the batch are the same
	as in the reference sample.
	"""
	
	expected_shape = batch_dims + reference_sample.shape
	assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'


def update(state: RunningStatisticsState,
           batch: jnp.ndarray,
           std_min_value: float = 1e-6,
           std_max_value: float = 1e6,
           validate_shapes: bool = True) -> RunningStatisticsState:
	"""Updates the running statistics with the given batch of data.

	Note: by default will use int32 for counts and float32 for accumulated
	variance. This results in an integer overflow after 2^31 data points and
	degrading precision after 2^24 batch updates or even earlier if variance
	updates have large dynamic range.
	To improve precision, consider setting jax_enable_x64 to True, see
	https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
	"""

	batch_shape = batch.shape
	# We assume the batch dimensions always go first.
	batch_dims = batch_shape[:len(batch_shape) - state.mean.ndim]

	batch_axis = range(len(batch_dims))
	step_increment = jnp.prod(jnp.array(batch_dims))
	new_count = state.count + step_increment


	if validate_shapes:
		_validate_batch_shapes(batch, state.mean, batch_dims)

	diff_to_old_mean = batch - state.mean
	mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / new_count
	new_mean = state.mean + mean_update

	diff_to_new_mean = batch - new_mean
	variance_update = diff_to_old_mean * diff_to_new_mean
	variance_update = jnp.sum(variance_update, axis=batch_axis)
	new_summed_variance = state.summed_variance + variance_update	

	new_std = jnp.sqrt( jnp.maximum(new_summed_variance, 0) / new_count)
	new_std = jnp.clip(new_std, std_min_value, std_max_value)

	return RunningStatisticsState(
		count=new_count, mean=new_mean, summed_variance=new_summed_variance, std=new_std)


def normalize(batch: jnp.ndarray, state: RunningStatisticsState) -> jnp.ndarray:
    """Normalizes data using running statistics."""
    return (batch - state.mean) / (state.std)


def denormalize(batch: jnp.ndarray, state: RunningStatisticsState) -> jnp.ndarray:
    """Denormalizes data using running statistics."""
    return batch * state.std + state.mean