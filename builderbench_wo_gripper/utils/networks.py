import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Sequence
import pickle
from etils import epath
import functools

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.relu
    kernel_init: Any = nn.initializers.lecun_uniform()
    bias_init: Any = nn.initializers.zeros
    final_kernet_init: Any = nn.initializers.lecun_uniform()
    final_bias_init: Any = nn.initializers.zeros
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @nn.compact
    def get_penultimate(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.bias,
            )(hidden)
            hidden = self.activation(hidden)
            if self.layer_norm:
                hidden = nn.LayerNorm()(hidden)
        
        return hidden

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        hidden = self.get_penultimate(data)
        
        final_layer_index = len(self.layer_sizes) - 1
        hidden = nn.Dense(
                self.layer_sizes[-1],
                name=f'hidden_{final_layer_index}',
                kernel_init=self.final_kernet_init,
                bias_init=self.final_bias_init,
                use_bias=self.bias,
            )(hidden)
        if self.activate_final:
            hidden = self.activation(hidden)
            if self.layer_norm:
                hidden = nn.LayerNorm()(hidden)

        return hidden
    
def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)