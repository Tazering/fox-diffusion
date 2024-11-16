from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp

import haiku as hk

from diffusion.layers.linear_block import LinearBlock
from diffusion.layers.timestep_embedding import TimestepEmbedding


@dataclass
class DenoisingNetwork(hk.Module):
    hidden_dim: int
    timestep_dim: int
    num_layers: int

    def __call__(
        self, 
        z: Array,  # [B, D]
        conditioning: Array, # [B, D]
        alpha_squared: Array, # [B,1]
        *,
        is_training: bool = True,
    ) -> Array:
        timestep_embedding = TimestepEmbedding(self.timestep_dim)

        hidden = jnp.concatenate(axis=1, arrays=(
            z,
            conditioning,
            timestep_embedding(alpha_squared)
        ))

        hidden = hk.Linear(hidden.shape[-1])(hidden)

        for _ in range(self.num_layers):
            hidden = LinearBlock(hidden.shape[-1])(
                hidden, 
                is_training=is_training
            )

        return hk.Linear(self.hidden_dim)(hidden)
        