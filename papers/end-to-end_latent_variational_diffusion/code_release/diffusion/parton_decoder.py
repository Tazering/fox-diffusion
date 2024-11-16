from dataclasses import dataclass
from typing import Optional

import jax
from jax import Array
from jax import numpy as jnp
import haiku as hk

from diffusion.layers.linear_block import LinearBlock


@dataclass
class PartonDecoder(hk.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int

    trivial_vae: bool
    conditional_vae: bool
    
    def __call__(
        self, 
        embeddings: Array,  # [B, D]
        detector_embeddings: Optional[Array] = None,
        *,
        is_training: bool = True,
    ) -> jax.Array:
        hidden_dim = 2 * self.hidden_dim if self.conditional_vae else self.hidden_dim

        # Shortcut for no VAE
        if self.trivial_vae:
            return embeddings[:, :self.output_dim]
        
        if self.conditional_vae and (detector_embeddings is not None):
            embeddings = jnp.concatenate((embeddings, detector_embeddings), axis=-1)
            
        embeddings = hk.Linear(hidden_dim)(embeddings)

        for _ in range(self.num_layers):
            embeddings = LinearBlock(hidden_dim)(
                embeddings, 
                is_training=is_training
            )

        return hk.Linear(self.output_dim)(embeddings)
        