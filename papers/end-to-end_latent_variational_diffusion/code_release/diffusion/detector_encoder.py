from dataclasses import dataclass

import jax
from jax import numpy as jnp

import haiku as hk

from diffusion.layers.transformer_block import TransformerBlock


@dataclass
class DetectorEncoder(hk.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    
    def __call__(
        self, 
        detector_features: jax.Array,  # [B, T, D]
        detector_mask: jax.Array,  # [B, T],
        met_features: jax.Array, # [B, M]
        *,
        is_training: bool = True,
    ) -> jax.Array:       
        detector_embeddings = hk.Linear(self.hidden_dim)(detector_features)
        
        met_embeddings = hk.Linear(self.hidden_dim)(met_features)
        met_embeddings = met_embeddings[:, None, :]
        
        event_embeddings = hk.get_parameter(
            "event_embedding",
            (1, 1, self.hidden_dim),
            init=hk.initializers.RandomNormal()
        )
        
        mask = jnp.pad(detector_mask, ((0, 0), (2, 0)), constant_values=True)
        
        embeddings = jnp.concatenate(axis=1, arrays=(
            jnp.broadcast_to(event_embeddings, (detector_embeddings.shape[0], 1, self.hidden_dim)),
            jnp.broadcast_to(met_embeddings, (detector_embeddings.shape[0], 1, self.hidden_dim)),
            detector_embeddings            
        ))
        
        for _ in range(self.num_layers):
            embeddings = TransformerBlock(self.hidden_dim, self.num_heads)(
                embeddings, 
                mask, 
                is_training=is_training
            )

        return hk.Linear(self.hidden_dim)(embeddings[:, 0])
        