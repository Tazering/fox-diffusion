from dataclasses import dataclass
from typing import Tuple, Optional

import jax
from jax import numpy as jnp
from jax import Array

import haiku as hk

from diffusion.layers.linear_block import LinearBlock


@dataclass
class PartonEncoder(hk.Module):
    hidden_dim: int
    num_layers: int
    normalize: bool
    normalize_scale: bool
    trivial_vae:  bool
    conditional_vae: bool
    
    def __call__(
        self, 
        parton_features: Array,  # [B, D]
        detector_embeddings: Optional[Array] = None,
        *,
        is_training: bool = True,
    ) -> Tuple[Array, Array]:  
        # Shortcut for no VAE
        if self.trivial_vae:
            M = jnp.eye(self.hidden_dim, parton_features.shape[-1])
            mean = jnp.einsum("ij,bj->bi", M, parton_features)
            log_std = jnp.ones_like(mean) * 1e-6

            return mean, log_std
        
        if self.conditional_vae and (detector_embeddings is not None):
            parton_features = jnp.concatenate((parton_features, detector_embeddings), axis=-1)

        embeddings = hk.Linear(2 * self.hidden_dim)(parton_features)
        
        for _ in range(self.num_layers):
            embeddings = LinearBlock(2 * self.hidden_dim)(
                embeddings, 
                is_training=is_training
            )

        mean = hk.Linear(self.hidden_dim)(embeddings)
        log_std = hk.Linear(self.hidden_dim)(embeddings)

        if self.normalize:
            mean = mean / jnp.linalg.norm(mean, axis=1, keepdims=True)

            if self.normalize_scale:
                mean = mean * jnp.sqrt(self.hidden_dim)
            
        return mean, log_std
        