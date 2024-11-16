from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax import Array

import haiku as hk


@dataclass
class TimestepEmbedding(hk.Module):
  embedding_dim: int

  def __call__(self, timesteps: Array) -> Array:
    """
    Parameters
    ----------
    t: (B, 1) or (1,)

    Returns
    -------
    (B, D) or (D,)
    """    
    # Handle the case where we have many different scehdulers
    if timesteps.shape[-1] > 1:
      return 2 * timesteps - 1
    
    # Scale from [0, 1] to [0, 1_000]
    # timesteps = 1_0000.0 * timesteps[:, None] 
    timesteps = 1_0000.0 * timesteps

    cosine_dim = self.embedding_dim // 2
    cosine_time = jnp.arange(cosine_dim)

    embedding = jnp.log(10_000) / (cosine_dim - 1)
    embedding = jnp.exp(-cosine_time * embedding)
    embedding = timesteps * embedding[None, :]
    embedding = jnp.concatenate((jnp.sin(embedding), jnp.cos(embedding)), axis=1)

    return embedding