from dataclasses import dataclass
from jax import Array, nn

import haiku as hk

from diffusion.layers.linear_block import LinearBlock
from diffusion.layers.attention_block import AttentionBlock


@dataclass
class TransformerBlock(hk.Module):
    hidden_dim: int
    num_heads: int
    expansion: int = 2
    
    def __call__(
        self, 
        embeddings: Array,  # [B, T, D]
        mask: Array,  # [B, T]
        *,
        is_training: bool = True,
    ) -> Array:
        attention = AttentionBlock(self.hidden_dim, self.num_heads)
        linear = hk.BatchApply(LinearBlock(self.hidden_dim, self.expansion))

        return linear(attention(embeddings, mask))