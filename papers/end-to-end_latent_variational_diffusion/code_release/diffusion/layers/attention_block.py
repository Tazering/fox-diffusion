from jax import Array, nn

import haiku as hk


class AttentionBlock(hk.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        self.attention = hk.MultiHeadAttention(
            num_heads, 
            hidden_dim // num_heads, 
            w_init=hk.initializers.VarianceScaling()
        )
        
        self.skip_connection = hk.BatchApply(hk.GRU(hidden_dim))
        
    def __call__(
        self, 
        embeddings: Array,
        mask: Array,
        *,
        is_training: bool = True,
    ) -> Array:
        """
        Parameters
        ----------
        embeddings: (B, T, D)
        mask: (B, T)
        is_training: bool, keyword-only

        Returns
        -------
        embeddings: (B, T, D)
        """  
        
        hidden = self.norm(embeddings)
        hidden = self.attention(hidden, hidden, hidden, mask=mask[:, None, None, :])
        
        hidden, _ = self.skip_connection(hidden, embeddings)
        return hidden