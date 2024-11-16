from jax import Array, nn

import haiku as hk


class LinearBlock(hk.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        expansion: int = 2, 
        skip_connection_type="gru"
    ):
        super().__init__()
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        self.l0 = hk.Linear(hidden_dim, with_bias=False)
        self.l1 = hk.Linear(expansion * hidden_dim)
        self.l2 = hk.Linear(hidden_dim)
        
        self.skip_connection = hk.GRU(hidden_dim)
                
    def __call__(
          self,
          embeddings: Array,
          *,
          is_training: bool = True,
      ) -> Array:
        """
        Parameters
        ----------
        embeddings: (B, D)
        is_training: bool, keyword-only

        Returns
        -------
        embeddings: (B, D)
        """
        
        hidden = self.norm(self.l0(embeddings))
        hidden = self.l1(hidden)
        hidden = nn.gelu(hidden)
        hidden = self.l2(hidden)
        hidden, _ = self.skip_connection(hidden, embeddings)
        return hidden
    