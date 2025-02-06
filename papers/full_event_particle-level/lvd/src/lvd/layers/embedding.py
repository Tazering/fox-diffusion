from jax.numpy import ndarray as Array

from flax import linen as nn
from lvd.layers.linear_block import LinearBlock

# Dense[self.hidden_dim] -> Dropout[rate] -> self.num_layers * (LinearBlock) -> Dense
class Embedding(nn.Module):
    hidden_dim: int
    num_layers: int
    expansion: int = 2

    dropout: float = 0.0
    skip_connection_type: str = "gru"

    @nn.compact
    def __call__(self, x: Array, *, training: bool = False) -> Array:
        y = nn.Dense(self.hidden_dim)(x) # creates a dense layer
        y = nn.Dropout(self.dropout, deterministic=not training)(y) # creates a dropout layer

        for _ in range(self.num_layers):
            y = LinearBlock(
                self.hidden_dim, # int
                self.expansion, # int
                self.dropout, # float
                self.skip_connection_type # SkipConnectionClass
            )(
                y, 
                training = training
            )

        y = nn.Dense(self.hidden_dim)(y) # final dense layer

        return y
