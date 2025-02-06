from flax import linen as nn

class ExpansionLinear(nn.Module):
    hidden_dim: int
    expansion: int

    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, training: bool = True):
        x = nn.Dense(self.expansion * self.hidden_dim)(x) # changes the number of output features
        x = nn.Dropout(self.dropout, deterministic=not training)(x) # creates dropout layer with rate
            
        x = nn.gelu(x) # gaussian error linear unit/ type of activation function/weights units by their percentiles
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        
        return x