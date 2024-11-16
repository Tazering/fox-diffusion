from typing import Optional
from dataclasses import dataclass

from jax import Array, nn

import haiku as hk

@dataclass
class Simple(hk.Module):
     hidden_dim: int

     def __call__(self, inputs, state):
          return inputs + state
     
     
class GRUSkip(hk.GRU):
    def __call__(self, inputs, state):
        return super().__call__(inputs, state)[0]
    

@dataclass
class OutputGate(hk.Module):
    hidden_dim: int

    def __call__(self, inputs, state):
            gate = nn.sigmoid(hk.Linear(self.hidden_dim)(state))
            return state + gate * inputs
    

@dataclass
class Highway(hk.Module):
    hidden_dim: int

    def __call__(self, inputs, state):
            gate = nn.sigmoid(hk.Linear(self.hidden_dim)(state))
            return gate * state + (1 - gate) * inputs
    

def create_highway(name: str, hidden_dim: int):
    name = name.lower()

    if name == "simple":
        return Simple(hidden_dim)
    elif name == "output":
        return OutputGate(hidden_dim)
    elif name == "highway":
         return Highway(hidden_dim)
    elif name == "gru":
         return GRUSkip(hidden_dim)
    else:
         raise ValueError(f"Unkown skip connection type: {name}")