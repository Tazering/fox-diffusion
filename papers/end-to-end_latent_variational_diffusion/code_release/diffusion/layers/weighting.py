from dataclasses import dataclass

from jax.scipy import stats
from jax import numpy as jnp
from jax import nn

import haiku as hk
    
def make_weighting(name: str, sigmoid_offset: float = 2.0):
    name == name.lower()

    if name == "cosine":
        return CosineWeighting()
    elif name == "edm":
        return EDMWeighting()
    elif name == "unit":
        return UnitWeighting()
    elif name == "sigmoid":
        return SigmoidWeighting(sigmoid_offset)
    else:
        raise ValueError(f"Unkown weighting: {name}")


class UnitWeighting(hk.Module):
    def __call__(self, gamma):
        return jnp.ones_like(gamma)
    

class CosineWeighting(hk.Module):
    def __call__(self, gamma):
        return 1.0 / jnp.cosh(-gamma / 2)
    
class EDMWeighting(hk.Module):
    def __call__(self, gamma):
        w1 = stats.norm.pdf(-gamma, 2.4, 2.4)
        w2 = jnp.exp(gamma) + 0.25

        return 5.0 * w1 * w2    

@dataclass
class SigmoidWeighting(hk.Module):    
    offset: float
    
    def __call__(self, gamma):
        return nn.sigmoid(gamma + self.offset)