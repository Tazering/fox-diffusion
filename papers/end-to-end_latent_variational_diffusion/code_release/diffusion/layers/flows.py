from typing import NamedTuple, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import jax
from jax import numpy as jnp
from jax import Array, nn

import haiku as hk


class FlowResult(NamedTuple):
    value: Array
    log_det_jac: Array


class Flow(hk.Module, ABC):
    @abstractmethod
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        raise NotImplementedError()
        
    @abstractmethod
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        raise NotImplementedError()

        
class Affine(Flow):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_hidden_layers: int = 1,
        alpha: float = 2.0,
        name: Optional[str] = None
    ):
        super(Affine, self).__init__(name=name)
        
        self.input_dim = input_dim
        self.split_dim_1 = input_dim // 2
        self.split_dim_2 = input_dim - self.split_dim_1

        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        for i in range(num_hidden_layers):
            layers.append(hk.Linear(hidden_dim))
            layers.append(nn.gelu)
        
        layers.append(hk.Linear(2 * self.split_dim_2))
        
        self.hypernetwork = hk.Sequential(layers)
        
    def affine_parameters(self, x1: Array, c: Optional[Array] = None):
        if c is not None:
            x1c = jnp.concatenate((x1, c), axis=-1)
        else:
            x1c = x1
            
        log_scale, offset = jnp.split(self.hypernetwork(x1c), 2, axis=-1)
        log_scale = self.alpha * jnp.tanh(log_scale * 0.1)
        
        return log_scale, offset
    
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        x1, x2 = jnp.split(x, [self.split_dim_1], axis=-1)
        
        log_scale, offset = self.affine_parameters(x1, c)
        
        y1 = x1
        y2 = x2 * jnp.exp(log_scale) + offset
        
        y = jnp.concatenate((y1, y2), axis=-1)
        log_jac = log_scale.sum(-1)
        
        return FlowResult(y, log_jac)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        x1, x2 = jnp.split(x, [self.split_dim_1], axis=-1)
        
        log_scale, offset = self.affine_parameters(x1, c)
        
        y1 = x1
        y2 = (x2 - offset) * jnp.exp(-log_scale)
        
        y = jnp.concatenate((y1, y2), axis=-1)
        log_jac = -log_scale.sum(-1)
        
        return FlowResult(y, log_jac)
    

class GlobalAffine(Flow):  
    @staticmethod  
    def affine_parameters(input_dim: int):
        global_scale = 2.0 * np.log(np.exp(5.0) - 1)
        log_scale = hk.get_parameter("log_scale", (input_dim,), init=hk.initializers.Constant(global_scale))
        offset = hk.get_parameter("offset", (input_dim,), init=hk.initializers.Constant(0.0))
        
        scale = 0.2 * nn.softplus(0.5 * log_scale)
        
        return scale, offset
    
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        scale, offset = self.affine_parameters(x.shape[-1])
        
        y = x * scale + offset
        jac = jnp.sum(jnp.log(scale))
        
        return FlowResult(y, jac)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        scale, offset = self.affine_parameters(x.shape[-1])
        
        y = (x - offset) / scale
        jac = -jnp.sum(jnp.log(scale))
        
        return FlowResult(y, jac)


class GlobalAffineEXP(Flow):  
    @staticmethod  
    def affine_parameters(input_dim: int):
        log_scale = hk.get_parameter("log_scale", (input_dim,), init=hk.initializers.Constant(0.0))
        offset = hk.get_parameter("offset", (input_dim,), init=hk.initializers.Constant(0.0))
                
        log_scale = 4.0 * nn.tanh(log_scale / 4.0)

        return log_scale, offset
    
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        log_scale, offset = self.affine_parameters(x.shape[-1])
        
        y = x * jnp.exp(log_scale) + offset
        jac = jnp.sum(log_scale)
        
        return FlowResult(y, jac)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        log_scale, offset = self.affine_parameters(x.shape[-1])
        
        y = (x - offset) * jnp.exp(-log_scale)
        jac = -jnp.sum(log_scale)
        
        return FlowResult(y, jac)
    

class SequentialFlow(Flow):
    def __init__(self, flows: List[Flow], name: Optional[str] = None):
        super(SequentialFlow, self).__init__(name=name)
        
        self.flows = flows
        
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        sum_log_jac = jnp.zeros(x.shape[:-1], x.dtype)
        
        for flow in self.flows:
            x, log_jac = flow(x, c)
            sum_log_jac += log_jac
        
        return FlowResult(x, sum_log_jac)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        sum_log_jac = jnp.zeros(x.shape[:-1], x.dtype)
        
        for flow in reversed(self.flows):
            x, log_jac = flow.inverse(x, c)
            sum_log_jac += log_jac
        
        return FlowResult(x, sum_log_jac)
    
    
class RandomPermutation(hk.initializers.Initializer):
    def __call__(self, shape: List[int], dtype) -> jnp.ndarray:
        return jax.random.permutation(hk.next_rng_key(), shape[-1])
    

class InversePermutation(hk.initializers.Initializer):
    def __init__(self, permutation: Array):
        self.permutation = permutation
    
    def __call__(self, shape: List[int], dtype) -> Array:
        return jnp.argsort(self.permutation)
    

@dataclass
class PermutationFlow(Flow):
    base_flow: Flow
    
    @staticmethod
    def permutation_parameters(input_dim: int):
        permutation = hk.get_state(
            "permutation", 
            shape=(input_dim,), 
            dtype=jnp.int32, 
            init=RandomPermutation()
        )

        inverse_permutation = hk.get_state(
            "inverse_permutation", 
            shape=(input_dim,), 
            dtype=jnp.int32, 
            init=InversePermutation(permutation)
        )

        return permutation, inverse_permutation
        

    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        permutation, inverse_permutation = self.permutation_parameters(x.shape[-1])

        x = jnp.take(x, inverse_permutation, axis=-1)
        y, log_det = self.base_flow(x, c)
        y = jnp.take(y, permutation, axis=-1)

        return FlowResult(y, log_det)
            
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        permutation, inverse_permutation = self.permutation_parameters(x.shape[-1])

        x = jnp.take(x, inverse_permutation, axis=-1)
        y, log_det = self.base_flow.inverse(x, c)
        y = jnp.take(y, permutation, axis=-1)

        return FlowResult(y, log_det)
    

@dataclass
class InverseFlow(Flow):
    base_flow: Flow
    
    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        return self.base_flow.inverse(x, c)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        return self.base_flow(x, c)
    

class AllInOneFlow(Flow):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_hidden_layers: int = 1,
        name: Optional[str] = None
    ):
        super(AllInOneFlow, self).__init__(name=name)
        
        self.flow = PermutationFlow(SequentialFlow([
            Affine(input_dim, hidden_dim, num_hidden_layers),
            GlobalAffineEXP()
        ]))

        # self.flow = SequentialFlow([
        #     Affine(input_dim, hidden_dim, num_hidden_layers),
        #     GlobalAffine()
        # ])

    def __call__(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        return self.flow(x, c)
    
    def inverse(self, x: Array, c: Optional[Array] = None) -> FlowResult:
        return self.flow.inverse(x, c)
        