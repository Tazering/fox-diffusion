from typing import NamedTuple, Dict, Any, Optional, Tuple, List
from functools import partial

import numpy as np

import jax
from jax import Array
from jax import numpy as jnp

import haiku as hk

from tensorflow_probability.substrates.jax import distributions

from diffusion.layers.flows import AllInOneFlow, SequentialFlow

from diffusion.config import Config
from diffusion.dataset import Batch
from diffusion.detector_encoder import DetectorEncoder

from shared_model import make_shared_model, multi_without_random_and_return_state


class CINN(NamedTuple):
    forward_flow: Any
    inverse_flow: Any
    normalize: Any
    denormalize: Any
    derived_top_masses_squared: Any
    explicit_top_masses_squared: Any


class CINNMetrics(NamedTuple):
    prior_loss: Array
    log_determinant_loss: Array
    reconstruction_loss: Array
    self_mass_loss: Array


def create_flow_model(config: Config):
    @multi_without_random_and_return_state
    @hk.multi_transform_with_state
    def cinn():
        cinn = SequentialFlow([
            AllInOneFlow(
                config.parton_dim,
                config.parton_flow_units_per_layer,
                config.parton_flow_layers_per_block
            )

            for _ in range(config.parton_flow_blocks)
        ])

        detector_encoder = DetectorEncoder(
            config.hidden_dim,
            config.num_heads,
            config.num_detector_encoder_layers
        )

        def forward_flow(
            random_vector,
            detector_features: jax.Array,  # [B, T, D]
            detector_mask: jax.Array,  # [B, T],
            met_features: jax.Array,  # [B, M]):
        ):
            conditioning = detector_encoder(
                detector_features,
                detector_mask,
                met_features
            )

            return cinn(random_vector, conditioning)

        def inverse_flow(
            parton_features,
            detector_features: jax.Array,  # [B, T, D]
            detector_mask: jax.Array,  # [B, T],
            met_features: jax.Array,  # [B, M]):
        ):
            conditioning = detector_encoder(
                detector_features,
                detector_mask,
                met_features
            )

            return cinn.inverse(parton_features, conditioning)

        (
            normalize, 
            denormalize, 
            derived_top_masses_squared, 
            explicit_top_masses_squared
        ) = make_shared_model(config)

        
        def init(batch: Batch):
            batch = normalize(batch)

            return inverse_flow(
                batch.parton_features,
                batch.detector_features,
                batch.detector_mask,
                batch.met_features
            )

        return init, CINN(
            forward_flow,
            inverse_flow,
            normalize,
            denormalize,
            derived_top_masses_squared,
            explicit_top_masses_squared
        )


    def cinn_losses(params, state, key: jax.random.PRNGKey, batch: Batch):
        # Normalize Inputs based on training statistics
        # -------------------------------------------------------------------------
        batch = cinn.apply.normalize(params, state, batch)

        forward_map = cinn.apply.inverse_flow(
            params, 
            state, 
            batch.parton_features,
            batch.detector_features,
            batch.detector_mask,
            batch.met_features
        )

        prior_distribution = distributions.Normal(
            jnp.zeros_like(forward_map.value),
            jnp.ones_like(forward_map.value)
        )

        prior_loss = -prior_distribution.log_prob(forward_map.value).sum(1)
        
        log_determinant_loss = -forward_map.log_det_jac

        # Sample from the prior
        key, normal_key = jax.random.split(key, 2)
        z = jax.random.normal(
            normal_key, 
            batch.parton_features.shape, 
            batch.parton_features.dtype
        )

        # Pass the sample through the forward flow and compare the resulting masses.
        decoded_parton = cinn.apply.forward_flow(
            params,
            state,
            z,
            batch.detector_features,
            batch.detector_mask,
            batch.met_features
        ).value

        reconstruction_loss = jnp.mean(jnp.square(decoded_parton - batch.parton_features))
        
        if config.self_mass_loss_scale > 0:
            derived_masses = cinn.apply.derived_top_masses_squared(params, state, decoded_parton)
            explicit_masses = cinn.apply.explicit_top_masses_squared(params, state, decoded_parton)
            self_mass_loss = jnp.square(explicit_masses - derived_masses).mean(1)
        else:
            self_mass_loss = jnp.zeros_like(reconstruction_loss)


        return CINNMetrics(
            prior_loss,
            log_determinant_loss,
            reconstruction_loss,
            self_mass_loss
        ),  key


    @partial(jax.value_and_grad, has_aux=True)
    def cinn_step(params, state, key: jax.random.PRNGKey, batch: Batch):
        losses, key = cinn_losses(params, state, key, batch)
        
        total_loss = (
            + losses.prior_loss
            + losses.log_determinant_loss
            # + config.reconstruction_loss_scale * losses.reconstruction_loss
            + config.self_mass_loss_scale * losses.self_mass_loss
        )

        total_loss = batch.weights * total_loss
            
        return jnp.mean(total_loss), (jax.tree_map(jnp.mean, losses), key)
    

    return cinn, cinn_losses, cinn_step
