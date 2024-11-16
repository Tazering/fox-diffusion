from typing import NamedTuple, Dict, Any, Optional, Tuple, List
from functools import partial

import numpy as np

import jax
from jax import Array
from jax import numpy as jnp

import haiku as hk

from tensorflow_probability.substrates.jax import distributions

from diffusion.config import Config
from diffusion.dataset import Batch
from diffusion.detector_encoder import DetectorEncoder
from diffusion.parton_decoder import PartonDecoder
from diffusion.parton_encoder import PartonEncoder

from shared_model import make_shared_model, multi_without_random_and_return_state


class CVAE(NamedTuple):
    detector_encoder: Any
    parton_encoder: Any
    parton_decoder: Any
    normalize: Any
    denormalize: Any
    derived_top_masses_squared: Any
    explicit_top_masses_squared: Any


class CVAEMetrics(NamedTuple):
    reconstruction_loss: Array
    prior_loss: Array
    self_mass_loss: Array


def create_cvae_model(config: Config):
    @multi_without_random_and_return_state
    @hk.multi_transform_with_state
    def cvae():
        detector_encoder = DetectorEncoder(
            config.hidden_dim,
            config.num_heads,
            config.num_detector_encoder_layers
        )

        parton_encoder = PartonEncoder(
            config.hidden_dim,
            config.num_parton_encoder_layers,
            config.normalize_parton,
            config.normalize_parton_scale,
            config.trivial_vae,
            conditional_vae=True
        )

        parton_decoder = PartonDecoder(
            config.hidden_dim, 
            config.parton_dim, 
            config.num_parton_decoder_layers,
            config.trivial_vae,
            conditional_vae=True
        )

        (
            normalize, 
            denormalize, 
            derived_top_masses_squared, 
            explicit_top_masses_squared
        ) = make_shared_model(config)
        

        def init(batch: Batch):
            batch = normalize(batch)

            detector = detector_encoder(
                batch.detector_features,
                batch.detector_mask,
                batch.met_features
            )

            mean, log_std = parton_encoder(batch.parton_features, detector)

            encoded_distibution = distributions.Normal(mean, jnp.exp(log_std))
            encoded_parton = encoded_distibution.sample(seed=hk.next_rng_key())

            new_partons = parton_decoder(encoded_parton, detector)
            return new_partons
        
        return init, CVAE(
            detector_encoder,
            parton_encoder,
            parton_decoder,
            normalize,
            denormalize,
            derived_top_masses_squared,
            explicit_top_masses_squared
        )


    def cvae_losses(params, state, key: jax.random.PRNGKey, batch: Batch):
        # Normalize Inputs based on training statistics
        # -------------------------------------------------------------------------
        batch = cvae.apply.normalize(params, state, batch)

        # Embedd detector observations for conditionning.
        detector = cvae.apply.detector_encoder(
            params, 
            state, 
            batch.detector_features,
            batch.detector_mask,
            batch.met_features
        )

        # Encode the partons q(z|x,c).
        parton_mean, parton_log_std = cvae.apply.parton_encoder(
            params, 
            state, 
            batch.parton_features,
            detector
        )

        parton_distribution = distributions.MultivariateNormalDiag(
            parton_mean, 
            jnp.exp(parton_log_std)
        )

        # Sample a parton embedding from VAE z
        key, parton_key = jax.random.split(key, 2)

        if config.vae_prior_loss_scale > 0:
            encoded_parton = parton_distribution.sample(seed=parton_key)
        else:
            encoded_parton = parton_mean

        # Sample a reconstructed vector p(x|z,c)
        decoded_parton = cvae.apply.parton_decoder(
            params, 
            state,
            encoded_parton,
            detector
        )

        reconstruction_loss = jnp.square(decoded_parton - batch.parton_features).mean(1)

        if config.vae_prior_loss_scale > 0:
            parton_log_var = 2.0 * parton_log_std
            prior_loss = 0.5 * (
                + jnp.exp(parton_log_var) 
                + jnp.square(parton_mean) 
                - parton_log_var
                - 1.0
            ).mean(1)
        else:
            prior_loss = jnp.zeros_like(reconstruction_loss)
            
        if config.self_mass_loss_scale > 0:
            derived_masses = cvae.apply.derived_top_masses_squared(params, state, decoded_parton)
            explicit_masses = cvae.apply.explicit_top_masses_squared(params, state, decoded_parton)
            self_mass_loss = jnp.square(explicit_masses - derived_masses).mean(1)
        else:
            self_mass_loss = jnp.zeros_like(reconstruction_loss)


        return CVAEMetrics(
            reconstruction_loss,
            prior_loss,
            self_mass_loss
        ),  key


    @partial(jax.value_and_grad, has_aux=True)
    def cvae_step(params, state, key: jax.random.PRNGKey, batch: Batch):
        losses, key = cvae_losses(params, state, key, batch)
        
        total_loss = (
            + config.reconstruction_loss_scale * losses.reconstruction_loss
            + config.vae_prior_loss_scale * losses.prior_loss
            + config.self_mass_loss_scale * losses.self_mass_loss
        )

        total_loss = batch.weights * total_loss
            
        return jnp.mean(total_loss), (jax.tree_map(jnp.mean, losses), key)
    

    return cvae, cvae_losses, cvae_step
