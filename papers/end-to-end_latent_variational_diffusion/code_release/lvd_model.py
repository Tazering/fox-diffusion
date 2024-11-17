from typing import NamedTuple, Dict, Any
import numpy as np

import jax
from jax import numpy as jnp
from jax import Array, nn

from functools import partial

import haiku as hk
import tensorflow as tf

from tensorflow_probability.substrates.jax import distributions, math

from diffusion.dataset import Batch
from diffusion.config import Config
from diffusion.detector_encoder import DetectorEncoder
from diffusion.parton_encoder import PartonEncoder
from diffusion.parton_decoder import PartonDecoder

from diffusion.layers.noise_scheduler import make_noise_schedule
from diffusion.layers.weighting import make_weighting

from diffusion.denoising_network import DenoisingNetwork



FLOAT_GAMMA_MIN = -16.0
FLOAT_GAMMA_MAX = 16.0


def softclip(arr: Array, min: float) -> Array:
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    return min + nn.softplus(arr - min)


def no_state_and_rand(f):
    def wrapped(params, *args):
        return f(params, {}, None, *args)[0]
    return wrapped


def no_return_state(f):
    def wrapped(params, state, *args):
        return f(params, state, None, *args)[0]
    return wrapped


class VDMMetrics(NamedTuple):
    reconstruction_loss: Array

    mass_loss: Array
    self_mass_loss: Array

    diffusion_loss: Array

    prior_loss: Array
    gamma_prior_loss: Array
    kl_loss: Array

    diffusion_mse: Array
    gamma_0: Array
    gamma_1: Array


class GammaMetrics(NamedTuple):
    diffusion_variance: Array


class VDM(NamedTuple):
    detector_encoder: Any
    parton_encoder: Any
    parton_decoder: Any
    denoising_network: Any
    normalize: Any
    denormalize: Any
    gamma_limits: Any
    derived_top_masses_squared: Any
    explicit_top_masses_squared: Any


class Gamma(NamedTuple):
    gamma: Any
    gamma_prime: Any
    weights: Any

"""
This function below seems to create a new model.
"""
def create_model(config: Config):
    @hk.multi_transform_with_state
    def variation_diffusion_model():
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
            config.conditional_vae
        )

        parton_decoder = PartonDecoder(
            config.hidden_dim, 
            config.parton_dim, 
            config.num_parton_decoder_layers,
            config.trivial_vae,
            config.conditional_vae and (not config.unconditional_vae_decoder)
        )
        
        denoising_network = DenoisingNetwork(
            config.hidden_dim,
            config.timestep_dim,
            config.num_denoising_layers
        )
        
        def normalize(batch: Batch) -> Batch:
            parton_mean = hk.get_state("parton_mean", shape=(config.parton_dim,), init=hk.initializers.Constant(0.0))
            parton_std = hk.get_state("parton_std", shape=(config.parton_dim,), init=hk.initializers.Constant(1.0))

            detector_mean = hk.get_state("detector_mean", shape=(config.detector_dim,), init=hk.initializers.Constant(0.0))
            detector_std = hk.get_state("detector_std", shape=(config.detector_dim,), init=hk.initializers.Constant(1.0))

            met_mean = hk.get_state("met_mean", shape=(config.met_dim,), init=hk.initializers.Constant(0.0))
            met_std = hk.get_state("met_std", shape=(config.met_dim,), init=hk.initializers.Constant(1.0))

            return Batch(
                parton_features=(batch.parton_features - parton_mean) / parton_std,
                detector_features=(batch.detector_features - detector_mean) / detector_std,
                detector_mask=batch.detector_mask,
                met_features=(batch.met_features - met_mean) / met_std,
                reco_targets=batch.reco_targets,
                weights=batch.weights
            )

        def denormalize(partons):
            parton_mean = hk.get_state("parton_mean", shape=(config.parton_dim,), init=hk.initializers.Constant(0.0))
            parton_std = hk.get_state("parton_std", shape=(config.parton_dim,), init=hk.initializers.Constant(1.0))

            return parton_std * partons + parton_mean            

        def derived_top_masses_squared(parton):
            parton = denormalize(parton)
            parton = parton.reshape(parton.shape[0], 11, 5)

            px, py, pz, log_energy, mass = parton.transpose(2, 1, 0)
            energy = jnp.exp(log_energy) - 1

            lW = energy[1:3].sum(0) ** 2 - px[1:3].sum(0) ** 2 - py[1:3].sum(0) ** 2 - pz[1:3].sum(0) ** 2
            hW = energy[4:6].sum(0) ** 2 - px[4:6].sum(0) ** 2 - py[4:6].sum(0) ** 2 - pz[4:6].sum(0) ** 2

            lt = energy[0:3].sum(0) ** 2 - px[0:3].sum(0) ** 2 - py[0:3].sum(0) ** 2 - pz[0:3].sum(0) ** 2
            ht = energy[3:6].sum(0) ** 2 - px[3:6].sum(0) ** 2 - py[3:6].sum(0) ** 2 - pz[3:6].sum(0) ** 2

            tt = energy[0:6].sum(0) ** 2 - px[0:6].sum(0) ** 2 - py[0:6].sum(0) ** 2 - pz[0:6].sum(0) ** 2

            masses = jnp.stack([lW, hW, lt, ht, tt], axis=-1)
            full_tree_masses = energy[6:] ** 2 - px[6:] ** 2 - py[6:] ** 2 - pz[6:] ** 2
            full_tree_masses = full_tree_masses.T

            square_mass_mean = hk.get_state("square_mass_mean", shape=(5,), init=hk.initializers.Constant(0.0))
            square_mass_std = hk.get_state("square_mass_std", shape=(5,), init=hk.initializers.Constant(1.0))

            masses = (masses - square_mass_mean) / square_mass_std
            full_tree_masses = (full_tree_masses - square_mass_mean) / square_mass_std

            return jnp.concatenate((masses, full_tree_masses), axis=-1)

        def explicit_top_masses_squared(parton):
            parton = denormalize(parton)
            parton = parton.reshape(parton.shape[0], 11, 5)

            square_masses = jnp.square(parton[:, 6:, 4])

            square_mass_mean = hk.get_state("square_mass_mean", shape=(5,), init=hk.initializers.Constant(0.0))
            square_mass_std = hk.get_state("square_mass_std", shape=(5,), init=hk.initializers.Constant(1.0))
            square_masses = (square_masses - square_mass_mean) / square_mass_std

            return jnp.concatenate((square_masses, square_masses), axis=-1)

        def gamma_limits():
            gamma_min = hk.get_parameter("gamma_min", (config.noise_schedule_outputs,), init=hk.initializers.Constant(config.initial_gamma_min))
            gamma_max = hk.get_parameter("gamma_max", (config.noise_schedule_outputs,), init=hk.initializers.Constant(config.initial_gamma_max))
            
            gamma_min = softclip(gamma_min, FLOAT_GAMMA_MIN)
            gamma_max = -softclip(-gamma_max, -FLOAT_GAMMA_MAX)

            return gamma_min, gamma_max

        def init(batch: Batch):
            batch = normalize(batch)
            
            encoded_detector = detector_encoder(
                batch.detector_features,
                batch.detector_mask,
                batch.met_features
            )
            
            mean, log_std = parton_encoder(batch.parton_features, encoded_detector)
            encoded_distibution = distributions.Normal(mean, jnp.exp(log_std))
            encoded_parton = encoded_distibution.sample(seed=hk.next_rng_key())
            
            gamma_min, gamma_max = gamma_limits()
            timesteps = distributions.Uniform(gamma_min, gamma_max).sample((encoded_parton.shape[0],), seed=hk.next_rng_key())
            
            denoised_partons = denoising_network(encoded_parton, encoded_detector, timesteps)
            new_partons = parton_decoder(denoised_partons, encoded_detector)
            return new_partons
            # return derived_top_masses_squared(new_partons)

        
        return init, VDM(
            detector_encoder, 
            parton_encoder, 
            parton_decoder, 
            denoising_network, 
            normalize, 
            denormalize,
            gamma_limits,
            derived_top_masses_squared,
            explicit_top_masses_squared
        )

    @hk.multi_transform
    def noise_scheduler():       
        gamma = make_noise_schedule(
            config.noise_schedule,
            config.noise_schedule_outputs,
        )

        weights = make_weighting(
            config.weighting,
            config.sigmoid_weight_offset
        )
                
        gamma_prime = jax.vmap(jax.jacobian(gamma), in_axes=[0, None, None])
        
        def init(batch: Batch):
            t = jnp.linspace(0, 1, batch.parton_features.shape[0])
            gamma_min = -jnp.ones(config.noise_schedule_outputs)
            gamma_max = jnp.ones(config.noise_schedule_outputs)
            
            return gamma(t, gamma_min, gamma_max), gamma_prime(t, gamma_min, gamma_max), weights(gamma(t, gamma_min, gamma_max))
        
        return init, Gamma(gamma, gamma_prime, weights)

    # Extract simple versions of the networks
    (
        detector_encoder, 
        parton_encoder, 
        parton_decoder, 
        denoising_network, 
        normalize, 
        denormalize, 
        gamma_limits, 
        derived_top_masses_squared, 
        explicit_top_masses_squared 
    ) = variation_diffusion_model.apply

    detector_encoder = no_state_and_rand(detector_encoder)
    parton_encoder = no_state_and_rand(parton_encoder)
    parton_decoder = no_state_and_rand(parton_decoder)
    denoising_network = no_state_and_rand(denoising_network)
    normalize = no_return_state(normalize)
    denormalize = no_return_state(denormalize)
    gamma_limits = no_state_and_rand(gamma_limits)
    derived_top_masses_squared = no_return_state(derived_top_masses_squared)
    explicit_top_masses_squared = no_return_state(explicit_top_masses_squared)

    gamma, gamma_prime, weights = noise_scheduler.apply

    def compute_diffusion_loss(
        vdm_params, 
        gamma_params, 
        key: jax.random.PRNGKey, 
        detector,
        parton
    ):
        key, timestep_key, noise_key = jax.random.split(key, 3)
        
        # Generate evenly spread timesteps
        # -----------------------------------------------------------------------------------------
        # timesteps: (B,)
        # -----------------------------------------------------------------------------------------
        t0 = jax.random.uniform(timestep_key)    
        timesteps = jnp.linspace(0.0, 1.0, parton.shape[0])
        timesteps = jnp.mod(t0 + timesteps, 1.0)

        # Generate noise parameters at timesteps
        # -----------------------------------------------------------------------------------------
        # gamma_t: (B, G)
        # sigma_t: (B, G)
        # alpha_squared_t: (B, G)
        # alpha_t: (B, G)
        # -----------------------------------------------------------------------------------------
        gamma_t = gamma(gamma_params, None, timesteps, *gamma_limits(vdm_params))
        sigma_t = jnp.sqrt(nn.sigmoid(gamma_t))

        alpha_squared_t = nn.sigmoid(-gamma_t)
        alpha_t = jnp.sqrt(alpha_squared_t)

        # Generate noise and noisy latent
        # -----------------------------------------------------------------------------------------
        # eps: (B, D)
        # z_t: (B, D)
        # -----------------------------------------------------------------------------------------
        eps = jax.random.normal(noise_key, parton.shape, parton.dtype)
        z_t = alpha_t * parton + sigma_t * eps

        # Predict Noise and compute MSE loss
        # -----------------------------------------------------------------------------------------
        # eps_hat: (B, D)
        # weights_t: (B, G)
        # gamma_prime_t: (B, G)
        # -----------------------------------------------------------------------------------------
        eps_hat = denoising_network(vdm_params, z_t, detector, alpha_squared_t)

        weights_t = weights(gamma_params, None, gamma_t)
        gamma_prime_t = gamma_prime(gamma_params, None, timesteps, *gamma_limits(vdm_params))

        diffusion_mse = jnp.square(eps_hat - eps) * weights_t
        diffusion_loss = 0.5 * gamma_prime_t * diffusion_mse

        return diffusion_loss.mean(1), diffusion_mse.mean(1), key
    

    def vdm_losses(vdm_params, gamma_params, state, key: jax.random.PRNGKey, batch: Batch):
        # Normalize Inputs based on training statistics
        # -----------------------------------------------------------------------------------------
        batch = normalize(vdm_params, state, batch)
        
        # Encode partons and detector variables into latent space.
        # -----------------------------------------------------------------------------------------
        # detector: (B, D)
        # parton_*: (B, D)
        # -----------------------------------------------------------------------------------------
        detector = detector_encoder(
            vdm_params, 
            batch.detector_features, 
            batch.detector_mask, 
            batch.met_features
        )
        
        parton_mean, parton_log_std = parton_encoder(vdm_params, batch.parton_features, detector)

        # Construct p(z|x).
        parton_distribution = distributions.MultivariateNormalDiag(
            parton_mean, 
            jnp.exp(parton_log_std)
        )
            
        # Sample a parton embedding from VAE
        key, parton_key = jax.random.split(key, 2)

        if config.deterministic_vae:
            parton = parton_mean
        else:
            parton = parton_distribution.sample(seed=parton_key)
            
        # Noise Schedule reconstruction loss.
        # -----------------------------------------------------------------------------------------
        # gamma_0: (G,)
        # scale_sigma: (G,)
        # -----------------------------------------------------------------------------------------
        gamma_0 = gamma(gamma_params, None, 0.0, *gamma_limits(vdm_params))
        scale_sigma = jnp.exp(0.5 * gamma_0)

        key, noise_key = jax.random.split(key, 2)
        eps_0 = jax.random.normal(noise_key, parton.shape, parton.dtype)
        z_0_rescaled = parton + scale_sigma * eps_0

        # Explicitely separate the heirarchy between the VAE and the first diffusion timestep.
        # -----------------------------------------------------------------------------------------
        if config.kl_loss_scale > 0:
            decoded_parton = parton_decoder(vdm_params, parton, detector)
            reconstruction_loss = jnp.square(decoded_parton - batch.parton_features).mean(1)

            log_var_0 = 2.0 * parton_log_std
            log_var_1 = gamma_0

            kl_loss = 0.5 * (
                + jnp.exp(log_var_0 - log_var_1)
                + jnp.square(parton_mean - z_0_rescaled) / jnp.exp(log_var_1)
                + log_var_1
                - log_var_0
                - 1.0
            ).mean(1)

        # Or combine the fist timestep and the VAE into a single latent distribution
        else:
            decoded_parton = parton_decoder(vdm_params, z_0_rescaled, detector)
            reconstruction_loss = jnp.square(decoded_parton - batch.parton_features).mean(1)

            kl_loss = jnp.zeros_like(reconstruction_loss)

        # Extra reconstruction loss based on the top mass relationships.
        # -----------------------------------------------------------------------------------------
        if config.mass_loss_scale > 0:
            true_masses = derived_top_masses_squared(vdm_params, state, batch.parton_features)
            pred_masses = derived_top_masses_squared(vdm_params, state, decoded_parton)
            mass_loss = jnp.square(true_masses - pred_masses).mean(1)
        else:
            mass_loss = jnp.zeros_like(reconstruction_loss)


        if config.self_mass_loss_scale > 0:
            derived_masses = derived_top_masses_squared(vdm_params, state, decoded_parton)
            explicit_masses = explicit_top_masses_squared(vdm_params, state, decoded_parton)
            self_mass_loss = jnp.square(explicit_masses - derived_masses).mean(1)
        else:
            self_mass_loss = jnp.zeros_like(reconstruction_loss)
 
        # VAE explicit prior Loss (This is NOT theoretically correct)
        # -----------------------------------------------------------------------------------------
        if config.vae_prior_loss_scale > 0:
            parton_log_var = 2.0 * parton_log_std
            prior_loss = 0.5 * (
                + jnp.square(parton_mean) 
                + jnp.exp(parton_log_var) 
                - parton_log_var 
                - 1.0
            ).mean(1)
        else:
            prior_loss = jnp.zeros_like(reconstruction_loss)
        
        
        # Diffusion Prior Loss
        # -----------------------------------------------------------------------------------------
        # gamma_1: (G,)
        # var_1: (G,)
        # alpha_squared_1: (G,)
        # log_var_1: (G,)
        # -----------------------------------------------------------------------------------------
        gamma_1 = gamma(gamma_params, None, 1.0, *gamma_limits(vdm_params))
        var_1 = nn.sigmoid(gamma_1)
        alpha_squared_1 = nn.sigmoid(-gamma_1)
        log_var_1 = nn.log_sigmoid(gamma_1)

        mean1_sqr = alpha_squared_1 * jnp.square(parton)
        gamma_prior_loss = 0.5 * (
            + mean1_sqr 
            + var_1 
            - log_var_1 
            - 1.0
        ).mean(1)

        # Diffusion Loss
        # -----------------------------------------------------------------------------------------
        diffusion_loss, diffusion_mse, key = compute_diffusion_loss(
            vdm_params, 
            gamma_params, 
            key, 
            detector,
            parton
        )
        
        return VDMMetrics(
            reconstruction_loss, 
            mass_loss,
            self_mass_loss,
            diffusion_loss, 
            prior_loss, 
            gamma_prior_loss,
            kl_loss,
            diffusion_mse,
            gamma_0,
            gamma_1
        ), key
    

    def gamma_loss(vdm_params, gamma_params, state, key: jax.random.PRNGKey, batch: Batch):
        # Normalize Inputs based on training statistics
        # -----------------------------------------------------------------------------------------
        batch = normalize(vdm_params, state, batch)
        
        # Encode partons and detector variables into latent space.
        # -----------------------------------------------------------------------------------------
        detector = detector_encoder(
            vdm_params, 
            batch.detector_features, 
            batch.detector_mask, 
            batch.met_features
        )
        
        parton_mean, _ = parton_encoder(vdm_params, batch.parton_features, detector)
        
        # Diffusion Loss - Used for variance reduction
        # -----------------------------------------------------------------------------------------
        diffusion_loss, _, key = compute_diffusion_loss(
            vdm_params, 
            gamma_params, 
            key, 
            detector,
            parton_mean
        )

        return GammaMetrics(jnp.var(diffusion_loss)), key

    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def vdm_step(vdm_params, gamma_params, state, key: jax.random.PRNGKey, batch: Batch):
        losses, key = vdm_losses(vdm_params, gamma_params, state, key, batch)
        
        total_loss = (
            + config.reconstruction_loss_scale * losses.reconstruction_loss 
            + config.mass_loss_scale * losses.mass_loss
            + config.self_mass_loss_scale * losses.self_mass_loss
            + losses.diffusion_loss 
            + config.vae_prior_loss_scale * losses.prior_loss
            + config.kl_loss_scale * losses.kl_loss
            + losses.gamma_prior_loss
        )

        total_loss = batch.weights * total_loss
            
        return jnp.mean(total_loss), (jax.tree_map(jnp.mean, losses), key)
    
    @jax.jit
    @partial(jax.value_and_grad, has_aux=True, argnums=1)
    def gamma_step(vdm_params, gamma_params, state, key: jax.random.PRNGKey, batch: Batch):
        losses, key = gamma_loss(vdm_params, gamma_params, state, key, batch)
        
        return losses.diffusion_variance, (losses, key)

    return variation_diffusion_model, noise_scheduler, vdm_step, gamma_step