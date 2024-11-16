import numpy as np

import jax
from jax import Array
from jax import numpy as jnp

import haiku as hk

from diffusion.config import Config
from diffusion.dataset import Batch


def without_return_state(transformed: hk.TransformedWithState):
    def apply_fn(params, state, *args, **kwargs):
        return transformed.apply(params, state, *args, **kwargs)[0]

    return hk.TransformedWithState(
        init=transformed.init,
        apply=apply_fn
    )


def multi_without_random_and_return_state(transformed: hk.MultiTransformedWithState):
    def wrap_apply_function(f):
        def apply_fn(params, state, *args, **kwargs):
            return f(params, state, None, *args, **kwargs)[0]

        return apply_fn

    return hk.MultiTransformedWithState(
        init=transformed.init,
        apply=jax.tree_map(wrap_apply_function, transformed.apply)
    )


def make_shared_model(config: Config):

    def normalize(batch: Batch) -> Batch:
            parton_mean = hk.get_state(
                "parton_mean", 
                shape=(config.parton_dim,), 
                init=hk.initializers.Constant(0.0)
            )

            parton_std = hk.get_state(
                "parton_std", 
                shape=(config.parton_dim,), 
                init=hk.initializers.Constant(1.0)
            )

            detector_mean = hk.get_state(
                "detector_mean", 
                shape=(config.detector_dim,), 
                init=hk.initializers.Constant(0.0)
            )
            detector_std = hk.get_state(
                "detector_std", 
                shape=(config.detector_dim,), 
                init=hk.initializers.Constant(1.0)
            )

            met_mean = hk.get_state(
                "met_mean", 
                shape=(config.met_dim,), 
                init=hk.initializers.Constant(0.0)
            )

            met_std = hk.get_state(
                "met_std", 
                shape=(config.met_dim,), 
                init=hk.initializers.Constant(1.0)
            )

            return Batch(
                parton_features=(batch.parton_features - parton_mean) / parton_std,
                detector_features=(batch.detector_features - detector_mean) / detector_std,
                detector_mask=batch.detector_mask,
                met_features=(batch.met_features - met_mean) / met_std,
                reco_targets=batch.reco_targets,
                weights=batch.weights
            )

    def denormalize(partons):
        parton_mean = hk.get_state(
            "parton_mean", 
            shape=(config.parton_dim,), 
            init=hk.initializers.Constant(0.0)
        )

        parton_std = hk.get_state(
            "parton_std", 
            shape=(config.parton_dim,), 
            init=hk.initializers.Constant(1.0)
        )

        return parton_std * partons + parton_mean

    def derived_top_masses_squared(parton):
        parton = denormalize(parton)
        parton = parton.reshape(parton.shape[0], 11, 5)

        px, py, pz, log_energy, mass = parton.transpose(2, 1, 0)
        energy = jnp.exp(log_energy) - 1

        lW = energy[1:3].sum(0) ** 2 - px[1:3].sum(0) ** 2 - \
            py[1:3].sum(0) ** 2 - pz[1:3].sum(0) ** 2
        hW = energy[4:6].sum(0) ** 2 - px[4:6].sum(0) ** 2 - \
            py[4:6].sum(0) ** 2 - pz[4:6].sum(0) ** 2

        lt = energy[0:3].sum(0) ** 2 - px[0:3].sum(0) ** 2 - \
            py[0:3].sum(0) ** 2 - pz[0:3].sum(0) ** 2
        ht = energy[3:6].sum(0) ** 2 - px[3:6].sum(0) ** 2 - \
            py[3:6].sum(0) ** 2 - pz[3:6].sum(0) ** 2

        tt = energy[0:6].sum(0) ** 2 - px[0:6].sum(0) ** 2 - \
            py[0:6].sum(0) ** 2 - pz[0:6].sum(0) ** 2

        masses = jnp.stack([lW, hW, lt, ht, tt], axis=-1)
        full_tree_masses = energy[6:] ** 2 - \
            px[6:] ** 2 - py[6:] ** 2 - pz[6:] ** 2
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
    

    return normalize, denormalize, derived_top_masses_squared, explicit_top_masses_squared
