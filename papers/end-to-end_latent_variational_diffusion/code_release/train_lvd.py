from typing import NamedTuple, Dict
from os import makedirs
from glob import glob

from argparse import ArgumentParser
from functools import partial
from itertools import islice

import pickle
from omegaconf import OmegaConf
from tqdm import tqdm

import jax
from jax import Array


import optax
import tensorflow as tf

from flax.jax_utils import replicate, unreplicate

from diffusion.dataset import Dataset, Batch
from diffusion.config import Config

from lvd_model import create_model

# 
class TrainingState(NamedTuple):
    vdm_params: Dict[str, Array]
    gamma_params: Dict[str, Array]
    vdm_state: Dict[str, Array]

    vdm_optimizer_state: Dict[str, Array]
    gamma_optimizer_state: Dict[str, Array]

# creates a log folder
def create_log_folder(logdir: str, name: str):
    base_dir = f"{logdir}/{name}"
    makedirs(base_dir, exist_ok=True)

    next_version = len(glob(f"{base_dir}/version*"))
    log_folder = f"{base_dir}/version_{next_version}"
    makedirs(log_folder, exist_ok=True)

    return log_folder

# 
def create_vdm_update(vdm_step, vdm_optimizer):
    @partial(jax.pmap, axis_name="num_devices")
    def vdm_update(
        training_state: TrainingState,
        random_key: jax.random.PRNGKey,
        batch: Batch
    ):
        (total_loss, (metrics, random_key)), grads = vdm_step(
            training_state.vdm_params,
            training_state.gamma_params,
            training_state.vdm_state,
            random_key,
            batch
        )

        grads = jax.lax.pmean(grads, "num_devices")

        updates, vdm_optimizer_state = vdm_optimizer.update(
            grads,
            training_state.vdm_optimizer_state,
            training_state.vdm_params
        )

        vdm_params = optax.apply_updates(training_state.vdm_params, updates)

        training_state = training_state._replace(
            vdm_params=vdm_params,
            vdm_optimizer_state=vdm_optimizer_state
        )

        return training_state, total_loss, metrics, random_key

    return vdm_update


def create_gamma_update(gamma_step, gamma_optimizer):

    @partial(jax.pmap, axis_name="num_devices")
    def gamma_update(
        training_state: TrainingState,
        random_key: jax.random.PRNGKey,
        batch: Batch
    ):
        (total_loss, (metrics, random_key)), grads = gamma_step(
            training_state.vdm_params,
            training_state.gamma_params,
            training_state.vdm_state,
            random_key,
            batch
        )

        grads = jax.lax.pmean(grads, "num_devices")

        updates, gamma_optimizer_state = gamma_optimizer.update(
            grads,
            training_state.gamma_optimizer_state,
            training_state.gamma_params
        )

        gamma_params = optax.apply_updates(
            training_state.gamma_params, updates)

        training_state = training_state._replace(
            gamma_params=gamma_params,
            gamma_optimizer_state=gamma_optimizer_state
        )

        return training_state, total_loss, metrics, random_key

    return gamma_update


def make_optimizer(learning_rate, gradient_clip=None):
    optimizer = optax.adam(learning_rate)

    if gradient_clip is not None:
        clipping = optax.clip_by_global_norm(gradient_clip)
        optimizer = optax.chain(clipping, optimizer)

    return optimizer

# train the vld
def train(
    options_file: str,
    training_file: str,
    checkpoint_file: str,
    start_batch: int,
    name: str,
    weights_file
):
    # initialize cuda
    jax.random.normal(jax.random.PRNGKey(0))

    # create the dataset
    print("Loading Data")
    dataset = Dataset(training_file, weights_file=weights_file)

    # setup the configuration
    config = Config(
        **OmegaConf.load(options_file),
        parton_dim=dataset.parton_dim,
        detector_dim=dataset.detector_dim,
        met_dim=dataset.met_dim
    )

    # create a dataloader from a dataset
    dataloader = dataset.create_dataloader(config.batch_size)
    single_device_batch = jax.tree_map(lambda x: x[0], next(dataloader))

    # creates the variational diffusion model
    variation_diffusion_model, noise_scheduler, vdm_step, gamma_step = create_model(
        config)

    # make optimizers
    vdm_optimizer = make_optimizer(
        config.learning_rate, config.gradient_clipping)
    
    # gamma optimizer
    gamma_optimizer = make_optimizer(
        config.learning_rate, config.gradient_clipping)

    # Initialize Model on GPU 0
    # -------------------------------------------------------------------------
    print("Initializing Model")
    # initialize using pseudo-random number generator
    random_key = jax.random.PRNGKey(config.seed)
    random_key, vdm_key, gamma_key = jax.random.split(random_key, 3)

    if checkpoint_file is not None:
        with open(checkpoint_file, 'rb') as file:
            training_state = pickle.load(file)

    else:
        vdm_params, vdm_state = variation_diffusion_model.init(
            vdm_key, single_device_batch)
        gamma_params = noise_scheduler.init(gamma_key, single_device_batch)

        vdm_optimizer_state = vdm_optimizer.init(vdm_params)
        gamma_optimizer_state = gamma_optimizer.init(gamma_params)

        vdm_state["~"] = dataset.statistics

        training_state = TrainingState(
            vdm_params,
            gamma_params,
            vdm_state,

            vdm_optimizer_state,
            gamma_optimizer_state
        )

    # Create shared parameters on all devices.
    # -------------------------------------------------------------------------
    random_key = jax.random.split(random_key, jax.device_count())
    training_state = replicate(training_state)

    # Create Update functions
    # -------------------------------------------------------------------------
    vdm_update = create_vdm_update(vdm_step, vdm_optimizer)
    gamma_update = create_gamma_update(gamma_step, gamma_optimizer)

    logdir = create_log_folder("./logs", name)
    OmegaConf.save(OmegaConf.structured(config), f"{logdir}/config.yaml")

    summary_writer = tf.summary.create_file_writer(logdir)
    batch_number = start_batch

    with summary_writer.as_default():
        if config.num_batches > 0:
            pbar = tqdm(islice(dataloader, config.num_batches),
                        desc="Training", total=config.num_batches)
        else:
            pbar = tqdm(dataloader, desc="Training")

        for batch in pbar:
            training_state, _, vdm_metrics, random_key = vdm_update(
                training_state, random_key, batch)
            training_state, _, gamma_metrics, random_key = gamma_update(
                training_state, random_key, batch)

            if batch_number % config.log_interval == 0:
                vdm_metrics = {
                    f"train/vdm/{name}": value.mean().item()
                    for name, value
                    in vdm_metrics._asdict().items()
                }

                gamma_metrics = {
                    f"train/gamma/{name}": value.mean().item()
                    for name, value
                    in gamma_metrics._asdict().items()
                }

                metrics = vdm_metrics | gamma_metrics
                # metrics = vdm_metrics
                for name, value in metrics.items():
                    tf.summary.scalar(name, value, step=batch_number)

            if batch_number % config.save_interval == 0:
                with open(f"{logdir}/checkpoint.pickle", 'wb') as file:
                    pickle.dump(unreplicate(training_state), file)

            batch_number += 1

# parse out command line arguments
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("options_file", type=str)
    parser.add_argument("training_file", type=str)
    parser.add_argument("--checkpoint_file", "-c", type=str, default=None)
    parser.add_argument("--start_batch", "-s", type=int, default=0)
    parser.add_argument("--name", "-n", type=str, default="variational_diffusion")
    parser.add_argument("--weights_file", "-w", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    train(**parse_args().__dict__)
