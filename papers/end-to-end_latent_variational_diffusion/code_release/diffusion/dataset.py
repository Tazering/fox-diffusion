from typing import NamedTuple

import numpy as np

import jax
from jax import Array
from flax.jax_utils import prefetch_to_device

import tensorflow as tf


class Batch(NamedTuple):
    parton_features: Array
    detector_features: Array
    detector_mask: Array
    met_features: Array
    reco_targets: Array
    weights: Array


class Dataset:
    def __init__(self, filepath: str, split: float = 1.0, weights_file: str = None) -> None:
        with np.load(filepath) as file:
            self.parton_features = file["parton_features"]            
            self.detector_features = file["detector_features"]
            self.detector_mask = file["detector_mask"]
            self.met_features = file["met_features"]
            self.reco_targets = file["targets"]

        num_events = self.parton_features.shape[0]

        if weights_file is None:
            self.weights = np.ones(num_events)
        else:
            print(f"Loading Weights: {weights_file}")
            self.weights = np.load(weights_file)

        if split > 0:
            split = int(round(num_events * split))
            self.parton_features = self.parton_features[:split]
            self.met_features = self.met_features[:split]
            self.detector_features = self.detector_features[:split]
            self.detector_mask = self.detector_mask[:split]
            self.reco_targets = self.reco_targets[:split]
            self.weights = self.weights[:split]
        else:
            split = int(round(num_events * (split + 1)))
            self.parton_features = self.parton_features[split:]
            self.met_features = self.met_features[split:]
            self.detector_features = self.detector_features[split:]
            self.detector_mask = self.detector_mask[split:]
            self.reco_targets = self.reco_targets[split:]
            self.weights = self.weights[split:]

        self.num_events = self.parton_features.shape[0]
        self.parton_features = self.parton_features.reshape((self.num_events, -1))

        self.parton_dim = self.parton_features.shape[-1]
        self.detector_dim = self.detector_features.shape[-1]
        self.met_dim = self.met_features.shape[-1]

    @property
    def parton_statistics(self):
        mean = self.parton_features.mean(0)
        std = self.parton_features.std(0)
        std[std < 1e-6] = 1.0
        return mean, std
    
    @property
    def met_statistics(self):
        mean = self.met_features.mean(0)
        std = self.met_features.std(0)
        std[std < 1e-6] = 1.0
        return mean, std
    
    @property
    def detector_statistics(self):
        masked_detector = self.detector_features[self.detector_mask]
        mean = masked_detector.mean(0)
        std = masked_detector.std(0)
        std[std < 1e-6] = 1.0
        return mean, std
    
    @property
    def square_mass_statistics(self):
        try:
            square_masses = self.parton_features.reshape(-1, 11, 5)
            square_masses = square_masses[:, 6:, 4]
            square_masses = square_masses ** 2

            mean = square_masses.mean(0)
            std = square_masses.std(0)
            std[std < 1e-6] = 1.0
            return mean, std
        except:
            return 0, 0
    
    @property
    def statistics(self):
        statistics = (
            *self.parton_statistics, 
            *self.detector_statistics, 
            *self.met_statistics,
            *self.square_mass_statistics
        )

        names = (
            "parton_mean",
            "parton_std",
            
            "detector_mean",
            "detector_std",

            "met_mean",
            "met_std",

            "square_mass_mean",
            "square_mass_std"
        )

        return dict(zip(names, statistics))

    def create_dataloader(
            self, 
            batch_size: int = 1024, 
            num_devices: int = None, 
            num_epochs: int = None,
            shuffle: bool = True
        ):

        if num_devices is None:
            num_devices = jax.local_device_count()

        devices = jax.local_devices()[:num_devices]
            
        def split_batch(x):
            return tf.reshape(x, (num_devices, batch_size, *x.shape[1:]))
            
        def split_batches(*x):
            return tuple(map(split_batch, x))
        
        with tf.device("CPU"):
            dataset = tf.data.Dataset.from_tensor_slices((
                self.parton_features,
                self.detector_features,
                self.detector_mask,
                self.met_features,
                self.reco_targets,
                self.weights
            ))

            if shuffle:
                dataset = dataset.shuffle(2 * batch_size * num_devices)
                
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size * num_devices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(split_batches, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(Batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return prefetch_to_device(dataset.as_numpy_iterator(), 3, devices=devices)
    
    def create_eval_dataloader(
            self, 
            batch_size: int = 1024, 
            num_devices: int = None, 
        ):

        if num_devices is None:
            num_devices = jax.local_device_count()
            
        with tf.device("CPU"):
            dataset = tf.data.Dataset.from_tensor_slices((
                self.parton_features,
                self.detector_features,
                self.detector_mask,
                self.met_features,
                self.reco_targets,
                self.weights
            ))
            
            dataset = dataset.batch(batch_size * num_devices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(Batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset.as_numpy_iterator()