from dataclasses import dataclass

from omegaconf import OmegaConf

@dataclass(frozen=True)
class Config:
    # Basic network width options
    hidden_dim: int = 64
    timestep_dim: int = 32
    num_heads: int = 4
    expansion: int = 2

    # Basic network depth options
    num_detector_encoder_layers: int = 8
    num_parton_encoder_layers: int = 8
    num_parton_decoder_layers: int = 8
    num_denoising_layers: int = 8

    trivial_vae: bool = False
    conditional_vae: bool = False
    unconditional_vae_decoder: bool = False
    deterministic_vae: bool = False

    # Loss scales
    reconstruction_loss_scale: float = 1.0
    vae_prior_loss_scale: float = 0.0
    kl_loss_scale: float = 0.0
    mass_loss_scale: float = 0.0
    self_mass_loss_scale: float = 0.0

    # Data dimensions, set by dataset object.
    parton_dim: int = 55
    detector_dim: int = 9
    met_dim: int = 4

    # Training loop options
    seed: int = 0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    gradient_clipping: float = 1.0
    
    # Training loop iterations
    log_interval = 50
    save_interval = 1000
    num_batches: int = 100_000

    # VDM schedule
    noise_schedule: str = "cosine"
    noise_schedule_outputs: int = 1

    weighting: str = "sigmoid"
    sigmoid_weight_offset: float = 2.0

    initial_gamma_min = -13.3
    initial_gamma_max = 5.0

    normalize_parton: bool = False
    normalize_parton_scale: bool = False

    # Flow Options
    parton_flow_blocks: int = 24
    parton_flow_layers_per_block: int = 3
    parton_flow_units_per_layer: int = 256


    @classmethod
    def load(cls, filepath: str):
        return cls(**OmegaConf.load(filepath))
    
    
    def save(self, filepath: str):
        OmegaConf.save(
            OmegaConf.structured(self), 
            filepath
        )
