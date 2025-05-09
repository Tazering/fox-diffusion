from typing import Tuple, NamedTuple

import jax
from jax import Array
from jax import numpy as jnp

from flax import linen as nn

from lvd.config import NetworkConfig
from lvd.utils import masked_fill
from lvd.layers import (
      Embedding, 
      TransformerBlock,
      ContextVector
)


class DetectorEncoderOutputs(NamedTuple):
    summary: Array    # [B, D], 
    vectors: Array    # [B, 1 + T, D]
    mask: Array      # [B, 1 + T]


class DetectorEncoder(nn.Module):
    OutputType = DetectorEncoderOutputs # returns the class above - Tyler Kim

    # =============================================================================================
    # Options
    # region --------------------------------------------------------------------------------------
    config: NetworkConfig

    @property
    def embedding_config(self):
        return (
            self.config.hidden_dim, # int
            self.config.num_linear_layers, # int 
            self.config.transformer_expansion, # int
            self.config.dropout, # float
            self.config.skip_connection_type, # SkipConnectionType class
        )
    
    @property
    def context_vector_config(self):
        return (
            self.config.hidden_dim, # int
            self.config.ordered_detector_encoder # boolean
        )
    
    @property
    def transformer_config(self):
        return (
            2 * self.config.hidden_dim, # int
            self.config.transformer_heads, # int
            self.config.transformer_expansion, # int
            self.config.dropout, # float
            self.config.skip_connection_type # SkipConnectionType class
        )
    

    # getter for number of hidden dimensions
    @property
    def hidden_dim(self):
        return self.config.hidden_dim # int
    
    # getter for number of transformer layers
    @property
    def num_transformer_layers(self):
        return self.config.num_detector_encoder_layers # int
    
    # endregion ===================================================================================

    # =============================================================================================
    # Network Blocks
    # region --------------------------------------------------------------------------------------
    
    """
    These sets of functions seems to setup the types of embeddings by customizing the configuration
    of the neural network. The only difference between the different embeddings
    """
    
    @property
    def JetInputEmbedding(self):
        return Embedding(*self.embedding_config, name="jet_input_embedding")
    
    @property
    def EventInputEmbedding(self):
        return Embedding(*self.embedding_config, name="event_input_embedding")
    
    @property
    def JetContextVector(self):
        return ContextVector(*self.context_vector_config, name="jet_context_vector")
    
    @property
    def EventContextVector(self):
        return ContextVector(*self.context_vector_config, name="event_context_vector")
    
    @property
    def SummaryContextVector(self):
        return ContextVector(*self.context_vector_config, name="summary_context_vector")

    @property
    def JetOutput(self):
        return Embedding(*self.embedding_config, name="jet_output")

    @property
    def SummaryOutput(self):
        return Embedding(*self.embedding_config, name="summary_output")
    
    # endregion ===================================================================================

    # =============================================================================================
    # Network Parameters
    # region --------------------------------------------------------------------------------------
    @property
    def summary_embedding(self):
        return self.param(
            "summary_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.hidden_dim)
        )
    # endregion ===================================================================================

    # =============================================================================================
    # Network Functions
    # region --------------------------------------------------------------------------------------

    # B = jet of particle originating from a bottom quark
    # T = thrust
    # D = down quark or detector level?
    # C = charm quark?

    def create_jet_embeddings(
        self, 
        jet_features: Array,   # [B, T, D] 
        jet_mask: Array,       # [B, T],
        event_features: Array, # [B, D],
        *,
        training: bool = False
    ) -> Tuple[Array, Array]: # ([B, 1 + 1 + T, D], [B, 1 + 1 + T])
        B, C, D = jet_features.shape

        # Embedd the detector features into the initial latent space.
        # Figure 1: Conditional Set for the O_D values?
        jet_embeddings = self.JetInputEmbedding(jet_features, training = training)
        jet_embeddings = masked_fill(jet_embeddings, jet_mask)

        # Embedd the event features and add the extra time dimension.
        event_embeddings = self.EventInputEmbedding(event_features, training = training)

        # Expand event shape to allow for concatenation.
        event_mask = jnp.ones((B, 1), dtype=jet_mask.dtype)
        event_embeddings = jnp.expand_dims(event_embeddings, axis=1)

        # Construct an additional embedding vector to summarize event for jet multiplicity.
        summary_mask = jnp.ones((B, 1), dtype=jet_mask.dtype)
        summary_embeddings = jnp.broadcast_to(
            self.summary_embedding, 
            (B, 1, self.hidden_dim)
        )

        # Add context vectors to differentiate the inputs.
        jet_embeddings = self.JetContextVector(jet_embeddings, training = training)
        event_embeddings = self.EventContextVector(event_embeddings, training = training)
        summary_embeddings = self.SummaryContextVector(summary_embeddings, training = training)
        
        # Create a combined detector list with the summary, event, and jet features.
        jet_embeddings = jnp.concatenate(axis=1, arrays=(
            summary_embeddings,
            event_embeddings, 
            jet_embeddings
        ))

        jet_mask = jnp.concatenate(axis=1, arrays=(
            summary_mask,
            event_mask,
            jet_mask
        ))

        jet_embeddings = masked_fill(jet_embeddings, jet_mask)

        return jet_embeddings, jet_mask
    # endregion ===================================================================================

    @nn.compact
    def __call__(
        self, 
        jet_features: Array,   # [B, T, D]
        jet_masks: Array,      # [B, T],
        event_features: Array, # [B, D],
        *,
        training: bool = False,
    ) -> DetectorEncoderOutputs:
        jet_embeddings, jet_masks = self.create_jet_embeddings(
            jet_features, 
            jet_masks, 
            event_features,
            training = training
        )

        # Transformer Encoders to contextualize embeddings.
        for _ in range(self.num_transformer_layers):
            jet_embeddings = TransformerBlock(*self.transformer_config)(
                jet_embeddings, 
                jet_masks,
                training = training
            )

        # Split the two outputs and apply their per-vector transforms.
        summary_embeddings, jet_embeddings = jet_embeddings[:, 0], jet_embeddings[:, 1:]
        summary_mask, jet_masks = jet_masks[:, 0], jet_masks[:, 1:]

        summary_embeddings = self.SummaryOutput(summary_embeddings, training = training)

        jet_embeddings = self.JetOutput(jet_embeddings, training = training)
        jet_embeddings = masked_fill(jet_embeddings, jet_masks)

        return DetectorEncoderOutputs(summary_embeddings, jet_embeddings, jet_masks)