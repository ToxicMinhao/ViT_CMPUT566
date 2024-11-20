from typing import Any, Tuple

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class PositionalEmbedding(nn.Module):
    """This module adds a learnable positional embedding vector to each input position, which helps the model to understand the order of the sequence."""

    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """inputs.shape is (batch_size, sequence_length, embedding_dimension).
        batch_size: The number of samples (e.g., images) processed together in one forward pass through the model.
        sequence_length: The number of "tokens" (or elements) in each sample's sequence. In a Vision Transformer, this is the number of patches an image is divided into.
        Example: For a 256x256 image split into 16x16 patches, the sequence length is 256, then each of these 256 patches is transformed into a vector of size embedding_dimension.
        if embedding_dimension is 512, after the patch extraction and embedding step, you will have 256 vectors, each of size 512. Dimensional vector captures the characteristics (or features) of that specific patch."""

        assert inputs.ndim == 3, ('Number of dimensions should be 3,'  # dimension is (batch_size, sequence_length, embedding_dimension)
                                  ' but it is: %d' % inputs.ndim)
        positional_embedding_shape = (1, inputs.shape[1], inputs.shape[2])  # It guarantees that each image in the batch gets the same positional embedding for each position.

        # Positional embedding vectors can be updated during training to help the transformer understand the relative positions of each patch.
        positional_embedding = self.param('positional_embedding', nn.initializers.normal(stddev=0.02), positional_embedding_shape, self.dtype)
        return inputs + positional_embedding


class MLP_Block(nn.Module):
    """MLP Block in Transformer Encoder
    Dense → GELU → Dropout → Dense → Dropout"""

    MLP_dimension: int  # Number of neurons (or units) in the first dense layer of the MLP block
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'  # dimension is (batch_size, sequence_length, embedding_dimension)
                                  ' but it is: %d' % inputs.ndim)
        embedding_vector_dimension = inputs.shape[-1]

        x = nn.Dense(features=self.MLP_dimension, dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform())(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(features=embedding_vector_dimension, dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform())(x)
        return nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)


class Transformer_Encoder(nn.Module):
    """Transformer Encoder block containing self-attention and MLP sublayers."""
    MLP_dimension: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'  # dimension is (batch_size, sequence_length, embedding_dimension)
                                  ' but it is: %d' % inputs.ndim)
        
        # Self-Attention Block
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            dropout_rate=self.attention_dropout_rate,
            deterministic=deterministic
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs  # First residual connection

        # MLP Block
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MLP_Block(MLP_dimension=self.MLP_dimension, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)
        return x + y  # Second residual connection


class Encoder(nn.Module):
    num_layers: int
    MLP_dimension: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train):
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'  # dimension is (batch_size, sequence_length, embedding_dimension)
                                  ' but it is: %d' % inputs.ndim)
        
        # Add positional embeddings
        x = PositionalEmbedding(dtype=self.dtype)(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Stacking Transformer Encoder layers
        for i in range(self.num_layers):
            x = Transformer_Encoder(
                MLP_dimension=self.MLP_dimension,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
                name=f'Transformer_Encoder_{i}'
            )(x, deterministic=not train)
        
        return nn.LayerNorm(dtype=self.dtype)(x)


class VisionTransformer(nn.Module):
    """Vision Transformer without ResNet backbone."""
    hidden_size: int
    patches: Any  # Defines the size of the patches (e.g., 16x16)
    transformer: Any  # Configuration for the Transformer encoder
    num_classes: int  # Number of output classes for classification

    @nn.compact
    def __call__(self, inputs, *, train):  # Input will be a batch of images
        # Extract patches and project them to the hidden dimension
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding='VALID',
            name='patch_embedding')(inputs)
        
        # Reshape for Transformer input
        n, h, w, c = x.shape  # n: batch size, h: patches along height, w: patches along width, c: embedding dimension
        x = jnp.reshape(x, [n, h * w, c])  # Reshape to flatten 2D to 1D
        
        # Add a class token
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])  # Replicate token for each image in the batch
        x = jnp.concatenate([cls, x], axis=1)  # Concatenate token to patches

        # Apply Transformer encoder
        x = Encoder(
            num_layers=self.transformer['num_layers'],
            MLP_dimension=self.transformer['MLP_dimension'],
            num_heads=self.transformer['num_heads'],
            dropout_rate=self.transformer.get('dropout_rate', 0.1),
            attention_dropout_rate=self.transformer.get('dropout_rate_attention', 0.1),
            dtype=self.hidden_size
        )(x, train=train)

        # Use the class token output for classification
        x = x[:, 0]  # Extract the class token
        
        # Classification head
        x = nn.Dense(
            features=self.num_classes,
            name='head',
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )(x)
        
        return x
