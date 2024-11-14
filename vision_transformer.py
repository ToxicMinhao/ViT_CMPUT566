from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class Naminglayer(nn.Module):
    """In Jax, 'name' is a keyword that are used to assign a name to a specific instance; 
    Naminglayer(name = '__')(x)"""

    @nn.compact 
    def __call__(self, x):  
        return x

class PositionalEmbedding(nn.Module): 
    """This module adds a learnable positional embedding vector to each input position, which helps the model to understand the order of the sequence."""

    init_param_positional_embedding: Callable[[PRNGKey, Shape, Dtype], Array] # posemb_init should expect a function take [PRNGKey, Shape, Dtype] as input type and Array as output type.
    param_positional_embedding_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """inputs.shape is (batch_size, sequence_length, embedding_dimension).
        batch_size: The number of samples (e.g., images) processed together in one forward pass through the model.
        The number of "tokens" (or elements) in each sample's sequence. In a Vision Transformer, this is the number of patches an image is divided into.
        Example: For a 256x256 image split into 16x16 patches, the sequence length is 256, then for each of these 256 patches is transformed into a vector of size embedding_dimension.
        if embedding_dimension is 512, after the patch extraction and embedding step, you will have 256 vectors, each of size 512. Dimensional vector captures the characteristics (or features) of that specific patch"""

        assert inputs.ndim == 3, ('Number of dimensions should be 3,' # demension is (batch_size, sequence_length, embedding_dimension)
                                ' but it is: %d' % inputs.ndim)
        positional_embedding_shape = (1, inputs.shape[1], inputs.shape[2]) # it guarantees that each image in the batch gets the same positional embedding for each position (e.g if (image1, image2) in one batch, the positional embedding #0 will be apply to both iamge1 and image2).
        
        #positional_embedding_vectors can be updated during training,(e.g an image divided into 256 patches, and each patch is represented by a vector of length 512,each of these 256 patches will get a random positional embedding vector (like [0.1, -0.2, 0.05, ...]).)
        #During training, the values in these embedding vectors will be adjusted so that the transformer can better understand the relative positions of each patch.
        #For example, patches that are adjacent to each other might end up with similar positional embeddings, which helps the model learn relationships between them.
        postional_embedding_vectors = self.param('postional_embedding', self.init_param_positional_embedding, positional_embedding_shape, self.param_positional_embedding_dtype)
        return inputs + postional_embedding_vectors
    
class MLP_Encoder(nn.Module):
    """MLP Block in Transformer Encoder
    Dense → GELU → Dropout → Dense → Dropout"""

    MLP_dimension: int # Number of neurons (or units) in the first dense layer of the MLP block
    inference_dtype: Dtype = jnp.float32 # Used during the forward pass for computations, controls the precision of the calculations within the layer.
    param_dtype: Dtype = jnp.float32 # Specifies the precision for storing weights and biases (parameters).
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        embedding_vector_dimension = inputs.shape[-1] # Nov.13 update, force the output dimension equal to input embedding_dimension
        x = nn.Dense(features = self.MLP_dimension, dtype = self.inference_dtype, param_dtype = self.param_dtype, kernel_init = self.kernel_init, bias_init = self.bias_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout_rate)(x, deterministic = deterministic)
        output = nn.Dense(features = embedding_vector_dimension, dtype = self.inference_dtype, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        output = nn.Dropout(rate = self.dropout_rate)(output, deterministic = deterministic)
        return output