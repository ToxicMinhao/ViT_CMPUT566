import logging as python_logging
import os
import threading

from absl import logging
import jax
import jax.numpy as jnp
import tensorflow as tf


class GFileHandler(python_logging.StreamHandler):
    """Writes log messages to file using tf.io.gfile."""

    def __init__(self, filename, mode, flush_secs=1.0):
        super().__init__()
        tf.io.gfile.makedirs(os.path.dirname(filename))
        if mode == 'a' and not tf.io.gfile.exists(filename):
            mode = 'w'
        self.filehandle = tf.io.gfile.GFile(filename, mode)
        self.flush_secs = flush_secs
        self.flush_timer = None

    def flush(self):
        self.filehandle.flush()

    def emit(self, record):
        msg = self.format(record)
        self.filehandle.write(f'{msg}\n')
        if self.flush_timer is not None:
            self.flush_timer.cancel()
        self.flush_timer = threading.Timer(self.flush_secs, self.flush)
        self.flush_timer.start()


def create_learning_rate_schedule(total_steps, base_lr, decay_type='cosine', warmup_steps=0):
    """
    Creates a learning rate schedule based on the configuration.
    
    Args:
        total_steps: Total number of training steps.
        base_lr: Base learning rate.
        decay_type: Type of learning rate decay ('cosine', 'linear', or 'constant').
        warmup_steps: Number of warmup steps at the beginning of training.

    Returns:
        A function that takes a step number and returns the learning rate.
    """
    def learning_rate_fn(step):
        if warmup_steps > 0:
            warmup_ratio = jnp.minimum(1.0, step / warmup_steps)
        else:
            warmup_ratio = 1.0

        if decay_type == 'cosine':
            decay_ratio = 0.5 * (1 + jnp.cos(jnp.pi * step / total_steps))
        elif decay_type == 'linear':
            decay_ratio = 1 - step / total_steps
        else:  # constant
            decay_ratio = 1.0

        return base_lr * warmup_ratio * decay_ratio

    return learning_rate_fn


def accumulate_gradient(grad_fn, params, images, labels, accum_steps):
    """
    Accumulates gradients across multiple steps for gradient accumulation.

    Args:
        grad_fn: Gradient computation function.
        params: Model parameters.
        images: Input images.
        labels: Ground-truth labels.
        accum_steps: Number of steps to accumulate gradients.

    Returns:
        Accumulated loss and gradient.
    """
    def accumulate_step(carry, batch):
        grad_accum, loss_accum = carry
        loss, grads = grad_fn(params, batch['image'], batch['label'])
        grad_accum = jax.tree_map(lambda x, y: x + y, grad_accum, grads)
        loss_accum += loss
        return (grad_accum, loss_accum), None

    # Initialize gradient accumulator and loss accumulator
    initial_grads = jax.tree_map(jnp.zeros_like, params)
    initial_loss = 0.0
    carry = (initial_grads, initial_loss)

    (final_grads, final_loss), _ = jax.lax.scan(
        accumulate_step,
        carry,
        xs={'image': jnp.array_split(images, accum_steps),
            'label': jnp.array_split(labels, accum_steps)}
    )

    # Average gradients and loss
    final_grads = jax.tree_map(lambda x: x / accum_steps, final_grads)
    final_loss /= accum_steps
    return final_loss, final_grads

def save_checkpoint(directory, params, step, prefix='ckpt_'):
    """
    Saves model parameters as a checkpoint.

    Args:
        directory: Directory where the checkpoint will be saved.
        params: Model parameters to save.
        step: Training step number.
        prefix: Prefix for the checkpoint file name.
    """
    checkpoint_path = f"{directory}/{prefix}{step}.npz"
    with open(checkpoint_path, 'wb') as f:
        np.savez(f, **jax.tree_map(lambda x: np.asarray(x), params))
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, params):
    """
    Loads model parameters from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        params: Current model parameters (used to match the structure).

    Returns:
        Loaded parameters.
    """
    with open(checkpoint_path, 'rb') as f:
        loaded = np.load(f)
        loaded_params = {key: loaded[key] for key in loaded.files}

    def load_fn(current, checkpointed):
        return checkpointed if checkpointed.shape == current.shape else current

    return jax.tree_map(load_fn, params, loaded_params)

