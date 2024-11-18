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