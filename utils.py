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