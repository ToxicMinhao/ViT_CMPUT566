import os
import time
from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.training import checkpoints
from vit_jax import models, input_pipeline, utils

def create_update_fn(apply_fn, optimizer, accum_steps):
    """Creates an update function for training."""
    def update_fn(params, opt_state, batch, rng):
        dropout_rng, new_rng = jax.random.split(rng)

        def compute_loss(params, images, labels):
            logits = apply_fn({'params': params}, images, rngs={'dropout': dropout_rng}, train=True)
            log_softmax = jax.nn.log_softmax(logits)
            loss = -jnp.sum(log_softmax * labels) / labels.shape[0]
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(params, batch['image'], batch['label'])
        grads = jax.lax.pmean(grads, axis_name='batch')
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss = jax.lax.pmean(loss, axis_name='batch')
        return params, opt_state, loss, new_rng

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))

def train_and_evaluate(config, workdir):
    """Main training and evaluation loop."""
    logging.info("Starting training with config:\n%s", config)

    # Load dataset
    train_dataset, test_dataset = input_pipeline.get_datasets(config)
    dataset_info = input_pipeline.get_dataset_info(config.dataset, 'train')

    # Initialize model
    model_cls = models.VisionTransformer
    model = model_cls(num_classes=dataset_info['num_classes'], **config.model)

    def init_model():
        example_image = jnp.ones((1,) + tuple(config.model.image_shape))
        return model.init(jax.random.PRNGKey(0), example_image, train=False)

    variables = jax.jit(init_model, backend='cpu')()
    params = variables['params']

    # Load pretrained weights (if available)
    pretrained_path = os.path.join(config.pretrained_dir, f"{config.model.model_name}.npz")
    if tf.io.gfile.exists(pretrained_path):
        params = models.load_pretrained_weights(pretrained_path, params, config.model)
        logging.info("Loaded pretrained weights from %s", pretrained_path)

    # Set up optimizer and learning rate schedule
    lr_schedule = optax.cosine_decay_schedule(config.base_lr, config.total_steps)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    opt_state = optimizer.init(params)

    update_fn = create_update_fn(model.apply, optimizer, config.accum_steps)

    # Prepare datasets
    train_batches = input_pipeline.prefetch(train_dataset, config.prefetch)
    test_batches = input_pipeline.prefetch(test_dataset, config.prefetch)

    # Training loop
    params_repl = flax.jax_utils.replicate(params)
    opt_state_repl = flax.jax_utils.replicate(opt_state)
    rng = jax.random.PRNGKey(config.seed)

    for step, batch in enumerate(train_batches, start=1):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            params_repl, opt_state_repl, loss_repl, rng = update_fn(params_repl, opt_state_repl, batch, rng)

        if step % config.log_every == 0:
            loss = flax.jax_utils.unreplicate(loss_repl)
            logging.info(f"Step {step}: Training loss = {loss:.4f}")

        if step % config.eval_every == 0:
            logging.info("Evaluating model...")
            accuracies = []
            for test_batch in test_batches:
                logits = model.apply({'params': flax.jax_utils.unreplicate(params_repl)}, test_batch['image'], train=False)
                predictions = jnp.argmax(logits, axis=-1)
                labels = jnp.argmax(test_batch['label'], axis=-1)
                accuracies.append((predictions == labels).mean())

            accuracy = jnp.mean(jnp.array(accuracies))
            logging.info(f"Step {step}: Test accuracy = {accuracy:.4f}")

        if step % config.save_every == 0 or step == config.total_steps:
            checkpoints.save_checkpoint(workdir, flax.jax_utils.unreplicate(params_repl), step)
            logging.info("Saved checkpoint at step %d", step)

    return flax.jax_utils.unreplicate(params_repl)

if __name__ == "__main__":
    # Define your configuration dictionary here
    config = {
        "model": {
            "model_name": "ViT-B_16",
            "num_layers": 12,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "image_shape": (224, 224, 3),
        },
        "dataset": "imagenet",
        "pretrained_dir": "./pretrained",
        "total_steps": 1000,
        "base_lr": 0.001,
        "weight_decay": 0.1,
        "accum_steps": 1,
        "seed": 0,
        "log_every": 50,
        "eval_every": 100,
        "save_every": 500,
        "prefetch": 2,
    }

    workdir = "./workdir"
    train_and_evaluate(config, workdir)
