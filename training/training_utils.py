import os
import tensorflow as tf
import ml_collections
import datetime
import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from typing import Iterator, Any, Dict, Tuple
from collections.abc import Callable
from dataclasses import dataclass
import functools


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    enc_input: jax.Array
    enc_input_mask: jax.Array
    dec_input: jax.Array
    dec_input_mask: jax.Array
    labels: jax.Array


# Metric type that help to track the sum and the count for each metrics (loss or accuracy)
@jax.tree_util.register_dataclass
@dataclass
class Metric:
    sum: jnp.float32
    count: jnp.int32

    def get_metric_val(self) -> jnp.float32:
        return jnp.divide(self.sum, self.count)


Metrics = Dict[str, Metric]
PyTree = Any


class TrainState(train_state.TrainState):
    dropout_key: jax.Array


def generate_random_batch(init_prng: jax.Array,
                          batch_size: int,
                          max_seq_len: int,
                          vocab_size: int) -> Batch:
    prng_1, prng_2 = jax.random.split(init_prng, 2)
    enc_input = jax.random.randint(key=prng_1, shape=(batch_size, max_seq_len),
                                   dtype=jnp.int64, minval=0, maxval=vocab_size)
    dec_input_raw = jax.random.randint(key=prng_2, shape=(batch_size, max_seq_len+1),
                                   dtype=jnp.int64, minval=0, maxval=vocab_size)
    dec_input = dec_input_raw[:, :-1]
    labels = dec_input_raw[:, 1:]
    enc_input_mask = jnp.full(shape=(batch_size, max_seq_len), fill_value=1, dtype=jnp.int64)
    dec_input_mask = jnp.full(shape=(batch_size, max_seq_len), fill_value=1, dtype=jnp.int64)
    return Batch(enc_input, enc_input_mask, dec_input, dec_input_mask, labels)


def create_learning_rate_scheduler(base_lr: jnp.float32,
                                   warmup_epochs: jnp.float32,
                                   cosine_epochs: jnp.float32,
                                   steps_per_epochs: jnp.int32) -> Any:
    warmup_fn = optax.linear_schedule(init_value=0, end_value=base_lr,
                                      transition_steps=warmup_epochs * steps_per_epochs)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_lr,
                                            decay_steps=cosine_epochs * steps_per_epochs)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
                                       boundaries=[warmup_epochs * steps_per_epochs])
    return schedule_fn


# Generator for the training dataset
def get_dataset_iterator(dataset: tf.data.Dataset,
                         batch_size: int,
                         is_infinite: bool = False) -> Iterator[Batch]:

    ds = dataset.repeat() if is_infinite else dataset
    ds = ds.batch(batch_size=batch_size,
                                      drop_remainder=True,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    for sample in ds:
        yield Batch(enc_input=jnp.array(sample["de_input"].numpy()),
                    enc_input_mask=jnp.array(sample["de_input_mask"].numpy()),
                    dec_input=jnp.array(sample["en_input"].numpy()),
                    dec_input_mask=jnp.array(sample["en_input_mask"].numpy()),
                    labels=jnp.array(sample["en_output"].numpy()))


def merge_metrics(metric1: Metric, metric2: Metric) -> Metric:
    """Merge two metrics and return a new one."""
    return jax.tree.map(jnp.add, metric1, metric2)


def get_accuracy_metric(logits: jax.Array,
                     labels: jax.Array,
                     mask: jax.Array = None) -> Metric:
    """Given the logits, the labels and optionally the mask, this function
    compute the accuracy. The mask is used to discard the non eligible elements.

    Keyword arguments:
    logits -- Shape (batch_size, max_seq_len, vocab_size)
    labels -- Shape (batch_size, max_seq_len)
    mask -- Shape (batch_size, max_seq_len)

    Returns the accuracy metric.
    """
    predicted_label = jnp.argmax(logits, axis=-1)
    if mask is None:
        mask = jnp.ones_like(labels)
    predicted_label = jnp.where(mask == 0, 0, predicted_label)
    filtered_labels = jnp.where(mask == 0, -1, labels)
    matched_count = jnp.equal(predicted_label, filtered_labels)
    sum = jnp.sum(matched_count)
    count = jnp.sum(mask)
    return Metric(sum, count)


def compute_masked_loss_and_accuracy(params: PyTree,
                      apply_fn: Any,
                      batch: Batch,
                      training: bool,
                      dropout_rng_key: jax.Array | None) -> Tuple[PyTree, Metrics]:
    logits = apply_fn(
        {'params': params},
        enc_x=batch.enc_input,
        dec_x=batch.dec_input,
        enc_mask=batch.enc_input_mask,
        dec_mask=batch.dec_input_mask,
        training=training,
        rngs={'dropout': dropout_rng_key} if training else None,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                           labels=batch.labels)
    loss = loss * batch.dec_input_mask
    loss_val = jnp.divide(jnp.sum(loss), jnp.sum(batch.dec_input_mask))

    metrics = {
        "loss": Metric(jnp.sum(loss), jnp.sum(batch.dec_input_mask)),
        "acc": get_accuracy_metric(logits, batch.labels, batch.dec_input_mask)
        }
    return loss_val, metrics


@functools.partial(
    jax.jit,
    donate_argnames=(
        "state",
        "metrics"
    )
)
def train_step(state: TrainState,
               metrics: Metrics | None,
               batch: Batch) -> Tuple[TrainState, Metrics]:
    # Call the model to get the logits
    dropout_rng_key = jax.random.fold_in(key=state.dropout_key,
                                             data=state.step)
    grad_fn = jax.value_and_grad(compute_masked_loss_and_accuracy, has_aux=True)
    (loss_val, train_step_metrics), grads = grad_fn(state.params,
                                                    state.apply_fn,
                                                    batch,
                                                    training=True,
                                                    dropout_rng_key=dropout_rng_key,)
    new_state = state.apply_gradients(grads=grads)
    if metrics is None:
        new_metrics = train_step_metrics
    else:
        new_metrics = merge_metrics(metrics, train_step_metrics)

    return new_state, new_metrics


@functools.partial(
    jax.jit,
    donate_argnames=(
        "metrics"
    )
)
def eval_step(state: TrainState,
              metrics: Metrics | None,
              batch: Batch) -> Metrics:
    _, val_step_metrics = compute_masked_loss_and_accuracy(state.params,
                                                            state.apply_fn,
                                                            batch, 
                                                            training=False, 
                                                            dropout_rng_key=None)
    if metrics is None:
        new_metrics = val_step_metrics
    else:
        new_metrics = merge_metrics(metrics, val_step_metrics)
    return new_metrics



def create_train_state(model: nn.Module, config: ml_collections.ConfigDict, init_prng_key: jax.Array):
    param_init_key_1, param_init_key_2, param_init_key_3, dropout_key = jax.random.split(init_prng_key, 4)
    sample_enc_x = jax.random.choice(param_init_key_1, config.data.vocab_size, (1, config.data.max_seq_len))
    sample_dec_x = jax.random.choice(param_init_key_2, config.data.vocab_size, (1, config.data.max_seq_len))
    variables = model.init(param_init_key_3, sample_enc_x, sample_dec_x)

    # Create the optimizer
    cosine_epochs = config.optimizer.training_epochs - config.optimizer.warmup_epochs
    scheduler = create_learning_rate_scheduler(config.optimizer.base_lr,
                                               config.optimizer.warmup_epochs,
                                               cosine_epochs,
                                               config.optimizer.steps_per_epochs)

    return TrainState.create(
        apply_fn = model.apply,
        params = variables['params'],
        tx = optax.adam(scheduler),
        dropout_key = dropout_key,
    )


def train_and_evaluate(model: nn.Module, 
                       config: ml_collections.ConfigDict,
                       init_prng: jax.Array,
                       train_ds: tf.data.Dataset,
                       validation_ds: tf.data.Dataset,
                       log_dir_prefix: str | None) -> TrainState:
    train_ds_iterator = get_dataset_iterator(train_ds,
                                             config.data.batch_size,
                                             is_infinite=True)
    train_state = create_train_state(model, config, init_prng)

    # Create the checkpoint manager
    ckp_options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                               best_fn=lambda metrics: metrics['acc'])
    ckp_mngr = ocp.CheckpointManager(os.path.abspath(config.training_output.checkpoint_path),
                                     options=ckp_options)

    # Create tf.summary
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix_path = current_time if log_dir_prefix is None else log_dir_prefix
    train_log_dir = config.training_output.metric_path + '/' + prefix_path + '/train'
    val_log_dir = config.training_output.metric_path + '/' + prefix_path + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Get metric shape
    sample_batch = generate_random_batch(jax.random.key(1),
                                     config.data.batch_size,
                                     config.data.max_seq_len,
                                     config.data.vocab_size)
    _, train_metrics_shapes = jax.eval_shape(train_step, train_state, None, sample_batch)
    val_metrics_shapes = jax.eval_shape(eval_step, train_state, None, sample_batch)

    for epoch in range(config.optimizer.training_epochs):
        train_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), train_metrics_shapes)
        val_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), val_metrics_shapes)
        validation_ds_iterator = get_dataset_iterator(validation_ds,
                                                      config.data.batch_size,
                                                      is_infinite=False)

        # logging.info(f"Epoch {epoch + 1}")
        print(f"Epoch {epoch + 1}")
        for _ in tqdm(range(config.optimizer.steps_per_epochs)):
            train_state, train_metrics = train_step(train_state,
                                                    train_metrics,
                                                    next(train_ds_iterator))

        for val_batch in validation_ds_iterator:
            val_metrics = eval_step(train_state, val_metrics, val_batch)

        train_final_loss = float(train_metrics['loss'].get_metric_val())
        train_final_accuracy = float(train_metrics['acc'].get_metric_val())

        val_final_loss = float(val_metrics['loss'].get_metric_val())
        val_final_accuracy = float(val_metrics['acc'].get_metric_val())

        ckp_mngr.save(epoch,
                      args=ocp.args.StandardSave(train_state),
                      metrics={
                          'acc': val_final_accuracy,
                          'loss': val_final_loss,
                          }
                      )

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_final_loss, step=epoch)
            tf.summary.scalar('accuracy', train_final_accuracy, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_final_loss, step=epoch)
            tf.summary.scalar('accuracy', val_final_accuracy, step=epoch)

        # logging.info(f"Training:    Loss: {train_final_loss}    Accuracy: {train_final_accuracy}")
        # logging.info(f"Validation:  Loss: {val_final_loss}    Accuracy: {val_final_accuracy}")
        print(f"Training:    Loss: {train_final_loss}    Accuracy: {train_final_accuracy}")
        print(f"Validation:  Loss: {val_final_loss}    Accuracy: {val_final_accuracy}")

    ckp_mngr.wait_until_finished()
    return train_state
