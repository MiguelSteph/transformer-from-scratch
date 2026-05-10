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
from typing import Iterator, Any, Dict, Tuple, Sequence
from collections.abc import Callable
from dataclasses import dataclass
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import functools
from tqdm.auto import tqdm


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    enc_input: jax.Array
    dec_input: jax.Array
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
Parameter = jax.Array | nn.Partitioned


class TrainState(train_state.TrainState):
    dropout_key: jax.Array
    dec_pad_id: int


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    # From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


def sync_gradients(
    grads: PyTree,
    axis_names: Sequence[str],
) -> PyTree:
    # From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html
    """Synchronize gradients across devices.

    Gradients for parameters that are replicated over a given axis are averaged across devices.
    Parameters that are partitioned over a given axis are considered to already have a mean of
    the gradients on each device, and hence do not need to be altered.

    Args:
        grads: The gradients to synchronize.
        axis_names: The axis names to synchronize gradients across.

    Returns:
        The gradients averaged over the specified axes if they are replicated.
    """

    def sync_grad(g: Parameter) -> Parameter:
        if isinstance(g, nn.Partitioned):
            # Tree leaves for flattening potentially nested axis (multiple names can exist for single array axis).
            replication_axis_names = [
                name for name in axis_names if name not in jax.tree_util.tree_leaves(g.names)
            ]
            if len(replication_axis_names) == 0:
                # Parameters partitioned over all axes.
                return g
            else:
                # Average over remaining replicated axes.
                return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))
        else:
            # Parameters are replicated over all axes.
            return jax.lax.pmean(g, axis_name=axis_names)

    return jax.tree_map(sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned))


def generate_random_batch(init_prng: jax.Array,
                          batch_size: int,
                          max_seq_len: int,
                          vocab_size: int) -> Batch:
    prng_1, prng_2 = jax.random.split(init_prng, 2)
    enc_input = jax.random.randint(key=prng_1, shape=(batch_size, max_seq_len),
                                   dtype=jnp.int32, minval=0, maxval=vocab_size)
    dec_input_raw = jax.random.randint(key=prng_2, shape=(batch_size, max_seq_len+1),
                                   dtype=jnp.int32, minval=0, maxval=vocab_size)
    dec_input = dec_input_raw[:, :-1]
    labels = dec_input_raw[:, 1:]
    return Batch(enc_input, dec_input, labels)


def create_learning_rate_scheduler(base_lr: jnp.float32,
                                   warmup_epochs: jnp.float32,
                                   cosine_epochs: jnp.float32,
                                   steps_per_epochs: jnp.int32) -> Any:
    warmup_fn = optax.linear_schedule(init_value=0, end_value=base_lr,
                                      transition_steps=warmup_epochs * steps_per_epochs)
    # cosine_fn = optax.cosine_decay_schedule(init_value=base_lr,
    #                                         decay_steps=cosine_epochs * steps_per_epochs)
    # schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
    #                                    boundaries=[warmup_epochs * steps_per_epochs])
    constant_fn = optax.constant_schedule(value=base_lr)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, constant_fn],
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
                    dec_input=jnp.array(sample["en_input"].numpy()),
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
                      dropout_rng_key: jax.Array | None,
                      dec_pad_id: int,
                      label_smoothing = 0.1) -> Tuple[PyTree, Metrics]:
    with jax.named_scope("computing_logits"):
        logits = apply_fn(
            {'params': params},
            enc_x=batch.enc_input,
            dec_x=batch.dec_input,
            training=training,
            rngs={'dropout': dropout_rng_key} if training else None,
        )

    with jax.named_scope("computing_loss"):
        dec_input_mask = (batch.dec_input != dec_pad_id).astype(int)
        vocab_size = logits.shape[-1]
        one_hot = jax.nn.one_hot(batch.labels, vocab_size)
        soft_labels = (1.0 - label_smoothing) * one_hot + label_smoothing / vocab_size
        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss * dec_input_mask
        loss_val = jnp.divide(jnp.sum(loss), jnp.sum(dec_input_mask))

    with jax.named_scope("computing_metrics"):
        metrics = {
            "loss": Metric(jnp.sum(loss), jnp.sum(dec_input_mask)),
            "acc": get_accuracy_metric(logits, batch.labels, dec_input_mask)
            }
    return loss_val, metrics


def train_step_fsdp(state: TrainState, 
                    metrics: Metrics | None, 
                    batch: Batch, 
                    data_axis_name: str) -> Tuple[TrainState, Metrics]:
    # Call the model to get the logits
    dropout_rng_key = jax.random.fold_in(key=state.dropout_key, 
                                         data=state.step)
    device_specific_dropout_rng = fold_rng_over_axis(dropout_rng_key, 
                                                     data_axis_name)
    grad_fn = jax.value_and_grad(compute_masked_loss_and_accuracy, has_aux=True)
    (loss_val, train_step_metrics), grads = grad_fn(state.params,
                                                    state.apply_fn,
                                                    batch,
                                                    training=True,
                                                    dropout_rng_key=device_specific_dropout_rng,
                                                    dec_pad_id=state.dec_pad_id,)
    with jax.named_scope("sync_gradients"):
        grads = sync_gradients(grads, (data_axis_name,))
    new_state = state.apply_gradients(grads=grads)

    with jax.named_scope("sync_metrics"):
        train_step_metrics = jax.tree_map(
            lambda x: jax.lax.psum(x, axis_name=data_axis_name), train_step_metrics
        )

    if metrics is None:
        new_metrics = train_step_metrics
    else:
        new_metrics = merge_metrics(metrics, train_step_metrics)

    return new_state, new_metrics


def eval_step_fsdp(params: PyTree, 
                   metrics: Metrics | None,
                   batch: Batch,
                   apply_fn: Any,
                   data_axis_name: str,
                   dec_pad_id: int) -> Metrics:
    _, val_step_metrics = compute_masked_loss_and_accuracy(params, 
                                                           apply_fn,
                                                           batch, 
                                                           training=False, 
                                                           dropout_rng_key=None,
                                                           dec_pad_id=dec_pad_id,
                                                           label_smoothing = 0.0)
    with jax.named_scope("sync_metrics"):
        val_step_metrics = jax.tree_map(
            lambda x: jax.lax.psum(x, axis_name=data_axis_name), val_step_metrics
        )
    
    if metrics is None:
        new_metrics = val_step_metrics
    else:
        new_metrics = merge_metrics(metrics, val_step_metrics)
    return new_metrics


def init_model(init_prng_key: jax.Array,
              sample_enc_x: jax.Array,
              sample_dec_x: jax.Array,
              model_fspd: nn.Module,
              optimizer: Any,
              dec_pad_id: int):
    model_init_key, dropout_key = jax.random.split(init_prng_key, 2)
    variables = model_fspd.init({"params": model_init_key}, sample_enc_x, sample_dec_x)
    params = variables.pop("params")

    return TrainState.create(
        apply_fn = model_fspd.apply,
        params = params,
        tx = optimizer,
        dropout_key = dropout_key,
        dec_pad_id = dec_pad_id,
    )


def _create_summary_writers(config: ml_collections.ConfigDict, 
                         log_dir_prefix: str | None):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix_path = current_time if log_dir_prefix is None else log_dir_prefix
    train_log_dir = config.training_output.metric_path + '/' + prefix_path + '/train'
    val_log_dir = config.training_output.metric_path + '/' + prefix_path + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)
    return train_writer, val_writer


def get_train_step_fsdp_fn(mesh, train_state_fsdp_specs, data_axis_name):
    train_step_fsdp_fn = jax.jit(
        jax.shard_map(
            functools.partial(
                train_step_fsdp,
                data_axis_name=data_axis_name
            ),
            mesh,
            in_specs=(train_state_fsdp_specs, P(), P(data_axis_name)),
            out_specs=(train_state_fsdp_specs, P())
        ),
        donate_argnames=("state", "metrics"),
    )
    return train_step_fsdp_fn


def get_eval_step_fsdp_fn(mesh, params_fsdp_specs, apply_fn, data_axis_name, dec_pad_id):
    eval_step_fsdp_fn = jax.jit(
        jax.shard_map(
            functools.partial(
                eval_step_fsdp,
                apply_fn=apply_fn,
                data_axis_name=data_axis_name,
                dec_pad_id=dec_pad_id,
            ),
            mesh,
            in_specs=(params_fsdp_specs, P(), P(data_axis_name)),
            out_specs=P()
        ),
        donate_argnames=("metrics"),
    )
    return eval_step_fsdp_fn


def fsdp_init(model: nn.Module, mesh: Mesh, config: ml_collections.ConfigDict, dec_pad_id: int):
    sample_batch = generate_random_batch(jax.random.key(10),
                                        config.data.batch_size,
                                        config.data.max_seq_len,
                                        config.data.vocab_size)
    cosine_epochs = config.optimizer.training_epochs - config.optimizer.warmup_epochs
    scheduler_fn = create_learning_rate_scheduler(config.optimizer.base_lr,
                                                config.optimizer.warmup_epochs,
                                                cosine_epochs,
                                                config.optimizer.steps_per_epochs)
    optimizer = optax.adam(scheduler_fn)

    init_partial_fn = functools.partial(init_model, 
                            model_fspd=model, 
                            optimizer=optimizer,
                            dec_pad_id=dec_pad_id)
    init_model_fn_to_eval_shape = jax.jit(jax.shard_map(
            init_partial_fn,
            mesh=mesh,
            in_specs=(P(), P(config.fsdp.data_axis), P(config.fsdp.data_axis)),
            out_specs=P(),
            check_vma=False
        ))

    init_rng = jax.random.key(12)
    state_fsdp_shapes = jax.eval_shape(init_model_fn_to_eval_shape, 
                                    init_rng, 
                                    sample_batch.enc_input, 
                                    sample_batch.dec_input)
    state_fsdp_specs = nn.get_partition_spec(state_fsdp_shapes)

    init_model_fn = jax.jit(jax.shard_map(
            init_partial_fn,
            mesh=mesh,
            in_specs=(P(), P(config.fsdp.data_axis), P(config.fsdp.data_axis)),
            out_specs=state_fsdp_specs,
        ))

    train_state = init_model_fn(init_rng, sample_batch.enc_input, sample_batch.dec_input)
    return train_state, state_fsdp_specs


def train_and_evaluate(model: nn.Module, 
                       mesh: Mesh,
                       train_state: TrainState,
                       train_state_fsdp_specs: P,
                       config: ml_collections.ConfigDict,
                       init_prng: jax.Array,
                       train_ds: tf.data.Dataset,
                       validation_ds: tf.data.Dataset,
                       log_dir_prefix: str | None,
                       dec_pad_id: int) -> TrainState:
    train_step_fsdp_fn = get_train_step_fsdp_fn(mesh, train_state_fsdp_specs, 
                                                config.fsdp.data_axis)

    eval_step_fsdp_fn = get_eval_step_fsdp_fn(mesh, train_state_fsdp_specs.params, 
                                              train_state.apply_fn, 
                                              config.fsdp.data_axis, 
                                              dec_pad_id)

    train_ds_iterator = get_dataset_iterator(train_ds,
                                             config.data.batch_size,
                                             is_infinite=True)

    # Create the checkpoint manager
    ckp_options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                               best_fn=lambda metrics: metrics['acc'])
    ckp_mngr = ocp.CheckpointManager(os.path.abspath(config.training_output.checkpoint_path),
                                     options=ckp_options)

    # Create tf.summary
    train_writer, val_writer = _create_summary_writers(config, log_dir_prefix)

    # Get metric shape
    sample_batch = generate_random_batch(jax.random.key(1),
                                         config.data.batch_size,
                                         config.data.max_seq_len,
                                         config.data.vocab_size)
    _, train_metrics_shapes = jax.eval_shape(train_step_fsdp_fn, train_state, None, sample_batch, config.fsdp.data_axis)
    val_metrics_shapes = jax.eval_shape(eval_step_fsdp_fn, train_state.params, None, sample_batch, train_state.apply_fn, config.fsdp.data_axis, dec_pad_id)

    for epoch in range(config.optimizer.training_epochs):
        train_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), train_metrics_shapes)
        val_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), val_metrics_shapes)
        validation_ds_iterator = get_dataset_iterator(validation_ds,
                                                      config.data.batch_size,
                                                      is_infinite=False)

        print(f"Epoch {epoch + 1}")
        for _ in tqdm(range(config.optimizer.steps_per_epochs)):
            train_state, train_metrics = train_step_fsdp_fn(train_state,
                                                    train_metrics,
                                                    next(train_ds_iterator),
                                                    config.fsdp.data_axis)

        for val_batch in validation_ds_iterator:
            val_metrics = eval_step_fsdp_fn(train_state.params, val_metrics, val_batch, train_state.apply_fn, config.fsdp.data_axis, dec_pad_id)

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

        with train_writer.as_default():
            tf.summary.scalar('loss', train_final_loss, step=epoch)
            tf.summary.scalar('accuracy', train_final_accuracy, step=epoch)

        with val_writer.as_default():
            tf.summary.scalar('loss', val_final_loss, step=epoch)
            tf.summary.scalar('accuracy', val_final_accuracy, step=epoch)

        print(f"Training:    Loss: {train_final_loss}    Accuracy: {train_final_accuracy}")
        print(f"Validation:  Loss: {val_final_loss}    Accuracy: {val_final_accuracy}")

    ckp_mngr.wait_until_finished()
    return train_state


def evaluate_model(model_apply_fn: Any,
                   model_params: PyTree,
                   params_fsdp_specs: Any,
                   config: ml_collections.ConfigDict,
                   test_ds: tf.data.Dataset,
                   dec_pad_id: int) -> TrainState:
    test_ds_iterator = get_dataset_iterator(test_ds,
                                            config.data.batch_size,
                                            is_infinite=False)
    eval_step_fsdp_fn = get_eval_step_fsdp_fn(params_fsdp_specs, 
                                              model_apply_fn, 
                                              config.fsdp.data_axis, 
                                              dec_pad_id)

    # Get metric shape
    sample_batch = generate_random_batch(jax.random.key(1),
                                     config.data.batch_size,
                                     config.data.max_seq_len,
                                     config.data.vocab_size)
    eval_metrics_shapes = jax.eval_shape(eval_step_fsdp_fn, model_params, None, sample_batch, model_apply_fn, config.fsdp.data_axis, dec_pad_id)
    eval_metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), eval_metrics_shapes)

    for eval_batch in test_ds_iterator:
        eval_metrics = eval_step_fsdp_fn(model_params, eval_metrics, eval_batch, model_apply_fn, config.fsdp.data_axis, dec_pad_id)

    eval_final_loss = float(eval_metrics['loss'].get_metric_val())
    eval_final_accuracy = float(eval_metrics['acc'].get_metric_val())

    print(f"Eval:  Loss: {eval_final_loss}    Accuracy: {eval_final_accuracy}")
    return eval_final_loss, eval_final_accuracy
