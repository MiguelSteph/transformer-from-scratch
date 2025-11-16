import optax
import jax
from jax import numpy as jnp
from flax.training import train_state


class TrainState(train_state.TrainState):
    dropout_key: jax.Array


@jax.jit
def train_step(state, batch):
    # Call the model to get the logits
    current_dropout_key = jax.random.fold_in(key=state.dropout_key, data=state.step)
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            enc_x=batch['src_tokens'],
            dec_x=batch['trg_input_tokens'],
            enc_mask=batch['src_padding_mask'],
            dec_mask=batch['trg_padding_mask'],
            training=True,
            rngs={'dropout': current_dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels=batch['trg_output_tokens'])
        loss = loss * batch['trg_padding_mask']
        loss = jnp.divide(jnp.sum(loss), jnp.sum(batch['trg_padding_mask']))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    metrics = {
        'acc': compute_accuracy(logits, batch['trg_output_tokens'], batch['trg_padding_mask']),
        'loss': loss
    }
    metrics = jax.lax.pmean(metrics, "batch")
    return new_state, metrics


@jax.jit
def eval_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            enc_x=batch['src_tokens'],
            dec_x=batch['trg_input_tokens'],
            enc_mask=batch['src_padding_mask'],
            dec_mask=batch['trg_padding_mask'],
            training=False,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels=batch['trg_output_tokens'])
        loss = loss * batch['trg_padding_mask']
        loss = jnp.divide(jnp.sum(loss), jnp.sum(batch['trg_padding_mask']))
        return loss, logits
    loss, logits = loss_fn(state.params)

    metrics = {
        'acc': compute_accuracy(logits, batch['trg_output_tokens'], batch['trg_padding_mask']),
        'loss': loss
    }
    metrics = jax.lax.pmean(metrics, "batch")
    return metrics


def compute_accuracy(logits, labels, mask=None):
    predicted_label = jnp.argmax(logits, axis=-1)
    if mask is None:
        mask = jnp.ones_like(labels)
    accuracy = jnp.equal(predicted_label, labels)
    accuracy = accuracy * mask
    accuracy = jnp.divide(jnp.sum(accuracy), jnp.sum(mask))
    return accuracy


def create_train_state(config, param_init_prng_key, dropout_key):
    # Create the model
    model = create_model(config)
    
    key_1, key_2, key_3, key_4, key_5 = jax.random.split(param_init_prng_key, 5)
    sample_enc_mask = jax.random.choice(key_1, 2, (1, config.max_src_len))
    sample_dec_mask = jax.random.choice(key_2, 2, (1, config.max_trg_len))
    sample_enc_x = jax.random.choice(key_3, config.max_vocab_size, (1, config.max_src_len))
    sample_dec_x = jax.random.choice(key_4, config.max_vocab_size, (1, config.max_trg_len))
    variables = model.init(key_5, sample_enc_x, sample_dec_x)
    params = variables['params']

    # Create the optimizer
    cosine_epochs = config.training_epochs - config.warmup_epochs
    scheduler = create_learning_rate_scheduler(config.base_lr, config.warmup_epochs, 
                                           cosine_epochs, config.steps_per_epochs)

    state = TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = optax.adam(scheduler),
        dropout_key = dropout_key,
    )
    return state


def create_learning_rate_scheduler(base_lr, warmup_epochs, cosine_epochs, steps_per_epochs):
    warmup_fn = optax.linear_schedule(init_value=0, end_value=base_lr, 
                                      transition_steps=warmup_epochs * steps_per_epochs)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_lr, 
                                            decay_steps=cosine_epochs * steps_per_epochs)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], 
                                       boundaries=[warmup_epochs * steps_per_epochs])
    return schedule_fn


def create_model(config):
    return TransformerModule(config.num_blocks, config.ff_d_inner, 
                             config.emb_dim, config.dropout, 
                             config.num_heads, config.d_proj, 
                             config.max_vocab_size, config.max_trg_len)
