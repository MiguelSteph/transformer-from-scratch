import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import functools


def get_inference_fsdp_fn(mesh, params_fsdp_specs, params, model, data_axis_name):
    inference_fn = jax.jit(
        jax.shard_map(
            model.apply,
            mesh=mesh,
            in_specs=(params_fsdp_specs, P(data_axis_name), P(data_axis_name)),
            out_specs=P(),
            check_rep=False,
        ),
    )
    return functools.partial(inference_fn, {'params': params})


def run_inference(seed_key, src_sentence, config, inference_fn, src_tokenizer, trg_tokenizer):
    end_token_id = trg_tokenizer.encode('<|endoftext|>').ids[0]

    enc_pad_id = src_tokenizer.encode('<|pad|>').ids[0]
    dec_pad_id = trg_tokenizer.encode('<|pad|>').ids[0]

    src_tokens = src_tokenizer.encode(src_sentence).ids
    src_tokens = jnp.concatenate([jnp.array(src_tokens), jnp.full(config.data.max_seq_len - len(src_tokens), enc_pad_id)])
    trg_tokens = jnp.array(trg_tokenizer.encode('<|startoftext|>').ids)
    new_seed_key = seed_key

    for idx in range(config.data.max_seq_len):
        curr_key, new_seed_key = jax.random.split(new_seed_key)
        trg_input_tokens = jnp.concatenate([trg_tokens, jnp.full(config.data.max_seq_len - trg_tokens.shape[0], dec_pad_id)])

        model_output = inference_fn(jnp.expand_dims(src_tokens, 0),
                                    jnp.expand_dims(trg_input_tokens, 0))

        token_logits = model_output[0][trg_tokens.shape[0]-1]
        token_prob = jax.nn.softmax(token_logits)
        values, indices = jax.lax.top_k(token_prob, 5)
        output_token = jax.random.choice(curr_key, indices, p=values/values.sum())
        if output_token == end_token_id:
            break
        trg_tokens = jnp.concatenate([trg_tokens, jnp.array([output_token])])

    output_str = trg_tokenizer.decode(trg_tokens.tolist())
    return output_str
