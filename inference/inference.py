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
            out_specs=P(data_axis_name),
            check_vma=False,
        ),
    )
    return functools.partial(inference_fn, {'params': params})


def run_inference(seed_key, src_sentences, config, inference_fn, src_tokenizer, trg_tokenizer):
    batch_size = len(src_sentences)
    end_token_id = trg_tokenizer.encode('<|endoftext|>').ids[0]
    start_token_id = trg_tokenizer.encode('<|startoftext|>').ids[0]
    enc_pad_id = src_tokenizer.encode('<|pad|>').ids[0]
    dec_pad_id = trg_tokenizer.encode('<|pad|>').ids[0]

    # Tokenize, truncate, pad all sources → (batch_size, max_seq_len)
    src_tokens_list = []
    for sentence in src_sentences:
        tokens = src_tokenizer.encode(sentence).ids[:config.data.max_seq_len]
        tokens += [enc_pad_id] * (config.data.max_seq_len - len(tokens))
        src_tokens_list.append(tokens)
    src_batch = jnp.array(src_tokens_list, dtype=jnp.int32)

    # Decoder buffer: all sequences start with start_token, rest is padding
    dec_tokens = jnp.full((batch_size, config.data.max_seq_len), dec_pad_id, dtype=jnp.int32)
    dec_tokens = dec_tokens.at[:, 0].set(start_token_id)

    lengths = [1] * batch_size
    finished = [False] * batch_size

    new_seed_key = seed_key

    for _ in range(config.data.max_seq_len - 1):
        if all(finished):
            break

        curr_key, new_seed_key = jax.random.split(new_seed_key)

        # model_output: (batch_size, max_seq_len, vocab_size)
        model_output = inference_fn(src_batch, dec_tokens)

        for i in range(batch_size):
            if finished[i]:
                continue
            curr_key, seq_key = jax.random.split(curr_key)

            # Logit at the last valid position predicts the next token
            token_logits = model_output[i][lengths[i] - 1]
            token_prob = jax.nn.softmax(token_logits)
            values, indices = jax.lax.top_k(token_prob, 5)
            output_token = jax.random.choice(seq_key, indices, p=values / values.sum())

            if output_token == end_token_id:
                finished[i] = True
            else:
                dec_tokens = dec_tokens.at[i, lengths[i]].set(output_token)
                lengths[i] += 1

    # Decode each sequence, skipping the start token
    return [trg_tokenizer.decode(dec_tokens[i, 1:lengths[i]].tolist())
            for i in range(batch_size)]