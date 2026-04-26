import jax

main_rng_key = jax.random.key(18)

# Test model components
test_num_heads = 8
test_d_proj = 32
test_emb_dim = 256
test_src_len= 60
test_batch_size = 32
test_d_inner = 1024
test_dropout = 0.2
test_num_blocks = 6
test_ff_d_inner = 512
test_vocab_size = 5000
max_seq_len = 64

test_prng_key = jax.random.key(44)
key_1, key_2, key_3, key_4, key_5, dropout_key = jax.random.split(test_prng_key, 6)


# Test the positional encoding module
test_random_input = jax.random.normal(test_prng_key,
                                      (test_batch_size, max_seq_len, test_emb_dim))
test_pos_enc_module = PositionalEncoding(test_emb_dim, max_seq_len)
variables = test_pos_enc_module.init(test_prng_key, test_random_input)
pos_output = test_pos_enc_module.apply({}, test_random_input)
assert test_random_input.shape == pos_output.shape, "Incorrect expected output shape"

# Test multi head attention layer shape output
test_multi_head_att_module = MultiHeadAttentionModule(test_num_heads,
                                                      test_emb_dim,
                                                      test_d_proj,
                                                      test_d_proj)
k = jax.random.normal(key_1, (test_batch_size, test_src_len, test_emb_dim))
v = jax.random.normal(key_2, (test_batch_size, test_src_len, test_emb_dim))
q = jax.random.normal(key_3, (test_batch_size, test_src_len, test_emb_dim))
sample_mask = jax.random.choice(key_3, 2, (1, 1, test_src_len, test_src_len))

variables = test_multi_head_att_module.init(test_prng_key, k, v, q)
params = variables['params']
attentions = test_multi_head_att_module.apply({'params': params}, k, v, q, mask=sample_mask)
assert attentions.shape == (test_batch_size, test_src_len, test_emb_dim), "Multihead Attention: Incorrect expected output shape"



# Test Feedforward module
sample_input = jax.random.normal(key_1, (test_batch_size, test_src_len, test_emb_dim))

test_ff_module = FeedForwardModule(test_d_inner, test_emb_dim, test_dropout)
variables = test_ff_module.init(key_2, sample_input)

test_output = test_ff_module.apply(variables, sample_input)
assert test_output.shape == (test_batch_size, test_src_len, test_emb_dim), "FF Module: Incorrect expected output shape"


# Test Add&Norm module
sample_x = jax.random.normal(key_1, (test_batch_size, test_src_len, test_emb_dim))
sample_residual_x = jax.random.normal(key_2, (test_batch_size, test_src_len, test_emb_dim))

test_add_norm_module = AddAndNormModule(test_dropout)
variables = test_add_norm_module.init(key_3, sample_x, sample_residual_x, training=True)

test_output = test_add_norm_module.apply(variables,
                                         sample_x,
                                         sample_residual_x,
                                         training=True,
                                         rngs={'dropout': dropout_key})
assert test_output.shape == (test_batch_size, test_src_len, test_emb_dim), "Add&Norm Module: Incorrect expected output shape"



# Test Encoder block module
sample_mask = jax.random.choice(key_1, 2, (1, 1, test_src_len))
sample_x = jax.random.normal(key_2, (test_batch_size, test_src_len, test_emb_dim))

test_encoder_block_module = EncoderBlockModule(test_d_inner, test_emb_dim,
                                               test_dropout, test_num_heads, test_d_proj)
variables = test_encoder_block_module.init(key_3, sample_x)
test_output = test_encoder_block_module.apply(variables, sample_x,
                                              mask=sample_mask, training=True,
                                              rngs={'dropout': dropout_key})
assert test_output.shape == (test_batch_size, test_src_len, test_emb_dim), "Encoder Module: Incorrect expected output shape"




# Test Decoder block module
sample_mask = jax.random.choice(key_1, 2, (1, 1, test_src_len))
sample_x = jax.random.normal(key_2, (test_batch_size, test_src_len, test_emb_dim))
sample_enc_output = jax.random.normal(key_3, (test_batch_size, test_src_len, test_emb_dim))

test_decoder_block_module = DecoderBlockModule(test_d_inner, test_emb_dim,
                                               test_dropout, test_num_heads, test_d_proj)
variables = test_decoder_block_module.init(key_4, sample_x, sample_enc_output)
test_output = test_decoder_block_module.apply(variables, sample_x, sample_enc_output,
                                              dec_mask=sample_mask, enc_dec_mask=sample_mask,
                                              training=True, rngs={'dropout': dropout_key})
assert test_output.shape == (test_batch_size, test_src_len, test_emb_dim), "Decoder Module: Incorrect expected output shape"



# Test transformer output shape
sample_enc_x = jax.random.choice(key_3, test_vocab_size, (test_batch_size, test_src_len))
sample_dec_x = jax.random.choice(key_4, test_vocab_size, (test_batch_size, test_src_len))

test_transformer_module = TransformerModule(test_num_blocks, test_ff_d_inner,
                                            test_emb_dim, test_dropout,
                                            test_num_heads, test_d_proj,
                                            test_vocab_size, max_seq_len)
variables = test_transformer_module.init(key_5, sample_enc_x, sample_dec_x)
test_output = test_transformer_module.apply(variables, sample_enc_x, sample_dec_x,
                                            training=True, rngs={'dropout': dropout_key})
assert test_output.shape == (test_batch_size, test_src_len, test_vocab_size), "Full transformer Module: Incorrect expected output shape"


del test_random_input
del test_pos_enc_module
del pos_output
del sample_mask
del sample_input
del sample_x
del sample_residual_x
del test_ff_module
del sample_enc_output
del test_encoder_block_module
del test_decoder_block_module
del test_multi_head_att_module
del test_add_norm_module
del sample_enc_x
del sample_dec_x
del test_transformer_module
del k
del v
del q
del params
del attentions
del variables
del test_output