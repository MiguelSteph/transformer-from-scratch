import ml_collections
import numpy as np
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple




class PositionalEncoding(nn.Module):
    emb_dim: int        # embedding dimension of the model
    max_seq_len: int    # max sequence length that we expect

    def setup(self):
        internal_pos_encodings = np.zeros((self.max_seq_len, self.emb_dim))
        p = np.arange(self.max_seq_len, dtype=np.float32)[:, None]
        i = np.arange(self.emb_dim, step=2, dtype=np.float32)
        div_term = 10_000 ** (i / self.emb_dim)
        internal_pos_encodings[:, 0::2] = np.sin(p / div_term)
        internal_pos_encodings[:, 1::2] = np.cos(p / div_term)
        internal_pos_encodings = internal_pos_encodings[None]
        self.pos_encodings = jnp.array(internal_pos_encodings)

    def __call__(self, inputs):
        """Adds the positional encodings to the input and returns it.

        Keyword arguments:
        inputs -- the embeddings. The shape of X is (batch_size, max_seq_len, emb_dim)
        """
        seq_len = inputs.shape[1]
        x = inputs + self.pos_encodings[:, :seq_len]
        return x




class MultiHeadAttentionModule(nn.Module):
    num_heads: int # Number of heads
    d_q: int # Embedding dimension of the query
    d_k_proj: int # Projection dimension of the key or the query
    d_v_proj: int # Projection dimension of the value

    def setup(self):
        self.k_proj = nn.Dense(self.num_heads * self.d_k_proj,
                               kernel_init=nn.initializers.xavier_uniform(),
                               use_bias=False)
        self.v_proj = nn.Dense(self.num_heads * self.d_v_proj,
                               kernel_init=nn.initializers.xavier_uniform(),
                               use_bias=False)
        self.q_proj = nn.Dense(self.num_heads * self.d_k_proj,
                               kernel_init=nn.initializers.xavier_uniform(),
                               use_bias=False)
        self.proj_back = nn.Dense(self.d_q,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 use_bias=False)


    def __call__(self, k, v, q, mask=None):
        batch_size = k.shape[0]
        # Project K, V and Q
        k_proj_val = self.k_proj(k)
        v_proj_val = self.v_proj(v)
        q_proj_val = self.q_proj(q)

        # Reshape projections
        k_proj_val = k_proj_val.reshape(batch_size, -1, self.num_heads, self.d_k_proj)
        k_proj_val = k_proj_val.transpose(0, 2, 1, 3)

        v_proj_val = v_proj_val.reshape(batch_size, -1, self.num_heads, self.d_v_proj)
        v_proj_val = v_proj_val.transpose(0, 2, 1, 3)

        q_proj_val = q_proj_val.reshape(batch_size, -1, self.num_heads, self.d_k_proj)
        q_proj_val = q_proj_val.transpose(0, 2, 1, 3)

        # Compute the attention values for each head
        # (batch_size, num_heads, q_seq_len, d_v_proj)
        head_att_vals = self.compute_scaled_dot_product_attention(k_proj_val,
                                                                  v_proj_val,
                                                                  q_proj_val,
                                                                  mask)
        # Reshape the head attention values
        head_att_vals = head_att_vals.transpose(0, 2, 1, 3)
        head_att_vals = head_att_vals.reshape(batch_size, -1, self.num_heads * self.d_v_proj)

        # Projection back to the query dimension
        output = self.proj_back(head_att_vals)

        # (batch_size, q_seq_len, d_k_proj)
        return output


    def compute_scaled_dot_product_attention(self, k, v, q, mask=None):
        """Given the key, the value and the query, this function computes the scaled dot product attention.

        Keyword arguments:
        k -- the key. The shape of the key is (batch_size, num_heads, kv_seq_len, d_k)
        v -- the value. The shape of the value is (batch_size, num_heads, kv_seq_len, d_v)
        q -- the query. The shape of the query is (batch_size, num_heads, q_seq_len, d_k)
        mask -- the mask. The shape of the mask is (1, num_heads, q_seq_len, kv_seq_len)

        Returns the scaled dot product attention in the following shape: (batch_size, num_heads, q_seq_len, d_v)
        """
        d_k = k.shape[-1]
        kv_seq_len = k.shape[2]
        q_seq_len = q.shape[2]
        k_tr = jnp.matrix_transpose(k) # k_tr is now of shape (batch_size, num_heads, d_k, kv_seq_len)
        q_k_tr = jnp.matmul(q, k_tr)
        logits = q_k_tr / jnp.sqrt(d_k)
        logits = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, logits)
        attention = nn.softmax(logits, axis=-1)
        values = jnp.matmul(attention, v)
        return values


class FeedForwardModule(nn.Module):
    d_inner: int  # Inner dimension of the feed forward module
    d_output: int # Output dimension of the module
    dropout: float # Dropout rate

    @nn.compact
    def __call__(self, inputs, training=False):
        x = nn.Dense(self.d_inner, 
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6),
                    name='ff_inner')(inputs)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=not training, name='ff_dropout_inner')(x)
        x = nn.Dense(self.d_output,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6),
                    name='ff_output')(x)
        return x




class AddAndNormModule(nn.Module):
    dropout: float # Dropout rate

    @nn.compact
    def __call__(self, inputs, residual_x, training=False):
        x = nn.Dropout(self.dropout, deterministic=not training, name='dropout_module')(inputs)
        x = x + residual_x
        x = nn.LayerNorm()(x)
        return x




class EncoderBlockModule(nn.Module):
    ff_d_inner: int  # Feed forward inner dimension
    emb_dim: int # Embedding dimension
    dropout: float # Dropout rate
    num_heads: int # Number of attention heads
    d_proj: int # Key, Value and query projection dimension

    @nn.compact
    def __call__(self, inputs, mask=None, training=False):
        residual_x = inputs
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim,
                                     self.d_proj, self.d_proj)(inputs, inputs, inputs, mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = FeedForwardModule(self.ff_d_inner, self.emb_dim, self.dropout)(x, training)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        return x




class DecoderBlockModule(nn.Module):
    ff_d_inner: int  # Feed forward inner dimension
    emb_dim: int # Embedding dimension
    dropout: float # Dropout rate
    num_heads: int # Number of attention heads
    d_proj: int # Key, Value and query projection dimension

    @nn.compact
    def __call__(self, inputs, enc_output, dec_mask=None, enc_dec_mask=None, training=False):
        residual_x = inputs
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim,
                                     self.d_proj, self.d_proj)(inputs, inputs, inputs, dec_mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim,
                                     self.d_proj, self.d_proj)(enc_output, enc_output, x, enc_dec_mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = FeedForwardModule(self.ff_d_inner, self.emb_dim, self.dropout)(x, training)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        return x




class TransformerModule(nn.Module):
    num_blocks: int # number of transformer blocks
    ff_d_inner: int  # Feed forward inner dimension
    emb_dim: int # Embedding dimension
    dropout: float # Dropout rate
    num_heads: int # Number of attention heads
    d_proj: int # Key, Value and query projection dimension
    vocab_size: int
    max_seq_len: int # Maximum sequence length

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.emb_dim, 
                              embedding_init=nn.initializers.normal(stddev=1.0))
        self.pos_embed = PositionalEncoding(self.emb_dim, self.max_seq_len)
        self.encoders = [EncoderBlockModule(self.ff_d_inner, self.emb_dim,
                                            self.dropout, self.num_heads,
                                            self.d_proj)
                         for i in range(self.num_blocks)]
        self.decoders = [DecoderBlockModule(self.ff_d_inner, self.emb_dim,
                                            self.dropout, self.num_heads,
                                            self.d_proj)
                         for i in range(self.num_blocks)]
        self.norm = nn.LayerNorm()


    def __call__(self, enc_x, dec_x, training=False):
        enc_output = self.encode(enc_x, training)
        return self.decode(enc_x, dec_x, enc_output, training)


    def encode(self, enc_x, training=False):
        enc_mask = nn.make_attention_mask(enc_x > 0, enc_x > 0)

        with jax.named_scope("encoder"):
            enc_output = self.embed(enc_x)
            enc_output = self.pos_embed(enc_output)
            for i in range(self.num_blocks):
                enc_output = self.encoders[i](enc_output, enc_mask, training)

        return enc_output
        

    def decode(self, enc_x, dec_x, enc_output, training=False):
        dec_mask = nn.combine_masks(
          nn.make_attention_mask(dec_x > 0, dec_x > 0),
          nn.make_causal_mask(dec_x),
        )
        enc_dec_mask = nn.make_attention_mask(dec_x > 0, enc_x > 0)

        with jax.named_scope("decoder"):
            dec_output = self.embed(dec_x)
            dec_output = self.pos_embed(dec_output)
            for i in range(self.num_blocks):
                dec_output = self.decoders[i](dec_output, enc_output, dec_mask, enc_dec_mask, training)

        logits = self.embed.attend(dec_output)
        logits = logits / jnp.sqrt(self.vocab_size)
        return logits



def create_transformer_module(config: ml_collections.ConfigDict) -> TransformerModule:
    return TransformerModule(
          num_blocks=config.model.num_blocks,
          ff_d_inner=config.model.ff_d_inner_factor * config.model.emb_dim,
          emb_dim=config.model.emb_dim,
          dropout=config.model.dropout,
          num_heads=config.model.num_heads,
          d_proj=config.model.d_proj,
          vocab_size=config.data.vocab_size,
          max_seq_len=config.data.max_seq_len,
        )
