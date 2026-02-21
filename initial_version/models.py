import numpy as np
from jax import numpy as jnp
from flax import linen as nn


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

    def __call__(self, x):
        """Adds the positional encodings to the input and returns it.

        Keyword arguments:
        x -- the embeddings. The shape of X is (batch_size, max_seq_len, emb_dim)
        """
        seq_len = x.shape[1]
        x = x + self.pos_encodings[:, :seq_len]
        return x




class MultiHeadAttentionModule(nn.Module):
    num_heads: int # Number of heads
    d_q: int # Embedding dimension of the query
    d_k_proj: int # Projection dimension of the key or the query
    d_v_proj: int # Projection dimension of the value
    use_causal_mask: bool = False # Whether we should use a causal mask

    def setup(self):
        self.k_proj = nn.Dense(self.num_heads * self.d_k_proj, 
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        self.v_proj = nn.Dense(self.num_heads * self.d_v_proj, 
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        self.q_proj = nn.Dense(self.num_heads * self.d_k_proj, 
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
        self.proj_back = nn.Dense(self.d_q,
                                 kernel_init=nn.initializers.xavier_uniform(),
                                 bias_init=nn.initializers.zeros)


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
        mask -- the attention mask on the query. When  The shape of the mask is (batch_size, q_seq_len)
        use_causal_mask -- whether we should apply a causal mask

        Returns the scaled dot product attention in the following shape: (batch_size, num_heads, q_seq_len, d_v)
        """
        d_k = k.shape[-1]
        kv_seq_len = k.shape[2]
        q_seq_len = q.shape[2]
        k_tr = jnp.matrix_transpose(k) # k_tr is now of shape (batch_size, num_heads, d_k, kv_seq_len)
        q_k_tr = jnp.matmul(q, k_tr)
        logits = q_k_tr / jnp.sqrt(d_k)
        # By default we consider all the logits
        computed_mask = jnp.ones((1, 1, q_seq_len, kv_seq_len))
        if self.use_causal_mask:
            computed_mask = self.get_causal_attention_mask(q_seq_len)
   
        if mask is not None:
            reshaped_mask = jnp.reshape(mask, (-1, 1, q_seq_len, 1))
            computed_mask = jnp.minimum(computed_mask, reshaped_mask)
            
        logits = jnp.where(computed_mask == 0, -9e15, logits)
        attention = nn.softmax(logits, axis=-1)
        values = jnp.matmul(attention, v)
        return values

    def get_causal_attention_mask(self, seq_len):
        """
        Given the sequence length, returns the causal attention mask.

        seq_len -- int. The sequence length.
        """
        i = jnp.arange(seq_len)[:, None]
        j = jnp.arange(seq_len)
        mask = (i >= j).astype(jnp.int32)
        mask = jnp.reshape(mask, (1, 1, seq_len, seq_len))
        return mask




class FeedForwardModule(nn.Module):
    d_inner: int  # Inner dimension of the feed forward module
    d_output: int # Output dimension of the module

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_inner, name='ff_inner')(x)
        x = nn.relu(x)
        x = nn.Dense(self.d_output, name='ff_output')(x)
        return x




class AddAndNormModule(nn.Module):
    dropout: float # Dropout rate

    @nn.compact
    def __call__(self, x, residual_x, training=False):
        x = nn.Dropout(self.dropout, deterministic=not training, name='dropout_module')(x)
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
    def __call__(self, x, mask=None, training=False):
        residual_x = x
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim, 
                                     self.d_proj, self.d_proj, 
                                     use_causal_mask=False)(x, x, x, mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = FeedForwardModule(self.ff_d_inner, self.emb_dim)(x)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)
        
        return x




class DecoderBlockModule(nn.Module):
    ff_d_inner: int  # Feed forward inner dimension
    emb_dim: int # Embedding dimension
    dropout: float # Dropout rate
    num_heads: int # Number of attention heads
    d_proj: int # Key, Value and query projection dimension
    
    @nn.compact
    def __call__(self, x, enc_output, mask=None, training=False):
        residual_x = x
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim, 
                                     self.d_proj, self.d_proj, 
                                     use_causal_mask=True)(x, x, x, mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = MultiHeadAttentionModule(self.num_heads, self.emb_dim, 
                                     self.d_proj, self.d_proj, 
                                     use_causal_mask=False)(enc_output, enc_output, x, mask)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        residual_x = x
        x = FeedForwardModule(self.ff_d_inner, self.emb_dim)(x)
        x = AddAndNormModule(self.dropout)(x, residual_x, training)

        return x




class TransformerModule(nn.Module):
    num_blocks: int # number of transformer blocks
    ff_d_inner: int  # Feed forward inner dimension
    emb_dim: int # Embedding dimension
    dropout: float # Dropout rate
    num_heads: int # Number of attention heads
    d_proj: int # Key, Value and query projection dimension
    max_vocab_size: int
    max_seq_len: int # Maximum sequence length

    def setup(self):
        self.src_embed = nn.Embed(self.max_vocab_size, self.emb_dim)
        self.trg_embed = nn.Embed(self.max_vocab_size, self.emb_dim)
        self.pos_embed = PositionalEncoding(self.emb_dim, self.max_seq_len)
        self.encoders = [EncoderBlockModule(self.ff_d_inner, self.emb_dim, 
                                            self.dropout, self.num_heads, 
                                            self.d_proj) 
                         for i in range(self.num_blocks)]
        self.decoders = [DecoderBlockModule(self.ff_d_inner, self.emb_dim, 
                                            self.dropout, self.num_heads, 
                                            self.d_proj) 
                         for i in range(self.num_blocks)]
        self.head = nn.Dense(self.max_vocab_size)


    def __call__(self, enc_x, dec_x, enc_mask=None, dec_mask=None, training=False):
        enc_output = self.src_embed(enc_x)
        enc_output = self.pos_embed(enc_output)
        for i in range(self.num_blocks):
            enc_output = self.encoders[i](enc_output, enc_mask, training)

        dec_output = self.trg_embed(dec_x)
        dec_output = self.pos_embed(dec_output)
        for i in range(self.num_blocks):
            dec_output = self.decoders[i](dec_output, enc_output, dec_mask, training)

        output = self.head(dec_output)
        return output
