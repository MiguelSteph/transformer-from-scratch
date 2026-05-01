import ml_collections

def get_configs():
    data_config = ml_collections.ConfigDict(
        dict(
            special_tokens = ['<|startoftext|>','<|endoftext|>'],
            de_tokenizer_model_path = 'tokenizer_32_000_vocab_size_model',
            en_tokenizer_model_path = 'tokenizer_32_000_vocab_size_model',
            tokenizer_model_path = 'tokenizer_32_000_vocab_size_model',
            train_ds_path = 'preprocessed_data/train.tfrecord',
            validation_ds_path = 'preprocessed_data/validation.tfrecord',
            test_ds_path = 'preprocessed_data/test.tfrecord',
            vocab_size = 32_000,
            max_seq_len = 100,
            batch_size = 32,
        )
    )

    model_config = ml_collections.ConfigDict(
        dict(
            emb_dim = 512,
            num_heads = 8,
            d_proj = 64,
            ff_d_inner_factor = 4, # ff_d_inner = ff_d_inner_factor * emb_dim
            dropout = 0.1,
            num_blocks = 6,
        )
    )

    optimizer_config = ml_collections.ConfigDict(
        dict(
            base_lr = 0.0005,
            warmup_epochs = 4,
            training_epochs = 30,
            steps_per_epochs = 15_000
        )
    )

    training_output_config = ml_collections.ConfigDict(
        dict(
            metric_path = 'log_dir/metrics',
            checkpoint_path = 'log_dir/checkpoints',
            trace_path = 'log_dir/traces',
        )
    )

    config = ml_collections.ConfigDict()
    config.data = data_config
    config.model = model_config
    config.optimizer = optimizer_config
    config.training_output = training_output_config
    
    return ml_collections.FrozenConfigDict(config)
