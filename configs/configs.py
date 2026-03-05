import pathlib
import ml_collections

def get_configs():
    data_config = ml_collections.ConfigDict(
        dict(
            special_tokens = ['<|startoftext|>','<|endoftext|>','<|pad|>'],
            de_tokenizer_model_path = 'de_tokenizer_30_000_vocab_size_model',
            en_tokenizer_model_path = 'en_tokenizer_30_000_vocab_size_model',
            train_ds_path = 'proprocessed_data/train.tfrecord',
            validation_ds_path = 'proprocessed_data/validation.tfrecord',
            test_ds_path = 'proprocessed_data/test.tfrecord',
            vocab_size = 30_000,
            max_seq_len = 100,
            batch_size = 32,
        )
    )

    model_config = ml_collections.ConfigDict(
        dict(
            emb_dim = 128,
            num_heads = 8,
            d_proj = 128,
            ff_d_inner_factor = 4, # ff_d_inner = ff_d_inner_factor * emb_dim
            dropout = 0.1,
            num_blocks = 4,
        )
    )

    optimizer_config = ml_collections.ConfigDict(
        dict(
            base_lr = 0.001,
            warmup_epochs = 2,
            training_epochs = 30,
            steps_per_epochs = 10_000
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

    # Create the path if it does not exist.
    pathlib.Path(config.data.de_tokenizer_model_path).mkdir(exist_ok=True)
    pathlib.Path(config.data.en_tokenizer_model_path).mkdir(exist_ok=True)
    pathlib.Path(config.training_output.metric_path).mkdir(exist_ok=True)
    pathlib.Path(config.training_output.checkpoint_path).mkdir(exist_ok=True)
    pathlib.Path(config.training_output.trace_path).mkdir(exist_ok=True)
    pathlib.Path('proprocessed_data').mkdir(exist_ok=True)
    
    return ml_collections.FrozenConfigDict(config)