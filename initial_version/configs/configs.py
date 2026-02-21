import pathlib
import ml_collections

def get_configs():
    config = ml_collections.ConfigDict()

    config.emb_dim = 128
    config.num_heads = 8
    config.d_proj = 128
    config.ff_d_inner = 2 * config.emb_dim
    config.dropout = 0.1
    config.num_blocks = 4

    config.max_trg_len = 100
    config.max_src_len = 80
    config.max_vocab_size = 5000

    config.train_size = 150494
    config.val_size = 8318
    config.test_size = 8318
    
    config.batch_size = 64
    config.base_lr = 0.0001
    config.warmup_epochs = 2
    config.training_epochs = 18 
    config.steps_per_epochs = int(config.train_size / config.batch_size)
    config.val_steps = int(config.val_size / config.batch_size)
    config.test_steps = int(config.test_size / config.batch_size)

    config.dataset_folder = 'dataset'
    config.dataset_path = 'dataset/fra.txt'
    config.metric_path = 'metrics'
    config.checkpoint_path = 'checkpoints'
    config.saved_tokenizer_path = 'tokenizer_models'
    config.src_tokenizer_path = 'tokenizer_models/tokenizer_en'
    config.trg_tokenizer_path = 'tokenizer_models/tokenizer_fr'
    config.tokenizer_special_tokens = ['<|startoftext|>','<|endoftext|>','<|pad|>']

    # Create the path if it does not exist.
    pathlib.Path(config.saved_tokenizer_path).mkdir(exist_ok=True)
    pathlib.Path(config.src_tokenizer_path).mkdir(exist_ok=True)
    pathlib.Path(config.trg_tokenizer_path).mkdir(exist_ok=True)

    return config

