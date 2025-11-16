import pathlib
import requests
import zipfile
import jax
from jax import numpy as jnp
import grain.python as grain

class DataSource(grain.RandomAccessDataSource):
    def __init__(self, src_tokenizer, trg_tokenizer, source_data, target_data, max_src_len, max_trg_len):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_data = source_data
        self.trg_data = target_data
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __getitem__(self, idx):
        raw_src = self.src_data[idx].strip()
        raw_trg = self.trg_data[idx].strip()
        src_tokens = self.src_tokenizer.encode(raw_src).ids
        trg_input_tokens = self.trg_tokenizer.encode('<|startoftext|>' + raw_trg).ids
        trg_output_tokens = self.trg_tokenizer.encode(raw_trg + '<|endoftext|>').ids

        src_tokens = jnp.array(src_tokens)[:self.max_src_len]
        trg_input_tokens = jnp.array(trg_input_tokens)[:self.max_trg_len]
        trg_output_tokens = jnp.array(trg_output_tokens)[:self.max_trg_len]

        src_pad_encoding = self.src_tokenizer.encode('<|pad|>').ids[0]
        trg_pad_encoding = self.trg_tokenizer.encode('<|pad|>').ids[0]
        
        src_tokens = jnp.concatenate([src_tokens, jnp.full(self.max_src_len - src_tokens.shape[0], src_pad_encoding)])
        trg_input_tokens = jnp.concatenate([trg_input_tokens, jnp.full(self.max_trg_len - trg_input_tokens.shape[0], trg_pad_encoding)])
        trg_output_tokens = jnp.concatenate([trg_output_tokens, jnp.full(self.max_trg_len - trg_output_tokens.shape[0], trg_pad_encoding)])
        
        src_padding_mask = (src_tokens != src_pad_encoding).astype(jnp.int32)
        trg_padding_mask = (trg_input_tokens != trg_pad_encoding).astype(jnp.int32)
        
        return {'src_tokens': src_tokens, 
                'src_padding_mask': src_padding_mask,
                'trg_input_tokens': trg_input_tokens, 
                'trg_output_tokens': trg_output_tokens,
                'trg_padding_mask': trg_padding_mask,
               }
    
    def __len__(self):
        return len(self.src_data)


def load_and_prepare_dataset(src_tokenizer, 
                             trg_tokenizer, 
                             dataset_path, 
                             prng_key, 
                             num_epochs, 
                             train_batch_size, 
                             test_or_val_batch_size,
                             max_src_len, 
                             max_trg_len):
    prng_key_1, prng_key_2 = jax.random.split(prng_key)
    en_sentences, fr_sentences = load_dataset(dataset_path)
    unique_idx, non_unique_idx = get_unique_and_nonunique_indexes(en_sentences)
    shuffle_indexes = jax.random.permutation(prng_key_1, jnp.array(unique_idx))
    test_val_size = len(unique_idx) // 10
    test_set_indexes = shuffle_indexes[:test_val_size]
    val_set_indexes = shuffle_indexes[test_val_size: 2*test_val_size]
    train_indexes = jnp.concatenate([shuffle_indexes[2*test_val_size:], jnp.array(non_unique_idx)])
    train_indexes = jax.random.permutation(prng_key_2, train_indexes)

    fr_test = [fr_sentences[idx] for idx in test_set_indexes]
    en_test = [en_sentences[idx] for idx in test_set_indexes]

    fr_val = [fr_sentences[idx] for idx in val_set_indexes]
    en_val = [en_sentences[idx] for idx in val_set_indexes]

    fr_train = [fr_sentences[idx] for idx in train_indexes]
    en_train = [en_sentences[idx] for idx in train_indexes]

    test_dataset = DataSource(src_tokenizer, trg_tokenizer, en_test, fr_test, max_src_len, max_trg_len)
    val_dataset = DataSource(src_tokenizer, trg_tokenizer, en_val, fr_val, max_src_len, max_trg_len)
    train_dataset = DataSource(src_tokenizer, trg_tokenizer, en_train, fr_train, max_src_len, max_trg_len)

    test_sampler = grain.IndexSampler(
        len(test_set_indexes),
        shuffle=True,
        seed=12,
        shard_options=grain.NoSharding(),
        num_epochs=1)

    val_sampler = grain.IndexSampler(
        len(val_set_indexes),
        shuffle=True,
        seed=12,
        shard_options=grain.NoSharding(),
        num_epochs=1)

    train_sampler = grain.IndexSampler(
        len(train_indexes),
        shuffle=True,
        seed=12,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs)

    test_loader = grain.DataLoader(
        data_source = test_dataset,
        sampler = test_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            grain.Batch(test_or_val_batch_size)
        ]
    )

    val_loader = grain.DataLoader(
        data_source = val_dataset,
        sampler = val_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            grain.Batch(test_or_val_batch_size)
        ]
    )

    train_loader = grain.DataLoader(
        data_source = train_dataset,
        sampler = train_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            grain.Batch(train_batch_size)
        ]
    )
    
    return train_loader, val_loader, test_loader


def get_unique_and_nonunique_indexes(text_list):
    duplicate_set = set()
    unique_set = set()
    for text in text_list:
        if text in unique_set:
            duplicate_set.add(text)
        else:
            unique_set.add(text)

    unique_indexes = []
    non_unique_indexes = []
    for idx in range(len(text_list)):
        if text_list[idx] not in duplicate_set:
            unique_indexes.append(idx)
        else:
            non_unique_indexes.append(idx)
    return unique_indexes, non_unique_indexes


def load_dataset(dataset_path):
    fr_sentences = []
    en_sentences = []
    
    with open(dataset_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            line_split = line.split('\t')
            if len(line_split) == 2:
                fr_sentences.append(line_split[1].strip())
                en_sentences.append(line_split[0].strip())
    return en_sentences, fr_sentences


def download_data(config):
    data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip'
    dataset_folder_path = pathlib.Path(config.dataset_folder)
    dataset_folder_path.mkdir(exist_ok=True)
    zip_data_path = dataset_folder_path / 'fra-eng.zip'
    if not zip_data_path.exists():
        response = requests.get(data_url)
        zip_data_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_data_path, "r") as zip_ref:
        zip_ref.extractall(dataset_folder_path)
