import os
import numpy as np
import tensorflow as tf
from tensorflow.train import Int64List, Features, Feature, Example
import tensorflow_datasets as tfds
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import multiprocessing as mp
from itertools import repeat

from configs import get_configs
from tokenizer import build_and_save_tokenizer


def download_and_get_raw_datasets() -> list[tf.data.Dataset]:
    tfds_config = tfds.translate.wmt.WmtConfig(
        version="1.0.0",
        language_pair=("de", "en"),
        subsets={
            tfds.Split.TRAIN: [
                "europarl_v7",
                "commoncrawl",
                "multiun",
                "newscommentary_v8",
                "gigafren",
                "wikiheadlines_ru",
                "yandexcorpus",
                "czeng_10",
            ],
            tfds.Split.VALIDATION: [
                "newstest2012",
                "newstest2011",
                "newstest2010",
                "newstest2009",
                "newstest2008",
                "newssyscomb2009",
            ],
            tfds.Split.TEST: ["newstest2013"],
        },
    )
    builder = tfds.builder("wmt_translate", config=tfds_config)
    builder.download_and_prepare()
    # return builder.as_dataset(split=["train", "validation", "test"])

    # FOR TESTING, SHOULD BE REMOVED LATER
    train_ds, validation_ds, test_ds = builder.as_dataset(split=["train", "validation", "test"])
    return train_ds.take(10_000), validation_ds, test_ds


def build_and_save_tokenizer_models() -> None:
    config = get_configs()
    train_ds, validation_ds, test_ds = download_and_get_raw_datasets()
    full_ds = train_ds.concatenate(validation_ds).concatenate(test_ds)
    de_data = full_ds.map(lambda x: x["de"], num_parallel_calls=tf.data.AUTOTUNE)
    en_data = full_ds.map(lambda x: x["en"], num_parallel_calls=tf.data.AUTOTUNE)

    # German tokenizer
    print(f"--- Starting German tokenizer ---")
    build_and_save_tokenizer(dataset=de_data,
                            vocab_size=config.data.vocab_size,
                            tokenizer_path=config.data.de_tokenizer_model_path,
                            special_tokens=list(config.data.special_tokens))
    print(f"--- German tokenizer built and saved ---")

    # English tokenizer
    print(f"--- Starting English tokenizer ---")
    build_and_save_tokenizer(dataset=en_data,
                            vocab_size=config.data.vocab_size,
                            tokenizer_path=config.data.en_tokenizer_model_path,
                            special_tokens=list(config.data.special_tokens))
    print(f"--- English tokenizer built and saved ---")


def serialized_example(de_input_tokens: list[float],
                   de_input_mask: list[int],
                   en_input_tokens: list[float],
                   en_input_mask: list[int],
                   en_output_tokens: list[float]) -> str:
    return Example(
        features = Features(
            feature = {
                "de_input": Feature(int64_list=Int64List(value=de_input_tokens)),
                "de_input_mask": Feature(int64_list=Int64List(value=de_input_mask)),
                "en_input": Feature(int64_list=Int64List(value=en_input_tokens)),
                "en_input_mask": Feature(int64_list=Int64List(value=en_input_mask)),
                "en_output": Feature(int64_list=Int64List(value=en_output_tokens)),
            }
        )
    ).SerializeToString()


def get_serialized_examples(de_tokenizer, en_tokenizer, samples, max_seq_len) -> str:
    serialized_examples = []
    de_inputs = samples["de"]
    en_inputs = samples["en"]
    for index in range(de_inputs.shape[0]):
        de_input = de_inputs[index].decode("utf-8").strip()
        en_input = en_inputs[index].decode("utf-8").strip()
        de_input_tokens = de_tokenizer.encode(de_input).ids
        en_input_tokens = en_tokenizer.encode('<|startoftext|>' + en_input).ids
        en_output_tokens = en_tokenizer.encode(en_input + '<|endoftext|>').ids

        if len(de_input_tokens) > max_seq_len or len(en_input_tokens) > max_seq_len:
            continue
        de_pad_encoding = de_tokenizer.encode('<|pad|>').ids[0]
        en_pad_encoding = en_tokenizer.encode('<|pad|>').ids[0]

        de_input_tokens_with_padding = np.concatenate((de_input_tokens, np.full(max_seq_len - len(de_input_tokens), de_pad_encoding)))
        en_input_tokens_with_padding = np.concatenate((en_input_tokens, np.full(max_seq_len - len(en_input_tokens), en_pad_encoding)))
        en_output_tokens_with_padding = np.concatenate((en_output_tokens, np.full(max_seq_len - len(en_output_tokens), en_pad_encoding)))

        de_input_mask = (de_input_tokens_with_padding != de_pad_encoding).astype(np.int32)
        en_input_mask = (en_input_tokens_with_padding != en_pad_encoding).astype(np.int32)

        serialized_examples.append(serialized_example(de_input_tokens_with_padding,
                                    de_input_mask,
                                    en_input_tokens_with_padding,
                                    en_input_mask,
                                    en_output_tokens_with_padding))
    return serialized_examples


def preprocessed_and_saved_dataset(de_tokenizer: ByteLevelBPETokenizer, 
                                    en_tokenizer: ByteLevelBPETokenizer,
                                    dataset: tf.data.Dataset, 
                                    ds_path: str, 
                                    max_seq_len: int) -> None:
    if not os.path.exists(os.path.dirname(ds_path)):
        Path(os.path.dirname(ds_path)).mkdir(parents=True, exist_ok=True)

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(ds_path, options) as f:
        with mp.Pool(os.cpu_count()) as pool:
            for serialized_examples in pool.imap(get_serialized_examples, zip(repeat(de_tokenizer), repeat(en_tokenizer), dataset.batch(1000).as_numpy_iterator(), repeat(max_seq_len)), chunksize=16):
                for example_str in serialized_examples:
                    f.write(example_str)


def load_preprocessed_dataset(ds_path: str, max_seq_len: int) -> tf.data.Dataset:
    feature_description = {
        "de_input": tf.io.FixedLenFeature([max_seq_len], dtype=tf.int32, default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_len])),
        "de_input_mask": tf.io.FixedLenFeature([max_seq_len], dtype=tf.int32, default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_len])),
        "en_input": tf.io.FixedLenFeature([max_seq_len], dtype=tf.int32, default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_len])),
        "en_input_mask": tf.io.FixedLenFeature([max_seq_len], dtype=tf.int32, default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_len])),
        "en_output": tf.io.FixedLenFeature([max_seq_len], dtype=tf.int32, default_value=tf.constant(0, dtype=tf.int64, shape=[max_seq_len]))
    }
    def parse_example(serialized_example: str):
        example = tf.io.parse_single_example(serialized_example, feature_description)
        return {
            "de_input": example["de_input"],
            "de_input_mask": example["de_input_mask"],
            "en_input": example["en_input"],
            "en_input_mask": example["en_input_mask"],
            "en_output": example["en_output"],
        }

    loaded_ds = tf.data.TFRecordDataset([ds_path], compression_type="GZIP")
    return loaded_ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)