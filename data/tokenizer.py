import os
import tensorflow as tf
from pathlib import Path
from typing import Iterator, List
from tokenizers import ByteLevelBPETokenizer


def create_path_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def str_from_dataset_generator(dataset: tf.data.Dataset) -> Iterator[str]:
    for item in dataset:
        yield item.numpy().decode("utf-8")


def build_and_save_tokenizer(dataset: tf.data.Dataset,
                             vocab_size: int,
                             tokenizer_path: str,
                             special_tokens: List[str]) -> None:
    if os.path.exists(tokenizer_path):
        print("Tokenizer already exists. Skipping building the model.")
        return
    create_path_if_not_exists(tokenizer_path)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(str_from_dataset_generator(dataset),
                                    vocab_size=vocab_size,
                                    min_frequency=2,
                                    special_tokens=list(special_tokens))
    tokenizer.save_model(tokenizer_path)
