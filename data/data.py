import tensorflow as tf
import tensorflow_datasets as tfds
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

    special_tokens = config.data.special_tokens
    start_time = time.time()
    # German tokenizer
    build_and_save_tokenizer(dataset=de_data,
                            vocab_size=config.data.vocab_size,
                            tokenizer_path=config.data.de_tokenizer_model_path)
    print(f"--- German tokenizer seconds: {time.time() - start_time} ---")

    start_time = time.time()
    # English tokenizer
    build_and_save_tokenizer(dataset=en_data,
                            vocab_size=config.data.vocab_size,
                            tokenizer_path=config.data.en_tokenizer_model_path)
    print(f"--- English tokenizer seconds: {time.time() - start_time} ---")
