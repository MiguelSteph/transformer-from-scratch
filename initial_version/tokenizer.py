import os
from tokenizers import ByteLevelBPETokenizer

def build_and_save_tokenizer(config):
    if not os.path.exists(config.dataset_path):
        raise Exception("Please download the dataset first.")
    
    fr_sentences = []
    en_sentences = []

    with open(config.dataset_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            line_split = line.split('\t')
            if len(line_split) == 2:
                fr_sentences.append(line_split[1].strip())
                en_sentences.append(line_split[0].strip())

    fr_tokenizer = ByteLevelBPETokenizer()
    en_tokenizer = ByteLevelBPETokenizer()

    fr_tokenizer.train_from_iterator(fr_sentences, 
                                    vocab_size=config.max_vocab_size, 
                                    min_frequency=2, 
                                    special_tokens=config.tokenizer_special_tokens)
    en_tokenizer.train_from_iterator(en_sentences, 
                                    vocab_size=config.max_vocab_size, 
                                    min_frequency=2, 
                                    special_tokens=config.tokenizer_special_tokens)

    fr_tokenizer.save_model(config.trg_tokenizer_path)
    en_tokenizer.save_model(config.src_tokenizer_path)
