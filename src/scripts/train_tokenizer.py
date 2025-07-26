import os
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers
from utils.load_dataset import load_data

if __name__ == "__main__":
    base_folder_path = "./results"
    DATA_PATH = os.path.join(base_folder_path, "gotten_tokens.csv")
    print(f"Carregando dataset de '{DATA_PATH}'...")
    dataset = load_data(DATA_PATH)

    # Extração de tokens únicos
    tokens_unicos = sorted(dataset['token'].dropna().unique())

    # Tokens especiais
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # Construção do vocabulário com os tokens especiais no início
    vocab_dict = {token: idx for idx, token in enumerate(special_tokens + tokens_unicos)}

    # Criação do tokenizer WordLevel
    tokenizer = Tokenizer(models.WordLevel(vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Salvando o tokenizer
    tokenizer.save(os.path.join(base_folder_path, "tokenizer_custom.json"))
    print("Tokenizer treinado e salvo com sucesso!")
