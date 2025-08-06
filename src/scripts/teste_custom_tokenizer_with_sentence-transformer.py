import os
import json
from transformers import AutoTokenizer
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

if __name__=="__main__":
    # Caminhos
    model_name = "alfaneo/jurisbert-base-portuguese-uncased"
    custom_tokenizer_path = "./results/tokenizer_custom.json"
    output_path = "./results/tokenizer_merged"

    # 1. Carrega vocabulário do tokenizer_custom.json
    with open(custom_tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    custom_vocab = tokenizer_data.get("model", {}).get("vocab", {})
    custom_tokens = list(custom_vocab.keys())

    # Remove tokens especiais já presentes no tokenizer base
    tokens_a_adicionar = [tok for tok in custom_tokens if tok not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]]

    # 2. Carrega tokenizer base
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Adiciona novos tokens
    tokens_adicionados = tokenizer.add_tokens(tokens_a_adicionar)
    print(f"{tokens_adicionados} tokens adicionados ao tokenizer base.")

    # 4. Salva o novo tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"Tokenizer expandido salvo em '{output_path}'.")

    word_embedding_model = Transformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path=output_path
    )

    # Resize após carregar modelo
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))

    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    frase = "A violência_doméstica ocorre frequentemente na zona_rural. Tribunal de justiça TJES"
    embedding = sentence_model.encode(frase)
    print(embedding)

