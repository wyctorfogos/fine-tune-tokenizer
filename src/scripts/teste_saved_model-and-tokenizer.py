import os
import json
from transformers import AutoTokenizer
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

def carregar_tokens_customizados(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    vocab_dict = tokenizer_data.get("model", {}).get("vocab", {})
    return list(vocab_dict.keys())

def expandir_tokenizer(model_name, custom_tokens, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Remove tokens especiais padrão
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    novos_tokens = [t for t in custom_tokens if t not in special_tokens]

    adicionados = tokenizer.add_tokens(novos_tokens)
    print(f"✅ {adicionados} tokens adicionados ao tokenizer.")

    tokenizer.save_pretrained(output_path)
    print(f"📦 Tokenizer salvo em: {output_path}")
    return tokenizer

def construir_sentence_encoder(model_name, tokenizer_path, tokenizer):
    word_embedding_model = Transformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path=tokenizer_path
    )

    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("✅ SentenceTransformer criado com sucesso.")
    return encoder

if __name__ == "__main__":
    # --- Parâmetros ---
    base_model_name = "alfaneo/jurisbert-base-portuguese-uncased"
    tokenizer_custom_path = "./results/tokenizer_custom.json"
    tokenizer_output_path = f"./results/tokenizer_merged_{base_model_name.replace('/', '_')}"
    encoder_output_path = f"./results/encoder_custom_{base_model_name.replace('/', '_')}"

    # --- Execução ---
    print("🔍 Lendo tokens customizados...")
    custom_tokens = carregar_tokens_customizados(tokenizer_custom_path)

    print("🛠️ Expandindo tokenizer base...")
    tokenizer = expandir_tokenizer(base_model_name, custom_tokens, tokenizer_output_path)

    print("🏗️ Construindo encoder...")
    encoder = construir_sentence_encoder(base_model_name, tokenizer_output_path, tokenizer)

    print("💾 Salvando modelo completo...")
    encoder.save(encoder_output_path)
    print(f"\n✅ Modelo completo salvo em: {encoder_output_path}")

    # --- Teste opcional ---
    frases_teste = [
        "A violência_doméstica ocorre na zona_rural. Tribunal TJES decidiu.",
        "Um exemplo sem tokens especiais.",
        "TJES violência_doméstica zona_rural"
    ]

    print("\n🔍 Tokens das frases de teste:")
    for frase in frases_teste:
        tokens = tokenizer.tokenize(frase)
        print(f"\nFrase: {frase}")
        print(f"Tokens: {tokens}")

    print("\n📈 Testando embeddings...")
    embeddings = encoder.encode(frases_teste, normalize_embeddings=True)
    for frase, embedding in zip(frases_teste, embeddings):
        print(f"\nFrase: {frase}")
        print(f"Embedding (dim={embedding.shape[0]}): {embedding[:10]}...")  # só os 10 primeiros
