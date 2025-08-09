import os
import json
import numpy as np
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

    # Tokens especiais padrão
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # Filtrar e validar tokens
    novos_tokens = list({
        t.strip()
        for t in custom_tokens
        if isinstance(t, str) and t.strip() and t.strip() not in special_tokens and t.strip() not in tokenizer.get_vocab()
    })

    adicionados = tokenizer.add_tokens(novos_tokens)
    print(f"✅ {adicionados} tokens adicionados ao tokenizer.")

    # Salvar tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"📦 Tokenizer salvo em: {output_path}")
    return tokenizer

def construir_sentence_encoder(model_name, tokenizer_path, tokenizer):
    word_embedding_model = Transformer(
        model_name_or_path=model_name,
        tokenizer_name_or_path=tokenizer_path
    )

    # Ajusta matriz de embeddings para novos tokens
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("✅ SentenceTransformer criado com sucesso.")
    return encoder

def comparar_embeddings(encoder_original, encoder_expandido, frases):
    print("\n📊 Comparando projeções...")
    emb_orig = encoder_original.encode(frases, normalize_embeddings=True)
    emb_exp = encoder_expandido.encode(frases, normalize_embeddings=True)

    for i, frase in enumerate(frases):
        cos_sim = np.dot(emb_orig[i], emb_exp[i])
        diff_norm = np.linalg.norm(emb_orig[i] - emb_exp[i])
        print(f"\nFrase: {frase}")
        print(f"Similaridade coseno: {cos_sim:.4f}")
        print(f"Norma da diferença: {diff_norm:.4f}")

if __name__ == "__main__":
    # --- Parâmetros ---
    base_model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer_custom_path = "./results/tokenizer_custom.json"
    tokenizer_output_path = f"./results/tokenizer_merged_{base_model_name.replace('/', '_')}"
    encoder_output_path = f"./results/encoder_custom_{base_model_name.replace('/', '_')}"

    # --- Leitura dos tokens customizados ---
    print("🔍 Lendo tokens customizados...")
    custom_tokens = carregar_tokens_customizados(tokenizer_custom_path)

    # --- Modelo original para comparação ---
    print("📥 Carregando encoder original...")
    encoder_original = SentenceTransformer(base_model_name)

    # --- Expansão do tokenizer ---
    print("🛠️ Expandindo tokenizer base...")
    tokenizer = expandir_tokenizer(base_model_name, custom_tokens, tokenizer_output_path)

    # --- Construção do encoder expandido ---
    print("🏗️ Construindo encoder expandido...")
    encoder_expandido = construir_sentence_encoder(base_model_name, tokenizer_output_path, tokenizer)

    # Salvar encoder + tokenizer juntos
    encoder_expandido.save(encoder_output_path)
    tokenizer.save_pretrained(encoder_output_path)
    print(f"\n✅ Modelo completo salvo em: {encoder_output_path}")

    # --- Teste com frases ---
    frases_teste = [
        "A violência_doméstica ocorre na zona_rural. Tribunal TJES decidiu.",
        "Um exemplo sem tokens especiais.",
        "TJES violência_doméstica zona_rural"
    ]

    print("\n🔍 Tokens das frases de teste (expandido):")
    for frase in frases_teste:
        tokens = tokenizer.tokenize(frase)
        print(f"\nFrase: {frase}")
        print(f"Tokens: {tokens}")

    # --- Comparar embeddings ---
    comparar_embeddings(encoder_original, encoder_expandido, frases_teste)