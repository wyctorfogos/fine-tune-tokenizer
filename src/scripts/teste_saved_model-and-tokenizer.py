import os
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    # --- 1. Definição dos Caminhos ---
    # O nome do modelo BERT que foi a base de tudo
    base_model_name = "bert-base-multilingual-cased"
    
    # O diretório onde você salvou o tokenizer expandido (com vocab.txt, etc.)
    expanded_tokenizer_path = "./results/tokenizer_merged"
    
    print(f"Carregando encoder a partir do modelo base '{base_model_name}' e tokenizer de '{expanded_tokenizer_path}'...")

    # --- 2. Recriação do Encoder ---
    
    # Passo A: Carregar a camada Transformer, especificando o modelo base e o nosso tokenizer customizado.
    # O SentenceTransformer carregará os pesos do `base_model_name` e as regras de tokenização do `expanded_tokenizer_path`.
    word_embedding_model = models.Transformer(
        model_name_or_path=base_model_name,
        tokenizer_name_or_path=expanded_tokenizer_path
    )

    # Passo B: Redimensionar a camada de embeddings para compatibilidade.
    # Este passo é crucial para alinhar o modelo (com seu vocabulário original) ao tokenizer (com o vocabulário expandido).
    # Precisamos carregar o tokenizer para saber seu novo tamanho.
    tokenizer = AutoTokenizer.from_pretrained(expanded_tokenizer_path)
    word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    
    # Passo C: Criar a camada de Pooling para gerar um embedding de frase único.
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    # Passo D: Montar o objeto SentenceTransformer (o nosso "encoder" final).
    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    print("\n✅ Encoder carregado com sucesso!")
    print(f"   Rodando no dispositivo: {encoder.device}")

    # --- 3. Teste do Encoder ---
    print("\n--- Testando o encoder com frases de exemplo ---")
    
    frases_para_testar = [
        "A violência_doméstica é um crime grave.",
        "O tribunal de justiça TJES publicou o edital.",
        "A agricultura na zona_rural foi afetada pela seca.",
        "Um texto normal sem tokens customizados.",
        "violência_doméstica zona_rural TJES" # Apenas os tokens
    ]
    
    # O método .encode() é otimizado para processar listas de frases.
    embeddings = encoder.encode(frases_para_testar, show_progress_bar=True)
    
    # --- 4. Análise dos Resultados ---
    for frase, embedding in zip(frases_para_testar, embeddings):
        print(f"\nFrase: '{frase}'")
        # Mostrando apenas os 5 primeiros valores do vetor para não poluir a tela.
        print(f"   Vetor de Embedding (primeiros 5 valores): {embedding[:10]}")
        print(f"   Dimensão do Vetor: {embedding.shape}")