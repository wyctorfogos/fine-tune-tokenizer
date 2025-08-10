import pandas as pd
from itertools import combinations
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from utils.load_dataset import load_data
from tqdm import tqdm # NOVO: Adicionada biblioteca tqdm para a barra de progresso

# =============================================================================
# BLOCO DE CONFIGURAÇÃO
# (Altere os caminhos e parâmetros aqui)
# =============================================================================

# -- Caminhos dos Dados e Modelos --
INPUT_DATA_PATH = "data/processed_data_new_dataframe_complete_sentences-using_faker+filtro-de-sentencas_multiclusters.csv"
PAIRS_OUTPUT_CSV_PATH = "data/pares_treinamento_positivos.csv"
BASE_MODEL_PATH = "./results/encoder_custom_neuralmind_bert-base-portuguese-cased"
FINETUNED_MODEL_OUTPUT_PATH = "./results/modelo-chamados-finetuned"

# -- Parâmetros de Treinamento --
TRAIN_BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# =============================================================================
# PARTE 1: PREPARAÇÃO E SALVAMENTO DOS PARES DE DADOS
# =============================================================================

def preparar_dados_de_treinamento():
    """
    Carrega os dados originais, gera pares de sentenças positivas 
    (da mesma categoria) e salva em um arquivo CSV.
    """
    print("="*50)
    print("INICIANDO PREPARAÇÃO DE DADOS")
    print("="*50)

    if os.path.exists(PAIRS_OUTPUT_CSV_PATH):
        print(f"Arquivo de pares já encontrado em '{PAIRS_OUTPUT_CSV_PATH}'. Pulando geração.")
        return

    print("Arquivo de pares não encontrado. Gerando agora...")
    
    dataset = load_data(file_path=INPUT_DATA_PATH, separator=",")
    frases_dict = pd.DataFrame({
        "generated_text": dataset["generated_sentence"],
        "category": dataset["categoria"]
    })

    print("Gerando pares de sentenças da mesma categoria...")
    pares_positivos = []
    grouped = frases_dict.groupby('category')

    # MUDANÇA: Adicionado tqdm() para mostrar o progresso da geração de pares.
    # A descrição (desc) ajuda a saber o que a barra de progresso está medindo.
    for category, group in tqdm(grouped, desc="Processando categorias"):
        sentences = group['generated_text'].tolist()
        if len(sentences) >= 2:
            for pair in combinations(sentences, 2):
                pares_positivos.append({
                    "sentence1": pair[0],
                    "sentence2": pair[1],
                    "category": category
                })

    if not pares_positivos:
        raise ValueError("Nenhum par positivo foi gerado. Verifique se suas categorias têm pelo menos 2 exemplos cada.")

    df_pares = pd.DataFrame(pares_positivos)
    
    print(f"Salvando {len(df_pares)} pares de treinamento em '{PAIRS_OUTPUT_CSV_PATH}'...")
    df_pares.to_csv(PAIRS_OUTPUT_CSV_PATH, index=False)
    print("Preparação de dados concluída!")


# =============================================================================
# PARTE 2: FINE-TUNING DO MODELO
# =============================================================================

def treinar_modelo():
    """
    Carrega o modelo base, os dados de treinamento e executa o fine-tuning
    com otimizações para pouca VRAM.
    """
    print("\n" + "="*50)
    print("INICIANDO FINE-TUNING DO MODELO")
    print("="*50)

    print(f"Carregando modelo base de '{BASE_MODEL_PATH}'...")
    model = SentenceTransformer(BASE_MODEL_PATH)

    print(f"Carregando pares de treinamento de '{PAIRS_OUTPUT_CSV_PATH}'...")
    df_pares = pd.read_csv(PAIRS_OUTPUT_CSV_PATH)

    # Limitar a quantidade de pares
    df_pares = df_pares.sample(n=1000)

    train_examples = [
        InputExample(texts=[row['sentence1'], row['sentence2']])
        for _, row in df_pares.iterrows()
    ]

    print(f"Configurando DataLoader com batch size = {TRAIN_BATCH_SIZE}")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    num_update_steps_per_epoch = len(train_dataloader) // ACCUMULATION_STEPS
    warmup_steps = int(num_update_steps_per_epoch * NUM_EPOCHS * 0.1)

    print("\n--- INICIANDO TREINAMENTO ---")
    print(f"Batch Size Efetivo: {TRAIN_BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"Épocas: {NUM_EPOCHS}")
    print(f"Passos de Aquecimento (Warmup): {warmup_steps}")
    print(f"Salvando modelo final em: '{FINETUNED_MODEL_OUTPUT_PATH}'")

    # Com a biblioteca atualizada, este comando funcionará sem erros.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=FINETUNED_MODEL_OUTPUT_PATH,
        show_progress_bar=True,
        optimizer_params={'lr': LEARNING_RATE},
        weight_decay=0.01
    )

    print("\n--- TREINAMENTO CONCLUÍDO ---")
    print(f"Modelo salvo com sucesso em '{FINETUNED_MODEL_OUTPUT_PATH}'")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    preparar_dados_de_treinamento()
    treinar_modelo()