import pandas as pd
from itertools import combinations
import os
import mlflow
import sys
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# Adicionando o caminho para o módulo de utilitários
# Certifique-se de que o caminho relativo está correto para sua estrutura de pastas
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
    from load_dataset import load_data
except ImportError:
    print("Aviso: Módulo 'load_data' não encontrado. Usando um placeholder.")
    # Fallback caso o módulo não seja encontrado, para que o script não quebre imediatamente.
    def load_data(file_path, standard_sep):
        print(f"Placeholder: Carregando dados de {file_path}")
        data = {
            'generated_sentence': [f'Frase de exemplo {i} da categoria A' for i in range(10)] + 
                                  [f'Frase de exemplo {i} da categoria B' for i in range(10)],
            'categoria': ['A'] * 10 + ['B'] * 10
        }
        return pd.DataFrame(data)

# =============================================================================
# BLOCO DE CONFIGURAÇÃO
# =============================================================================
INPUT_DATA_PATH = "./results/processed_data_new_dataframe_complete_sentences-complete_description-using_faker+filtro-de-sentencas_multiclusters.csv"
PAIRS_OUTPUT_CSV_PATH = "./results/pares_treinamento_positivos_amostra.csv"
llm_model_name = "neuralmind/bert-base-portuguese-cased" # "sentence-transformers/all-MiniLM-L6-v2" #  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FINETUNED_MODEL_OUTPUT_PATH = f"./data/encoder/encoder-finetuned-{llm_model_name.replace('/', '_')}"
mlflow_experiment_name = "FINE-TUNING-ENCODER-MODELS"

# -- Parâmetros de Treinamento --
TRAIN_BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 500
LEARNING_RATE = 5e-4
NUM_SAMPLES_TO_TRAIN = 1000

# =============================================================================
# PARTE 1: PREPARAÇÃO E SALVAMENTO DOS PARES DE DADOS
# =============================================================================

def preparar_dados_de_treinamento():
    """
    Carrega os dados originais, gera pares de sentenças positivas,
    SELECIONA UMA AMOSTRA e salva em um arquivo CSV.
    """
    print("="*50)
    print("INICIANDO PREPARAÇÃO DE DADOS")
    print("="*50)

    if os.path.exists(PAIRS_OUTPUT_CSV_PATH):
        print(f"Arquivo de pares de amostra já encontrado em '{PAIRS_OUTPUT_CSV_PATH}'. Pulando geração.")
        return

    print("Arquivo de pares não encontrado. Gerando agora...")

    dataset = load_data(file_path=INPUT_DATA_PATH, standard_sep=",")
    frases_dict = pd.DataFrame({
        "generated_text": dataset["generated_sentence"],
        "category": dataset["categoria"]
    })

    print("Gerando todos os pares de sentenças da mesma categoria...")
    pares_positivos = []
    grouped = frases_dict.groupby('category')

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

    df_pares_total = pd.DataFrame(pares_positivos)
    print(f"\nTotal de {len(df_pares_total)} pares gerados.")

    print(f"Selecionando uma amostra aleatória de {NUM_SAMPLES_TO_TRAIN} pares antes de salvar...")
    num_samples = min(NUM_SAMPLES_TO_TRAIN, len(df_pares_total))
    if num_samples < NUM_SAMPLES_TO_TRAIN:
        print(f"Aviso: O número total de pares gerados ({len(df_pares_total)}) é menor que o desejado ({NUM_SAMPLES_TO_TRAIN}). Usando todos os pares.")

    df_pares_amostra = df_pares_total.sample(n=num_samples, random_state=42)
    
    print(f"Salvando {len(df_pares_amostra)} pares de treinamento em '{PAIRS_OUTPUT_CSV_PATH}'...")
    df_pares_amostra.to_csv(PAIRS_OUTPUT_CSV_PATH, index=False)
    print("Preparação de dados concluída!")

# =============================================================================
# PARTE 2: FINE-TUNING DO MODELO
# =============================================================================

def treinar_modelo():
    """
    Carrega o modelo base e os dados de treinamento (que já são uma amostra)
    e executa o fine-tuning.
    """
    print("\n" + "="*50)
    print("INICIANDO FINE-TUNING DO MODELO")
    print("="*50)

    # Verifica se o modelo finetuned já existe.
    if os.path.exists(FINETUNED_MODEL_OUTPUT_PATH):
        print(f"Modelo finetuned já encontrado em '{FINETUNED_MODEL_OUTPUT_PATH}'. Pulando treinamento.")
        return

    print(f"Carregando modelo base do Hugging Face: '{llm_model_name}'...")
    model = SentenceTransformer(llm_model_name)

    print(f"Carregando pares de treinamento de '{PAIRS_OUTPUT_CSV_PATH}'...")
    df_pares = pd.read_csv(PAIRS_OUTPUT_CSV_PATH)
    print(f"{len(df_pares)} pares carregados para o treinamento.")

    train_examples = [
        InputExample(texts=[str(row['sentence1']), str(row['sentence2'])])
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

    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=f"FINETUNING ENCODER: {llm_model_name}"):
        mlflow.log_params({
            "model_name": llm_model_name,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "num_samples_to_train": NUM_SAMPLES_TO_TRAIN
        })

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=NUM_EPOCHS,
            output_path=FINETUNED_MODEL_OUTPUT_PATH,
            optimizer_params={'lr': LEARNING_RATE},
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )

        mlflow.log_artifact(FINETUNED_MODEL_OUTPUT_PATH)
        print("\n--- TREINAMENTO CONCLUÍDO ---")
        print(f"Modelo salvo com sucesso em '{FINETUNED_MODEL_OUTPUT_PATH}'")

def teste_encoder_treinado(senteca_teste:str):
    modelo_de_base = SentenceTransformer(llm_model_name)
    modelo_treinado = SentenceTransformer(model_name_or_path=FINETUNED_MODEL_OUTPUT_PATH)
    print(f"Modelo de base: {modelo_de_base.encode(sentences=senteca_teste)}\n")
    print(f"Modelo treinado: {modelo_treinado.encode(sentences=senteca_teste)}\n")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Preparação do dataset
    preparar_dados_de_treinamento()
    # Treino do encoder
    treinar_modelo()
    # Teste do modelo treinado
    teste_encoder_treinado(senteca_teste="violência doméstica na zona rural")