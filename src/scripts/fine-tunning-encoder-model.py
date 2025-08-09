import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# =========================
# Carregar o modelo com vocabulário expandido
# =========================
model_path = "./results/encoder_custom_neuralmind_bert-base-portuguese-cased"
output_model_path = "./results/meu-modelo-finetuned-com-acumulacao"
model = SentenceTransformer(model_path)

# =========================
# 1. Carregar os dados de treinamento preparados
# =========================
try:
    df_pares = pd.read_parquet("./data/pares_treinamento_positivos.parquet")
except (FileNotFoundError, ImportError):
    print("Arquivo Parquet não encontrado, tentando carregar CSV...")
    df_pares = pd.read_csv("./data/pares_treinamento_positivos.csv")

print(f"Carregados {len(df_pares)} pares para o treinamento.")

# Converte o DataFrame para o formato InputExample
train_examples = [
    InputExample(texts=[row['sentence1'], row['sentence2']])
    for index, row in df_pares.iterrows()
]

# =========================
# 2. Configurar DataLoader e Loss Function
# =========================
# **AJUSTE PARA POUCA VRAM**
# Comece com um batch_size pequeno. 4, 2 ou até 1 se necessário.
train_batch_size = 4 

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size, persistent_workers=True, num_workers=3)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# =========================
# 3. Executar o Fine-Tuning com Acumulação de Gradientes
# =========================
num_epochs = 3
# Calcule o número de passos de acumulação para simular um batch size maior.
# Ex: Simular um batch de 32 (4 * 8)
accumulation_steps = 8 

warmup_steps = int(len(train_dataloader) // accumulation_steps * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_model_path,
    show_progress_bar=True,
    # **A MÁGICA ACONTECE AQUI**
    optimizer_params={'lr': 2e-5},
    weight_decay=0.01,
    gradient_accumulation_steps=accumulation_steps
)