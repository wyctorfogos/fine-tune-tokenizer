import numpy as np
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from utils.load_dataset import load_data

# =========================
# Caminhos
# =========================
encoder_original_path = "neuralmind/bert-base-portuguese-cased" # "alfaneo/jurisbert-base-portuguese-uncased"
encoder_expanded_path = "./results/encoder_custom_neuralmind_bert-base-portuguese-cased" # "./results/finetuned-custom_neuralmind_bert-base-portuguese-cased" # "./results/encoder_custom_sentence-transformers_all-MiniLM-L6-v2" # "./results/encoder_custom_alfaneo_jurisbert-base-portuguese-uncased"

# =========================
# Carregar encoders
# =========================
print("Carregando encoder original...")
encoder_original = SentenceTransformer(encoder_original_path)

print("Carregando encoder expandido...")
encoder_expanded = SentenceTransformer(encoder_expanded_path)

# =========================
# Carregar dataset já processado
# =========================
dataset = load_data(
    file_path="data/processed_data_new_dataframe_complete_sentences-using_faker+filtro-de-sentencas_multiclusters.csv",
    separator=","
)

# Supondo que 'frases_dict' é o seu DataFrame
X = dataset['generated_sentence']
y = dataset['categoria']

# Instanciar o RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Resampling
X_resampled, y_resampled = rus.fit_resample(X.to_frame(), y)

# Criar um novo DataFrame com os dados reequilibrados para análise
frases_dict = pd.DataFrame({
    'generated_sentence': X_resampled['generated_sentence'],
    'categoria': y_resampled
})


# Verificar a contagem de classes após o undersampling
print("Distribuição das classes após o Undersampling:")
print(frases_dict['categoria'].value_counts())

## Carrega apenas 100 amostras do dataset
#dataset = dataset.sample(n=250)

# =========================
# Pré-processamento
# =========================
frases = frases_dict["generated_sentence"].tolist()
categorias = frases_dict["categoria"].tolist()

# Remove números
frases_sem_num = [re.sub(r"\d", "*", f) for f in frases]
frases = frases_sem_num

# =========================
# Gerar embeddings reais
# =========================
print("Gerando embeddings originais...")
embeddings_orig = encoder_original.encode(frases_sem_num, convert_to_numpy=True, show_progress_bar=True)

print("Gerando embeddings expandidos...")
embeddings_exp = encoder_expanded.encode(frases_sem_num, convert_to_numpy=True, show_progress_bar=True)

# =========================
# t-SNE
# =========================
all_embeddings = np.vstack([embeddings_orig, embeddings_exp])
labels_tipo = ["orig"] * len(frases) + ["expand"] * len(frases)

tsne = TSNE(n_components=3, perplexity=15, random_state=42, init="random", learning_rate="auto")
embeddings_2d = tsne.fit_transform(all_embeddings)

# =========================
# Cores por categoria
# =========================
categorias_unicas = list(set(categorias))
paleta = plt.cm.get_cmap("tab10", len(categorias_unicas))
cores_categorias = {cat: paleta(i) for i, cat in enumerate(categorias_unicas)}

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 8))

for i, frase in enumerate(frases):
    cat = categorias[i]
    cor = cores_categorias[cat]

    # Original
    plt.scatter(
        embeddings_2d[i, 0], embeddings_2d[i, 1],
        c=[cor], marker="o", edgecolor="k", s=70,
        label=f"{cat} (orig)" if f"{cat} (orig)" not in plt.gca().get_legend_handles_labels()[1] else ""
    )

    # Expandido
    plt.scatter(
        embeddings_2d[i + len(frases), 0], embeddings_2d[i + len(frases), 1],
        c=[cor], marker="s", edgecolor="k", s=70,
        label=f"{cat} (expand)" if f"{cat} (expand)" not in plt.gca().get_legend_handles_labels()[1] else ""
    )

    # Linha de ligação
    plt.plot(
        [embeddings_2d[i, 0], embeddings_2d[i + len(frases), 0]],
        [embeddings_2d[i, 1], embeddings_2d[i + len(frases), 1]],
        c=cor, linestyle="--", alpha=0.4
    )

plt.title("t-SNE: Original vs Expandido (cores por categoria)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.show()
plt.savefig("./results/comparacao_original_vs_custom_model-just-add-tokens.png", dpi=400)