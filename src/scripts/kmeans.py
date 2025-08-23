import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Baixar as stopwords do NLTK (executar apenas uma vez se não tiver)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def plot_clusters_2d(X, labels, embbeding_model_name, figure_name):
    """
    Projeta os embeddings em 2D (PCA ou t-SNE) e plota os clusters
    """
    print("Gerando projeção 2D dos embeddings...")

    # Primeiro reduz para 50D com PCA (mais rápido/estável para t-SNE)
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    # Depois projeta em 2D com t-SNE
    X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Visualização dos clusters (t-SNE 2D)")

    os.makedirs("./results/kmeans", exist_ok=True)
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    plt.savefig(f"./results/kmeans/{figure_name}_clusters_plot-{safe_model_name}.png", dpi=400)
    plt.close()

# ---------- Funções utilitárias ----------
def save_data(file_folder_path: str = './results/kmeans/found_categories.csv',
              found_categories: list = [],
              embbeding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Salva os dados em CSV
    """
    try:
        dataset = pd.DataFrame(found_categories)
        os.makedirs(os.path.dirname(file_folder_path), exist_ok=True)
        dataset.to_csv(path_or_buf=file_folder_path, index=False, sep=',', encoding='utf-8')
    except Exception as e:
        raise Exception(f"Erro ao salvar os dados. Exception: {e}\n")


def find_optimal_k_alternatives(X, max_k: int = 15, embbeding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Calcula e plota o Método do Cotovelo (WCSS) e o Índice Davies-Bouldin
    para ajudar a encontrar o número ideal de clusters.
    """
    k_range = range(2, max_k + 1)

    # --- 1. Método do Cotovelo (Elbow Method) ---
    wcss = []
    print("Calculando o Método Elbow ...")
    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans_model.fit(X)
        wcss.append(kmeans_model.inertia_)  # inertia_ é o WCSS

    # Plotando o gráfico do Cotovelo
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Método do Elbow')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(k_range)
    plt.grid(False)

    # --- 2. Índice Davies-Bouldin ---
    db_scores = []
    print("Calculando o Índice Davies-Bouldin...")
    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans_model.fit_predict(X)
        db_score = davies_bouldin_score(X, labels)
        db_scores.append(db_score)

    # Encontrando o k com o menor score
    optimal_k_db = k_range[np.argmin(db_scores)]
    print(f"\nO k ideal pelo Índice Davies-Bouldin é: {optimal_k_db} (menor valor)")

    # Plotando o gráfico Davies-Bouldin
    plt.subplot(1, 2, 2)
    plt.plot(k_range, db_scores, marker='o', linestyle='--', color='r')
    plt.title('Índice Davies-Bouldin')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.xticks(k_range)
    plt.grid(False)

    os.makedirs("./results/kmeans", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f"./results/kmeans/silhoet-method_using_sentences-embbeding-{str(embbeding_model_name).replace('/', '-')}_optimal_k-{optimal_k_db}.png",            
        dpi=400
    )
    plt.close()
    return optimal_k_db


# ---------- Função principal ----------
def main(file_path: str,
         num_words_to_analyse: int = 10,
         num_frequent_sentences: int = 10,
         text_column: str = "generated_sentence",
         select_num_clusters: str = "auto",
         num_clusters: int = 50,
         figure_name: str = "analise_k_means",
         embbeding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Executa pipeline de KMeans sobre embeddings de sentenças
    """
    # 1. Ler dataset
    print(f"Lendo dataset de {file_path} ...")
    dataset = pd.read_csv(filepath_or_buffer=file_path)

    if text_column not in dataset.columns:
        raise Exception(f"A coluna '{text_column}' não existe no dataset.")

    sentences = dataset[text_column].dropna().astype(str).tolist()
    print(f"Total de sentenças: {len(sentences)}")

    # 2. Criar embeddings
    print(f"Carregando modelo de embeddings: {embbeding_model_name} ...")
    model = SentenceTransformer(embbeding_model_name)
    X = model.encode(sentences, show_progress_bar=True)

    # 3. Encontrar número ótimo de clusters
    if select_num_clusters == "auto":
        num_clusters = find_optimal_k_alternatives(X, max_k=15, embbeding_model_name=embbeding_model_name)

    print(f"Treinando KMeans com k={num_clusters} ...")
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans_model.fit_predict(X)

    # --- NOVO: gerar plot 2D dos clusters
    plot_clusters_2d(X, labels, embbeding_model_name, figure_name)

    # 4. Salvar resultados
    dataset["cluster"] = labels
    save_path = f"./results/kmeans/{figure_name}_results.csv"
    save_data(file_folder_path=save_path,
              found_categories=dataset,
              embbeding_model_name=embbeding_model_name)
    print(f"Resultados salvos em: {save_path}")

    return dataset


# ---------- Execução ----------
if __name__ == '__main__':
    llm_model_name = "gemma3:1b"
    input_file_folder_path = (
        'processed_data_llm_gemma3:1b_analysis_multiclasses_only-user-description_using_new_clusters_names_with_dynamic-list-of-categories.csv'
    )
    text_column = "generated_sentence"
    num_clusters = 50
    embbeding_model_name = "neuralmind/bert-base-portuguese-cased"

    main(file_path=f'./results/{input_file_folder_path}',
         num_words_to_analyse=10,
         num_frequent_sentences=10,
         text_column=text_column,
         select_num_clusters="auto",   # ou "manual"
         num_clusters=num_clusters,
         figure_name=f"analise_k_means-clusters_using_sentence-embbeding_cluster_using-{llm_model_name}",
         embbeding_model_name=str(embbeding_model_name))
