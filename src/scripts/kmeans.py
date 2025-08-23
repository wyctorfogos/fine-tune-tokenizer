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
from matplotlib.patches import Patch

# Baixar as stopwords do NLTK (executar apenas uma vez se não tiver)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def plot_clusters_2d(X, labels, categorias, embbeding_model_name, figure_name):
    """
    Projeta os embeddings em 2D (PCA -> t-SNE) e plota os clusters
    comparando a clusterização com as categorias reais.
    A legenda mostra 'C1', 'C2'... para clusters e nomes reais para categorias.
    """
    print("Gerando projeção 2D dos embeddings...")

    # =========================
    # Redução dimensional
    # =========================
    n_pca = min(50, X.shape[1])  
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

    # =========================
    # Preparar categorias
    # =========================
    categorias = pd.Series(categorias)  # garante formato Series
    categorias_encoded = categorias.astype("category").cat.codes
    categorias_labels = dict(enumerate(categorias.astype("category").cat.categories))

    # =========================
    # Plotagem
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Plot clusters (C1, C2...)
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                               cmap="tab20", alpha=0.7, s=20)

    # Legenda clusters
    cluster_colors = scatter1.cmap(scatter1.norm(range(len(set(labels)))))
    cluster_legend = [Patch(color=cluster_colors[i], label=f"C{i+1}") for i in range(len(set(labels)))]
    axes[0].legend(handles=cluster_legend, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].set_title("Clusters (t-SNE 2D)")

    # ---- Plot categorias reais (nomes)
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=categorias_encoded,
                               cmap="tab20", alpha=0.7, s=20)

    # Legenda categorias
    cat_colors = scatter2.cmap(scatter2.norm(range(len(categorias_labels))))
    cat_legend = [Patch(color=cat_colors[i], label=nome) for i, nome in categorias_labels.items()]
    axes[1].legend(handles=cat_legend, title="Categorias", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].set_title("Categorias (t-SNE 2D)")

    plt.tight_layout()

    # =========================
    # Salvamento da figura
    # =========================
    os.makedirs("./results/kmeans", exist_ok=True)
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    image_path = f"./results/kmeans/{figure_name}_clusters-e-categorias_plot-{safe_model_name}.png"

    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura salva em: {image_path}")
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

    # 2. Verificar se já existem embeddings salvos
    os.makedirs("./results/kmeans", exist_ok=True)
    emb_path = f"./results/kmeans/{figure_name}_embeddings.npy"
    sent_path = f"./results/kmeans/{figure_name}_sentences.csv"

    if os.path.exists(emb_path) and os.path.exists(sent_path):
        print(f"Carregando embeddings salvos de {emb_path} ...")
        X = np.load(emb_path)
        sentences_df = pd.read_csv(sent_path)
        if len(sentences_df) != len(sentences):
            raise ValueError("O número de sentenças mudou. Recalcule os embeddings.")
    else:
        print(f"Gerando embeddings com modelo: {embbeding_model_name} ...")
        model = SentenceTransformer(embbeding_model_name)
        X = model.encode(sentences, show_progress_bar=True)

        # salvar embeddings e sentenças
        np.save(emb_path, X)
        pd.DataFrame({"sentence": sentences}).to_csv(sent_path, index=False, encoding="utf-8")
        print(f"Embeddings salvos em: {emb_path}")
        print(f"Sentenças salvas em: {sent_path}")

    # 3. Encontrar número ótimo de clusters
    if select_num_clusters == "auto":
        num_clusters = find_optimal_k_alternatives(X, max_k=15, embbeding_model_name=embbeding_model_name)

    print(f"Treinando KMeans com k={num_clusters} ...")
    
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans_model.fit_predict(X)
    dataset["cluster"] = labels
    dataset["cluster_name"] = ["C" + str(i+1) for i in labels]
    # --- Plot 2D
    if "categoria" in dataset.columns:
        plot_clusters_2d(X, dataset["cluster"], dataset["categoria"], embbeding_model_name, figure_name)

    # 4. Salvar resultados (sentenças + clusters + categorias)
    save_path = f"./results/kmeans/{figure_name}_results.csv"
    dataset.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Resultados salvos em: {save_path}")

    return dataset



# ---------- Execução ----------
if __name__ == '__main__':
    llm_model_name = "gemma3:1b"
    input_file_folder_path = (
        "processed_data_llm_gemma3:1b_analysis_multiclasses_only-user-description_using_new_clusters_names.csv" #'processed_data_llm_gemma3:1b_analysis_multiclasses_only-user-description_using_new_clusters_names_with_dynamic-list-of-categories.csv'
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
