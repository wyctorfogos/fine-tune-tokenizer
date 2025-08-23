import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from matplotlib.patches import Patch
from utils.request_to_ollama import request_embedding_from_ollama

# Baixar stopwords se necessário
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==============================
# Funções auxiliares
# ==============================

def preprocess_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stopwords.words("portuguese") and len(t) > 2]


def filter_generated_response(generated_sentence:str=None):
    '''
        Filtra as sentenças geradas por LLM
    '''
    try:
        if "</think>" in generated_sentence:
            after_think = generated_sentence.split("</think>", 1)[1].strip()
            print("✅ Extracted text after </think>:\n")
            # Retorno da sentença filtrada
            return after_think
        else:
            print("❌ No </think> found in text.")
            after_think = generated_sentence
        # Retorno da sentença filtrada
        return after_think

    except Exception as e:
        raise ValueError(f"Erro ao realizar a chamada dos dados:{e}\n")
    

def plot_clusters_2d(X, labels, categorias, embbeding_model_name, figure_name, cluster_names):
    print("Gerando projeção 2D dos embeddings...")

    n_pca = min(50, X.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

    categorias = pd.Series(categorias)
    categorias_encoded = categorias.astype("category").cat.codes
    categorias_labels = dict(enumerate(categorias.astype("category").cat.categories))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Plot clusters (C1, C2...)
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=20)
    cluster_colors = scatter1.cmap(scatter1.norm(range(len(set(labels)))))
    cluster_legend = [Patch(color=cluster_colors[i], label=f"C{i+1} - {cluster_names.get(i,'')}") for i in range(len(set(labels)))]
    axes[0].legend(handles=cluster_legend, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].set_title("Clusters (t-SNE 2D)")

    # ---- Plot categorias reais (nomes)
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=categorias_encoded, cmap="tab20", alpha=0.7, s=20)
    cat_colors = scatter2.cmap(scatter2.norm(range(len(categorias_labels))))
    cat_legend = [Patch(color=cat_colors[i], label=nome) for i, nome in categorias_labels.items()]
    axes[1].legend(handles=cat_legend, title="Categorias", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].set_title("Categorias (t-SNE 2D)")

    plt.tight_layout()
    os.makedirs("./results/kmeans", exist_ok=True)
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    image_path = f"./results/kmeans/{figure_name}_clusters-e-categorias_plot-{safe_model_name}.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura salva em: {image_path}")


def save_data(file_folder_path: str, found_categories: pd.DataFrame):
    try:
        os.makedirs(os.path.dirname(file_folder_path), exist_ok=True)
        found_categories.to_csv(path_or_buf=file_folder_path, index=False, sep=',', encoding='utf-8')
    except Exception as e:
        raise Exception(f"Erro ao salvar os dados. Exception: {e}\n")


def find_optimal_k_alternatives(X, max_k: int = 15, embbeding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    k_range = range(2, max_k + 1)

    wcss = []
    print("Calculando o Método Elbow ...")
    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans_model.fit(X)
        wcss.append(kmeans_model.inertia_)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Método do Elbow')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(k_range)

    db_scores = []
    print("Calculando o Índice Davies-Bouldin...")
    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans_model.fit_predict(X)
        db_score = davies_bouldin_score(X, labels)
        db_scores.append(db_score)

    optimal_k_db = k_range[np.argmin(db_scores)]
    print(f"\nO k ideal pelo Índice Davies-Bouldin é: {optimal_k_db} (menor valor)")

    plt.subplot(1, 2, 2)
    plt.plot(k_range, db_scores, marker='o', linestyle='--', color='r')
    plt.title('Índice Davies-Bouldin')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Davies-Bouldin Score')
    plt.xticks(k_range)

    os.makedirs("./results/kmeans", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"./results/kmeans/silhoet-method_using_sentences-embbeding-{str(embbeding_model_name).replace('/', '-')}_optimal_k-{optimal_k_db}.png", dpi=400)
    plt.close()
    return optimal_k_db


# ==============================
# Função principal
# ==============================

def main(file_path: str,
         text_column: str = "generated_sentence",
         select_num_clusters: str = "auto",
         num_clusters: int = 50,
         num_of_sentences_clostest_to_centroid:int=10,
         figure_name: str = "analise_k_means",
         embbeding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
         llm_model_name: str = "neuralmind/bert-base-portuguese-cased"):

    print(f"Lendo dataset de {file_path} ...")
    dataset = pd.read_csv(filepath_or_buffer=file_path)

    if text_column not in dataset.columns:
        raise Exception(f"A coluna '{text_column}' não existe no dataset.")

    sentences = dataset[text_column].dropna().astype(str).tolist()
    print(f"Total de sentenças: {len(sentences)}")

    # Embeddings
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    emb_path = f"./results/kmeans/embeddings_{safe_model_name}.npy"

    if os.path.exists(emb_path):
        print("Carregando embeddings salvos...")
        X = np.load(emb_path)
    else:
        print(f"Gerando embeddings com {embbeding_model_name} ...")
        model = SentenceTransformer(embbeding_model_name)
        X = model.encode(sentences, show_progress_bar=True)
        np.save(emb_path, X)
        print(f"Embeddings salvos em: {emb_path}")

    # Número ótimo de clusters
    if select_num_clusters == "auto":
        num_clusters = find_optimal_k_alternatives(X, max_k=15, embbeding_model_name=embbeding_model_name)

    print(f"Treinando KMeans com k={num_clusters} ...")
    kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans_model.fit_predict(X)
    dataset["cluster"] = labels

    # Nomear clusters via LLM
    cluster_names = {}
    for i in range(num_clusters):
        cluster_data = dataset[dataset["cluster"] == i]
        all_tokens = []
        for sentence in cluster_data[text_column].tolist():
            all_tokens.extend(preprocess_text(sentence))
        most_common_terms = [w for w, _ in Counter(all_tokens).most_common(num_of_sentences_clostest_to_centroid)]

        prompt_message = f"""
        Você é um assistente virtual especializado em sistemas de processo judicial eletrônico (como PJe, e-SAJ, Projudi). Analise os seguintes dados de um cluster de tickets de suporte:
        - Exemplos de sentenças centrais: {cluster_data[text_column].head(num_of_sentences_clostest_to_centroid).tolist()}

        Sua tarefa:
        1. Sugira um **nome de categoria** (em poucas palavras) que melhor represente esse cluster.
        2. Se já existir uma categoria parecida na lista anterior, refine o nome para diferenciá-la (ex.: "Acesso PJe - Senha", "Acesso PJe - Token").
        3. Retorne apenas o nome, nada mais.
        4. Antes de sugerir uma categoria ou subcategoria nova, verifique se ela já existe ou se tem termos parecidos na lista anterior:
                - Categorias já registradas: {list(cluster_names)}
        """

        try:
            llm_response = request_embedding_from_ollama(prompt=prompt_message, sentence="", model_name=llm_model_name)
            cluster_name = filter_generated_response(llm_response)
        except Exception as e:
            print(f"⚠️ Erro ao gerar nome da categoria: {e}")
            cluster_name = f"C{i+1}"

        cluster_name = str(cluster_name).strip()
        cluster_name_norm = re.sub(r'\s+', ' ', cluster_name).lower()
        cluster_names[i] = cluster_name_norm

        dataset.loc[dataset['cluster'] == i, 'cluster_category'] = cluster_name
        print(f"Cluster {i}: {cluster_name}")

    # Plot com nomes sugeridos
    plot_clusters_2d(X, dataset["cluster"], dataset.get("categoria", pd.Series(["NA"]*len(dataset))), embbeding_model_name, figure_name, cluster_names)

    # Salvar resultados
    save_path = f"./results/kmeans/{figure_name}_results.csv"
    save_data(file_folder_path=save_path, found_categories=dataset)
    print(f"Resultados salvos em: {save_path}")

    return dataset


# ==============================
# Execução
# ==============================
if __name__ == '__main__':
    llm_model_name = "qwen3:latest" # "deepseek-r1:7b" # "gemma3:4b" # "qwen3:latest" # "qwen3:0.6b"
    input_file_folder_path = (
        "processed_data_llm_gemma3:1b_analysis_multiclasses_only-user-description_using_new_clusters_names.csv"
    )
    text_column = "generated_sentence"
    num_clusters = 50
    embbeding_model_name = "neuralmind/bert-base-portuguese-cased"

    main(file_path=f'./results/{input_file_folder_path}',
         text_column=text_column,
         select_num_clusters="auto",
         num_clusters=num_clusters,
         num_of_sentences_clostest_to_centroid=25,
         figure_name=f"analise_k_means-clusters_using_sentence-embbeding_cluster_using-{llm_model_name}",
         embbeding_model_name=str(embbeding_model_name),
         llm_model_name=llm_model_name)
