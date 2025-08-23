import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

import umap
import hdbscan

def save_data(file_folder_path: str, dataset: pd.DataFrame):
    """
    Salva dataset em CSV
    """
    os.makedirs(os.path.dirname(file_folder_path), exist_ok=True)
    dataset.to_csv(file_folder_path, index=False, sep=',', encoding='utf-8')
    print(f"Resultados salvos em: {file_folder_path}")


def plot_clusters_umap(X, labels, embbeding_model_name, figure_name):
    """
    Projeta embeddings em 2D com UMAP e plota clusters de HDBSCAN
    """
    print("Gerando projeção 2D com UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Visualização dos clusters (UMAP + HDBSCAN)")

    os.makedirs("./results/hdbscan", exist_ok=True)
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    plt.savefig(f"./results/hdbscan/{figure_name}_clusters-UMAP-HDBSCAN-{safe_model_name}.png", dpi=400)
    plt.close()
    print("Plot salvo.")

import os
import numpy as np
import pandas as pd
import hdbscan
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt

def main_hdbscan(file_path: str,
                 text_column: str = "generated_sentence",
                 embbeding_model_name: str = "intfloat/multilingual-e5-base",
                 figure_name: str = "analise_hdbscan"):
    """
    Executa pipeline com HDBSCAN + UMAP e salva/carrega embeddings
    """
    print(f"Lendo dataset de {file_path} ...")
    dataset = pd.read_csv(filepath_or_buffer=file_path)

    if text_column not in dataset.columns:
        raise Exception(f"A coluna '{text_column}' não existe no dataset.")

    sentences = dataset[text_column].dropna().astype(str).tolist()
    print(f"Total de sentenças: {len(sentences)}")

    # ---- Nome seguro para salvar ----
    os.makedirs("./results/embeddings", exist_ok=True)
    safe_model_name = str(embbeding_model_name).replace("/", "-")
    emb_path = f"./results/embeddings/{figure_name}_embeddings-{safe_model_name}.npy"

    # ---- Carregar ou gerar embeddings ----
    if os.path.exists(emb_path):
        print(f"Carregando embeddings de {emb_path} ...")
        X = np.load(emb_path)
    else:
        print(f"Gerando embeddings com modelo {embbeding_model_name} ...")
        model = SentenceTransformer(embbeding_model_name)
        X = model.encode(sentences, show_progress_bar=True)
        np.save(emb_path, X)
        print(f"Embeddings salvos em {emb_path}")

    # ---- HDBSCAN ----
    print("Rodando HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(X)

    dataset["cluster"] = labels

    # ---- Salvar resultados ----
    save_path = f"./results/hdbscan/{figure_name}_results.csv"
    os.makedirs("./results/hdbscan", exist_ok=True)
    dataset.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Resultados salvos em {save_path}")

    # ---- Plot 2D ----
    plot_clusters_umap(X, labels, embbeding_model_name, figure_name)

    # ---- Informações ----
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Sentenças marcadas como ruído: {n_noise}")

    return dataset


if __name__ == '__main__':
    llm_model_name = "gemma3:1b"
    input_file_folder_path = (
        'processed_data_llm_gemma3:1b_analysis_multiclasses_only-user-description_using_new_clusters_names_with_dynamic-list-of-categories.csv'
    )
    text_column = "generated_sentence"
    embbeding_model_name = "intfloat/multilingual-e5-base"

    main_hdbscan(file_path=f'./results/{input_file_folder_path}',
                 text_column=text_column,
                 embbeding_model_name=embbeding_model_name,
                 figure_name=f"clusters_HDBSCAN_using-{llm_model_name}")
