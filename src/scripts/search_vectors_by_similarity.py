import os
import faiss
import logging
import json
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import Union

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class FAISS():
    """
    Classe para gerenciar a indexação e busca no FAISS com embeddings de sentenças.
    Armazena índice + metadados em um único arquivo .pkl.
    """
    def __init__(self, filepath: str = None, column_name: str = "generated_sentence",
                 sentence_embedding_model_name: str = "neuralmind/bert-base-portuguese-cased"):
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_model = SentenceTransformer(model_name_or_path=self.sentence_embedding_model_name)
        self.obj_faiss = None
        self.filepath = filepath
        self.column_name = column_name
        self.dataset = None
        if filepath:  # só carrega dataset se foi passado
            self.dataset = self._load_data()
    
    def _create_index(self, D: int) -> None:
        try:
            self.obj_faiss = faiss.IndexFlatIP(D)  # Inner Product (cosine similarity após normalização)
            logging.info(f"Índice FAISS criado com dimensão D={D}")
        except Exception as e:
            logging.error(f"Erro ao criar o índice FAISS: {e}")

    def _add_to_index(self, vectors: np.ndarray) -> None:
        try:
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors)
            self.obj_faiss.add(vectors)
            logging.info(f"{self.obj_faiss.ntotal} vetores adicionados ao índice.")
        except Exception as e:
            logging.error(f"Erro ao adicionar vetores: {e}")

    def _search_index(self, vector_to_search: np.ndarray, k: int = 5):
        try:
            query_vector = np.ascontiguousarray(vector_to_search, dtype=np.float32)
            faiss.normalize_L2(query_vector)
            distances, indices = self.obj_faiss.search(query_vector, k=k)
            return distances, indices
        except Exception as e:
            logging.error(f"Erro durante a busca: {e}")
            return None, None

    def _load_data(self) -> Union[pd.DataFrame, None]:
        try:
            logging.info(f"Carregando dados de: {self.filepath}")
            df = pd.read_csv(self.filepath)
            if self.column_name not in df.columns:
                raise ValueError(f"A coluna '{self.column_name}' não existe no dataset. Colunas disponíveis: {list(df.columns)}")
            df.dropna(subset=[self.column_name], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Encontradas {len(df)} linhas válidas.")
            return df
        except Exception as e:
            logging.error(f"Erro ao carregar os dados: {e}")
            return None

    # --- Métodos Públicos ---
    
    def build_index(self) -> bool:
        """Gera embeddings e constrói o índice FAISS."""
        if self.dataset is None:
            logging.error("Dataset não carregado. Abortando construção do índice.")
            return False
            
        logging.info("Gerando embeddings para os documentos...")
        try:
            sentences = self.dataset[self.column_name].tolist()
            sentence_embeddings = self.embedding_model.encode(
                sentences, batch_size=64, show_progress_bar=True
            )
            D = sentence_embeddings.shape[1]
            self._create_index(D)
            self._add_to_index(sentence_embeddings)
            return self.obj_faiss.is_trained
        except Exception as e:
            logging.error(f"Falha ao construir o índice FAISS: {e}")
            return False

    def save_store(self, store_path: str) -> None:
        """Salva índice + metadados em um único arquivo .pkl"""
        if self.obj_faiss is None or self.dataset is None:
            logging.warning("Nada para salvar. Construa o índice primeiro.")
            return
        try:
            data = {
                "index": faiss.serialize_index(self.obj_faiss),
                "metadata": self.dataset.to_dict(orient="records")
            }
            with open(store_path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"Store salvo em: {store_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar store: {e}")

    def load_store(self, store_path: str) -> None:
        """Carrega índice + metadados de um único arquivo .pkl"""
        if not os.path.exists(store_path):
            logging.error(f"Store não encontrado em: {store_path}")
            return
        try:
            with open(store_path, "rb") as f:
                data = pickle.load(f)
            self.obj_faiss = faiss.deserialize_index(data["index"])
            self.dataset = pd.DataFrame(data["metadata"])
            logging.info(f"Store carregado com {len(self.dataset)} registros.")
        except Exception as e:
            logging.error(f"Erro ao carregar store: {e}")

    def search(self, query: str, k: int = 5) -> str:
        """
        Realiza busca semântica e retorna resultados em JSON.
        """
        if self.obj_faiss is None or not self.obj_faiss.is_trained:
            return json.dumps({"error": "Índice não construído."}, ensure_ascii=False)

        test_embedding = self.embedding_model.encode([query])
        distances, indices = self._search_index(test_embedding, k=k)
        
        results = []
        if indices is not None:
            for i, idx in enumerate(indices[0]):
                result_row = self.dataset.iloc[idx]
                results.append({
                    "rank": i+1,
                    "id_chamado": str(result_row.get("id_chamado", "")),
                    "prioridade do chamado":str(result_row.get("prioridade", "")),
                    "generated_sentence": str(result_row.get(self.column_name, "")),
                    "sugestao": str(result_row.get("llm_suggestion", "")),
                    "similaridade": float(distances[0][i])
                })
        
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)


def main():
    FILE_PATH = "./results/kmeans/analise_k_means-clusters_using_sentence-embbeding_cluster_using-gemma3:4b_results.csv"
    STORE_PATH = "./results/faiss_store.pkl"
    TEXT_COLUMN = "generated_sentence"
    K_RESULTS = 3
    TEXTO_DE_TESTE = "Erro ao tentar acessar um processo específico do TJE."
    
    faiss_system = FAISS(filepath=FILE_PATH, column_name=TEXT_COLUMN)
    
    if faiss_system.dataset is not None:
        if not os.path.exists(STORE_PATH):
            index_ready = faiss_system.build_index()
            if index_ready:
                faiss_system.save_store(STORE_PATH)
        else:
            faiss_system.load_store(STORE_PATH)
        
        json_results = faiss_system.search(query=TEXTO_DE_TESTE, k=K_RESULTS)
        print(json_results)


if __name__=="__main__":
    main()
