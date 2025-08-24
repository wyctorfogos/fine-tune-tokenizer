import os
import faiss
import logging
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import Union, List, Dict

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class FAISS():
    """
    Classe para gerenciar a indexação e busca no FAISS com embeddings de sentenças.
    Retorna os resultados no formato JSON.
    """
    def __init__(self, filepath: str, column_name: str = "descrição",
                 sentence_embedding_model_name: str = "neuralmind/bert-base-portuguese-cased"):
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_model = SentenceTransformer(model_name_or_path=self.sentence_embedding_model_name)
        self.obj_faiss = None
        self.filepath = filepath
        self.column_name = column_name
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

    def _search_index(self, vector_to_search: np.ndarray, k: int = 5) -> tuple:
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
            sentence_embeddings = self.embedding_model.encode(sentences, batch_size=64, show_progress_bar=True)
            D = sentence_embeddings.shape[1]
            self._create_index(D)
            self._add_to_index(sentence_embeddings)
            return self.obj_faiss.is_trained
        except Exception as e:
            logging.error(f"Falha ao construir o índice FAISS: {e}")
            return False

    def save_index(self, index_path: str) -> None:
        if self.obj_faiss is not None:
            faiss.write_index(self.obj_faiss, index_path)
            logging.info(f"Índice salvo em: {index_path}")
        else:
            logging.warning("Nenhum índice para salvar.")

    def load_index(self, index_path: str) -> None:
        if os.path.exists(index_path):
            self.obj_faiss = faiss.read_index(index_path)
            logging.info(f"Índice carregado de: {index_path}")
        else:
            logging.error(f"Arquivo de índice não encontrado em: {index_path}")

    def search(self, query: str, k: int = 5) -> str:
        """
        Realiza busca semântica e retorna resultados em JSON.
        """
        if self.obj_faiss is None or not self.obj_faiss.is_trained:
            logging.error("O índice não foi construído. Execute build_index() primeiro.")
            return json.dumps({"error": "Índice não construído."}, ensure_ascii=False)

        logging.info(f"Buscando por textos similares a: '{query}'")

        test_embedding = self.embedding_model.encode([query])
        distances, indices = self._search_index(test_embedding, k=k)
        
        results = []
        if indices is not None:
            for i, idx in enumerate(indices[0]):
                result_row = self.dataset.iloc[idx]
                results.append({
                    "rank": i+1,
                    "id_chamado": str(result_row.get("id_chamado", "")),
                    "descricao": str(result_row.get(self.column_name, "")),
                    "sugestao": str(result_row.get("llm_suggestion", "")),
                    "similaridade": float(distances[0][i])
                })
        
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)


def main():
    FILE_PATH = "./results/kmeans/analise_k_means-clusters_using_sentence-embbeding_cluster_using-gemma3:4b_results.csv"
    INDEX_PATH = "./results/faiss_index.bin"
    TEXT_COLUMN = "descrição"
    K_RESULTS = 5
    TEXTO_DE_TESTE = "Erro ao tentar acessar o sistema de processo judicial eletrônico, a página não carrega."
    
    faiss_system = FAISS(filepath=FILE_PATH, column_name=TEXT_COLUMN)
    
    if faiss_system.dataset is not None:
        if not os.path.exists(INDEX_PATH):
            index_ready = faiss_system.build_index()
            if index_ready:
                faiss_system.save_index(INDEX_PATH)
        else:
            faiss_system.load_index(INDEX_PATH)
        
        json_results = faiss_system.search(query=TEXTO_DE_TESTE, k=K_RESULTS)
        print(json_results)


if __name__=="__main__":
    main()
