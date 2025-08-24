import os
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import Union

class FAISS():
    """
    Uma classe para gerenciar a indexação e busca no FAISS com embeddings de sentenças.
    """
    def __init__(self, filepath:str, column_name:str="descrição", sentence_embedding_model_name:str = "neuralmind/bert-base-portuguese-cased"):
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_model = SentenceTransformer(model_name_or_path=self.sentence_embedding_model_name)
        self.obj_faiss = None
        self.filepath = filepath
        self.column_name = column_name
        # O dataset é carregado na inicialização e armazenado no atributo 'dataset'
        self.dataset = self._load_data()
    
    def _create_index(self, D: int) -> None:
        try:
            self.obj_faiss = faiss.IndexFlatL2(D)
            print(f"Índice FAISS criado com dimensão D={D}")
        except Exception as e:
            print(f"Erro ao criar o índice FAISS: {e}\n")

    def _add_to_index(self, vectors: np.ndarray) -> None:
        try:
            vectors_norm = np.ascontiguousarray(vectors, dtype=np.float32)
            self.obj_faiss.add(vectors_norm)
            print(f"Adicionados {self.obj_faiss.ntotal} vetores ao índice.")
        except Exception as e:
            print(f"Erro ao adicionar novos vetores: {e}")

    def _search_index(self, vector_to_search: np.ndarray, k:int=5) -> tuple:
        try:
            query_vector_norm = np.ascontiguousarray(vector_to_search, dtype=np.float32)
            distances, indices = self.obj_faiss.search(query_vector_norm, k=k)
            return distances, indices
        except Exception as e:
            print(f"Erro durante a busca: {e}")
            return None, None

    def _load_data(self) -> Union[pd.DataFrame, None]:
        """
        Método privado para carregar os dados. É chamado pelo __init__.
        """
        try:
            print(f"Carregando dados de: {self.filepath}")
            df = pd.read_csv(self.filepath)
            df.dropna(subset=[self.column_name], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"Encontradas {len(df)} linhas válidas para processar.")
            return df
        except FileNotFoundError:
            print(f"Erro: O arquivo não foi encontrado no caminho especificado: {self.filepath}")
            return None
        except KeyError:
            print(f"Erro: O arquivo CSV deve conter uma coluna chamada '{self.column_name}'.")
            return None
        except Exception as e:
            print(f"Um erro inesperado ocorreu ao carregar os dados: {e}")
            return None

    # --- Métodos Públicos ---
    
    def build_index(self) -> bool:
        """
        # MELHORIA: Método público que usa o dataset interno para construir o índice.
        Não precisa mais receber 'sentences' como parâmetro.
        """
        if self.dataset is None:
            print("Dataset não carregado. Abortando a construção do índice.")
            return False
            
        print("\nGerando embeddings para os documentos. Isso pode demorar um pouco...")
        try:
            sentences = self.dataset[self.column_name].tolist()
            # CORREÇÃO: Chamando o modelo de embedding a partir de 'self'
            sentence_embeddings = self.embedding_model.encode(sentences, show_progress_bar=True)
            D = sentence_embeddings.shape[1]
            
            # CORREÇÃO: Chamando os métodos da própria classe com 'self'
            self._create_index(D)
            self._add_to_index(sentence_embeddings)
            
            # CORREÇÃO: Acessando o atributo 'is_trained' corretamente
            return self.obj_faiss.is_trained
        except Exception as e:
            print(f"Falha ao construir o índice FAISS: {e}")
            return False

    def search(self, query: str, k: int = 5):
        """
        # MELHORIA: Método público que realiza a busca e exibe os resultados.
        Não precisa mais receber o 'dataframe' como parâmetro.
        """
        if self.obj_faiss is None or not self.obj_faiss.is_trained:
            print("O índice não foi construído. Execute o método build_index() primeiro.")
            return

        print("\n" + "="*50)
        print(f"Buscando por textos similares a: \n'{query}'")
        print("="*50 + "\n")

        # CORREÇÃO: Chamando o modelo e a busca a partir de 'self'
        test_embedding = self.embedding_model.encode([query])
        distances, indices = self._search_index(test_embedding, k=k)
        
        if indices is not None:
            print(f"Top {k} resultados mais similares:\n")
            for i, idx in enumerate(indices[0]):
                result_row = self.dataset.iloc[idx]
                
                id_chamado = result_row['id_chamado']
                descricao = result_row['descrição']
                sugestao = result_row['llm_suggestion']
                distance = distances[0][i]

                print(f"--- Resultado {i+1} ---")
                print(f"  - ID do Chamado: {id_chamado}")
                print(f"  - Distância L2: {distance:.4f}")
                print(f"  - Descrição: '{descricao}'")
                print(f"  - Sugestão: '{sugestao}'\n")

def main():
    """
    Função principal para orquestrar o processo de busca semântica.
    """
    # --- Configurações ---
    FILE_PATH = "./results/kmeans/analise_k_means-clusters_using_sentence-embbeding_cluster_using-gemma3:4b_results.csv"
    TEXT_COLUMN = "descrição"
    K_RESULTS = 5
    TEXTO_DE_TESTE = "Erro ao tentar acessar o sistema de processo judicial eletrônico, a página não carrega."
    
    # --- Execução do Processo (Agora muito mais simples) ---
    
    # 1. Inicializar o sistema FAISS (que já carrega os dados)
    faiss_system = FAISS(filepath=FILE_PATH, column_name=TEXT_COLUMN)
    
    # 2. Verificar se os dados foram carregados com sucesso antes de prosseguir
    if faiss_system.dataset is not None:
        # 3. Construir o índice (o método agora não precisa de parâmetros)
        index_ready = faiss_system.build_index()
        
        # 4. Realizar a busca se o índice estiver pronto
        if index_ready:
            # O método de busca também ficou mais simples
            faiss_system.search(query=TEXTO_DE_TESTE, k=K_RESULTS)

if __name__=="__main__":
    main()