import pandas as pd

# --- Leitura do dataset ---
def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep="^")
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: O arquivo não foi encontrado no caminho: {file_path}")
    except Exception as e:
        raise ValueError(f"Erro ao carregar os dados: {e}")