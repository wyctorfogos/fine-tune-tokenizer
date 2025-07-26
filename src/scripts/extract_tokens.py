import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils.load_dataset import load_data

# --- Caminho personalizado do NLTK ---
NLTK_DATA_PATH = '/home/wytcor/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)

# --- Baixa os recursos do NLTK, se necessário ---
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Baixando o tokenizador 'punkt'...")
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Baixando a lista de 'stopwords'...")
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=True)

# --- Tokenização ---
def tokenize_portuguese(text: str) -> list:
    if not isinstance(text, str):
        return []
    return word_tokenize(text, language='portuguese')

# --- Salvar tokens únicos extraídos ---
def salvar_tokens_extraidos(tokens_series: pd.Series, caminho: str = "./results/gotten_tokens.csv"):
    try:
        # Garante que a pasta existe
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        # Achatar a lista de listas e obter tokens únicos
        todos_tokens = [token for sublist in tokens_series.dropna() for token in sublist]
        tokens_unicos = sorted(set(todos_tokens))

        # Salvar no CSV
        pd.DataFrame(tokens_unicos, columns=["token"]).to_csv(caminho, index=False, sep=",")
        print(f"Arquivo salvo com sucesso em: {caminho}")
    except Exception as e:
        print(f"Erro ao salvar os tokens: {e}")
        raise

# --- Execução principal ---
if __name__ == "__main__":
    setup_nltk()
    stop_words = set(stopwords.words('portuguese'))

    DATA_PATH = "./data/output.csv"
    print(f"Carregando dataset de '{DATA_PATH}'...")
    dataset = load_data(DATA_PATH)

    print("1. Tokenizando a coluna 'descrição'...")
    dataset['tokens'] = dataset['descrição'].apply(tokenize_portuguese)

    print("2. Salvando os tokens extraídos...")
    salvar_tokens_extraidos(dataset['tokens'])

    print("\nExemplo de tokens extraídos:")
    print(dataset[['descrição', 'tokens']].head().to_markdown(index=False))
