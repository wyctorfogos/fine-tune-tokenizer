import nltk
nltk.data.path.append('/home/wytcor/nltk_data')

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', download_dir='/home/wytcor/nltk_data', quiet=True)

frase = "Olá, tudo bem? Este é um teste."
tokens = word_tokenize(frase, language='english')
print(tokens)