from transformers import AutoModel, PreTrainedTokenizerFast
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

if __name__=="__main__":
    # Caminho do tokenizer e modelo base
    model_path = "bert-base-multilingual-cased"
    tokenizer_path = "./results/tokenizer_custom.json"

    # Cria modelo de embeddings
    word_embedding_model = Transformer(model_name_or_path=model_path, tokenizer_name_or_path=tokenizer_path)

    # Pooling
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    # Junta tudo
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Teste
    sentence = "violência doméstica na zona rural"
    embeddings = sentence_model.encode(sentence)
    print(embeddings.shape)
