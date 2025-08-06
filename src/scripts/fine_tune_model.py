import os
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.models import Transformer, Pooling
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Caminhos
model_name_or_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # "bert-base-multilingual-cased" # "all-MiniLM-L6-v2" # "paraphrase-multilingual-MiniLM-L12-v2"
tokenizer_path = "./results/tokenizer_merged"
output_path = "./results/fine_tuned_model"

# Garante que o diretório de saída existe
os.makedirs(output_path, exist_ok=True)

# 1. Carrega o tokenizer expandido e o modelo pré-treinado
# Carregamos primeiro o modelo e o tokenizer base para garantir a compatibilidade
base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
base_model = AutoModel.from_pretrained(model_name_or_path)

# Carregamos o tokenizer local com a expansão e o atualizamos no modelo base
expanded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
base_model.resize_token_embeddings(len(expanded_tokenizer))

# 2. Cria os módulos para o SentenceTransformer
word_embedding_model = Transformer(
    model_name_or_path=model_name_or_path,
    tokenizer_name_or_path=tokenizer_path,
    model_args={'use_auth_token': False}
)

# O SentenceTransformer já faz o resize do embedding se necessário,
# mas é bom garantir que o modelo base carregado localmente já esteja ajustado.
word_embedding_model.auto_model.resize_token_embeddings(len(expanded_tokenizer))
word_embedding_model.tokenizer = expanded_tokenizer

pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

# 3. Cria o SentenceTransformer com os módulos ajustados
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# 4. Cria dataset de pares de frases
train_examples = [
    InputExample(texts=["violência doméstica na zona rural", "agressão em áreas afastadas"], label=1.0),
    InputExample(texts=["direito civil", "legislação de contratos"], label=0.9),
    InputExample(texts=["zona urbana", "região central da cidade"], label=0.8),
    InputExample(texts=["violência doméstica", "direito tributário"], label=0.2),
    InputExample(texts=["direito penal", "processo civil"], label=0.3)
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = CosineSimilarityLoss(model=model)

# 5. Treina o modelo
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=10,
    show_progress_bar=True
)

# 6. Salva o modelo fine-tunado
model.save(output_path)
print(f"Modelo salvo em: {output_path}")