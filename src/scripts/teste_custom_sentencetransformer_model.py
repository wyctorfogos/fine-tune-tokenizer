from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("./results/encoder_custom_alfaneo_jurisbert-base-portuguese-uncased")
frase = "A violência_doméstica ocorre frequentemente na zona_rural. Tribunal de justiça TJES"
embedding = sentence_model.encode(frase)
print(embedding)
