from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./results/tokenizer_custom.json")
encoded = tokenizer.encode("violência doméstica na zona rural")
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
