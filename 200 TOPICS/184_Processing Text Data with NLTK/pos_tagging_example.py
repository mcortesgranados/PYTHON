import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize


text = "NLTK is a leading platform for building Python programs to work with human language data."
tokens = word_tokenize(text)

pos_tags = nltk.pos_tag(tokens)

print("POS Tags:", pos_tags)
