import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a leading platform for building Python programs to work with human language data."
tokens = word_tokenize(text)
sentences = sent_tokenize(text)

print("Word Tokens:", tokens)
print("Sentence Tokens:", sentences)
