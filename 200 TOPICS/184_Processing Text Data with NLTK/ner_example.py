import nltk
from nltk.tokenize import word_tokenize
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Apple is looking at buying U.K. startup for $1 billion"
tokens = word_tokenize(text)

ner_tags = nltk.ne_chunk(nltk.pos_tag(tokens))

print("NER Tags:", ner_tags)
