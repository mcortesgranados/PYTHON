import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Get word vector for a word
word_vector = model["king"]

print("Word Vector for 'king':", word_vector)
