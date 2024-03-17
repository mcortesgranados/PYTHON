from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["playing", "played", "plays"]

stemmed_words = [stemmer.stem(word) for word in words]

print("Stemmed Words:", stemmed_words)
