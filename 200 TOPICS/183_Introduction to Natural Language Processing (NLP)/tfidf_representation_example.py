# Text Representation (TF-IDF):

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
corpus = ["This is the first document.",
          "This document is the second document.",
          "And this is the third one.",
          "Is this the first document?"]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

print("Feature Names:", feature_names)
print("TF-IDF Representation:")
print(X.toarray())
