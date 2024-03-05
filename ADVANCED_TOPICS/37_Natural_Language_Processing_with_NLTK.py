# FileName: 37_Natural_Language_Processing_with_NLTK.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Natural Language Processing (NLP) with NLTK

# Python provides powerful tools for natural language processing (NLP), with libraries such as NLTK (Natural Language Toolkit).

# Installation Instructions:
# Before running this code, you need to install the NLTK library and download the necessary datasets. You can install NLTK using pip:
# pip install nltk

# After installing NLTK, you need to download the stopwords corpus by running the following commands in your Python interpreter:
# import nltk
# nltk.download('stopwords')

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Example: Text Tokenization and Stopword Removal

# Sample text
text = "NLTK is a leading platform for building Python programs to work with human language data."

# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Print tokens after stopword removal
print("Tokens after stopword removal:", filtered_tokens)
