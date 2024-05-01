"""
15. Natural Language Processing (NLP): Analyzing and processing text data.

Natural Language Processing (NLP) is a field of study focused on the interaction between computers and human language. 
Scikit-learn provides some basic tools for text processing and analysis, although for more advanced NLP tasks, libraries like 
NLTK (Natural Language Toolkit) or spaCy are often used. Here's a simple example of text classification using scikit-learn:

Explanation:

Loading the 20 Newsgroups Dataset: We load the 20 Newsgroups dataset, which contains 18,846 documents from 20 different newsgroups.

Splitting the Dataset: We split the dataset into training and testing sets using the train_test_split function.

Preprocessing the Text Data: We use the TfidfVectorizer to convert text documents into TF-IDF (Term Frequency-Inverse Document Frequency) features, which represent the importance of a word in a document relative to a collection of documents.

Training a Logistic Regression Classifier: We train a logistic regression classifier using the TF-IDF features.

Making Predictions: We make predictions on the testing set using the trained classifier.

Evaluating the Model: We evaluate the model's performance by calculating accuracy and generating a classification report using scikit-learn's accuracy_score and classification_report functions.


"""

# Importing necessary libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=0.2, random_state=42)

# Preprocessing the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training a logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vectorized, y_train)

# Making predictions
y_pred = clf.predict(X_test_vectorized)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=newsgroups_data.target_names)

# Displaying the classification report
print("Classification Report:")
print(classification_rep)

# Displaying the accuracy
print("Accuracy:", accuracy)
