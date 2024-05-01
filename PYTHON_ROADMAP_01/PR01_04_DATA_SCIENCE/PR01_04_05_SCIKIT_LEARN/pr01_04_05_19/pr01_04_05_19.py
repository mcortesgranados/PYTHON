"""
19. Transfer Learning: Leveraging knowledge from pre-trained models for new tasks.

Transfer learning is a machine learning technique where a model trained on one task is reused or adapted for a different but related task. 
In scikit-learn, we can use pre-trained models from the sklearn.feature_extraction module, such as CountVectorizer or TfidfVectorizer, 
and fine-tune them on new data. Here's a Python example demonstrating transfer learning using scikit-learn:

Explanation:

Import Libraries: We import necessary classes and functions from scikit-learn, including fetch_20newsgroups to load the 20 Newsgroups dataset, 
CountVectorizer to extract features from text data, MultinomialNB for training a Naive Bayes classifier, and evaluation metrics.

Load Dataset: We load the 20 Newsgroups dataset, which consists of newsgroup documents categorized into different topics.

Extract Features: We use CountVectorizer to convert the text documents into a matrix of token counts. This step prepares the data for training the classifier.

Train Classifier: We train a Multinomial Naive Bayes classifier on the extracted features. This classifier is chosen due to its effectiveness 
for text classification tasks.

Make Predictions: We make predictions on the test set using the trained classifier.

Calculate Accuracy: We calculate the accuracy of the classifier by comparing the predicted labels with the true labels in the test set.

"""

# Importing necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the 20 Newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Extract features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = newsgroups_train.target
X_test = vectorizer.transform(newsgroups_test.data)
y_test = newsgroups_test.target

# Train a Naive Bayes classifier on the extracted features
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
