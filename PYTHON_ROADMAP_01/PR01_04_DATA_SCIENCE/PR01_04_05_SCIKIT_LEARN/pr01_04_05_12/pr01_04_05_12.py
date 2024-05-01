"""
12. Text Classification: Categorizing text documents into predefined classes or categories.

Text classification is a common task in natural language processing (NLP) where the goal is to categorize text documents into predefined 
classes or categories. Scikit-learn provides tools for text classification using machine learning algorithms. 
Here's a Python example demonstrating how to perform text classification using scikit-learn:

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets.

Loading the Dataset: We load the 20 Newsgroups dataset using the fetch_20newsgroups function from scikit-learn's datasets module. 
This dataset consists of newsgroup documents categorized into different classes.

Splitting the Dataset: We split the loaded dataset into training and testing sets using the train_test_split function from scikit-learn's model_selection module.

Feature Extraction: We convert the raw text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization method. 
We use the TfidfVectorizer class from scikit-learn's feature_extraction.text module for this purpose.

Training the Classifier: We initialize a logistic regression classifier and train it on the TF-IDF features of the training data using the fit method.

Making Predictions: We make predictions on the TF-IDF features of the testing data using the trained classifier.

Printing Classification Report: We print the classification report, which includes metrics such as precision, recall, and F1-score, to evaluate the 
performance of the text classifier.

"""

# Importing necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the 20 Newsgroups dataset
data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_tfidf)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))


