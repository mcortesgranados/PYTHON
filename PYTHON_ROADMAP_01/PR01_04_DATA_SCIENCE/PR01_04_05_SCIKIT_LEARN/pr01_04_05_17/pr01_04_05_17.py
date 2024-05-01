"""
17. Sentiment Analysis: Determining the sentiment or opinion expressed in text data.

Sentiment analysis is a natural language processing task that involves determining the sentiment or opinion expressed in text data, 
such as positive, negative, or neutral. While scikit-learn is not primarily designed for text processing tasks, 
we can still perform basic sentiment analysis using machine learning algorithms for classification. 
Here's a Python example using scikit-learn to perform sentiment analysis:

Explanation:

Import Libraries: We import necessary classes and functions from scikit-learn.

Sample Text Data: We define a sample list of text data and corresponding sentiment labels. Each text represents a review or opinion.

Split Data: We split the text data and labels into training and testing sets using train_test_split.

Create Pipeline: We create a pipeline using make_pipeline, which consists of CountVectorizer to convert text data into numerical features and LogisticRegression as the classification model.

Train Model: We train the pipeline on the training data using the fit method.

Make Predictions: We make predictions on the test set using the trained pipeline.

Calculate Accuracy: We calculate the accuracy of the model on the test set using the accuracy_score function.

"""

# Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample text data (replace with your own dataset)
texts = [
    "I love this movie! It's so entertaining.",
    "The product is good, but the service is poor.",
    "This book is boring and poorly written.",
    "The restaurant had excellent food and service.",
    "The customer support team was helpful and friendly."
]

# Corresponding sentiment labels (0 for negative, 1 for positive)
labels = [1, 0, 0, 1, 1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Logistic Regression
pipeline = make_pipeline(
    CountVectorizer(),  # Convert text data into numerical features
    LogisticRegression()  # Classification model
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
predictions = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print("Accuracy:", accuracy)
