"""
Sentiment analysis and text mining tasks involve extracting insights from textual data, such as analyzing the sentiment expressed in a 
text or extracting meaningful information from it. 
NLP (Natural Language Processing) libraries like NLTK (Natural Language Toolkit) and spaCy provide powerful tools for conducting such tasks. 
Let's explore an example of sentiment analysis using NLTK:

Explanation:

We import the nltk library, which includes NLTK for natural language processing.
We import the SentimentIntensityAnalyzer class from nltk.sentiment module for sentiment analysis.
We provide a sample text for sentiment analysis.
We initialize the SentimentIntensityAnalyzer as sia.
We analyze the sentiment of the text using the polarity_scores() method of sia, which returns a dictionary containing sentiment scores.
We determine the overall sentiment based on the compound score from the sentiment scores dictionary.
We display the original text, sentiment scores, and the determined sentiment.
Documentation:

SentimentIntensityAnalyzer: A class in NLTK used to perform sentiment analysis on textual data.
SentimentIntensityAnalyzer.polarity_scores(text): A method to analyze the sentiment of the given text and return a dictionary containing sentiment scores.
The compound score ranges from -1 (extremely negative) to 1 (extremely positive), with values close to 0 indicating a neutral sentiment.
Sentiment analysis and text mining tasks using NLTK or spaCy allow us to extract valuable insights from textual data, enabling applications in various domains like social media monitoring, customer feedback analysis, and more.

This example demonstrates how to conduct sentiment analysis using NLTK by analyzing the sentiment of a sample text and determining whether it's positive, negative, or neutral based on the sentiment scores.


"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources (uncomment the line below if needed)
# nltk.download('vader_lexicon')

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Analyze the sentiment of the text
sentiment_scores = sia.polarity_scores(text)

# Determine the sentiment based on the compound score
if sentiment_scores['compound'] >= 0.05:
    sentiment = 'Positive'
elif sentiment_scores['compound'] <= -0.05:
    sentiment = 'Negative'
else:
    sentiment = 'Neutral'

# Display the sentiment analysis results
print("Text:", text)
print("Sentiment Scores:", sentiment_scores)
print("Sentiment:", sentiment)
