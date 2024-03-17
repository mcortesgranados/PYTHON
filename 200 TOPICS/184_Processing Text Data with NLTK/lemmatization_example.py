from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
words = ["dogs", "walking", "ate"]

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("Lemmatized Words:", lemmatized_words)
