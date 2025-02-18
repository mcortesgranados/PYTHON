import string

def analyze_reviews(reviews):
    total_rating = 0
    word_count = {}
    monthly_reviews = {}
    STOPWORDS = set(["the", "and", "a", "to", "of", "in", "but", "some", "is", "it", "i", "for", "on", "with", "was"])

    for review in reviews:
        total_rating += review["rating"]  # Accumulate total rating for average calculation
        words = review["review"].lower().split()
        
        for word in words:
            word = word.strip(string.punctuation)  # Remove punctuation from words
            if word and word not in STOPWORDS:  # Check if word is not empty and not in STOPWORDS
                word_count[word] = word_count.get(word, 0) + 1  # Increment word count
        
        month = review["date"][5:7]  # Extract month from date
        monthly_reviews[month] = monthly_reviews.get(month, 0) + 1  # Increment monthly review count

    average_rating = total_rating / len(reviews)  # Calculate average rating
    
    max_word_count = max(word_count.values(), default=0)  # Find max word count
    most_common_words = sorted([word for word, count in word_count.items() if count == max_word_count])  # Find most common words
    
    most_reviews_month = max(monthly_reviews, key=monthly_reviews.get)  # Find month with most reviews
    month_name = {"01": "January", "02": "February", "03": "March", "04": "April", "06": "June"}  # Month name mapping
    most_reviews_month = month_name.get(most_reviews_month, "Unknown")  # Get month name from mapping

    # Print results
    print(f"Average Rating: {average_rating:.1f}")
    print(f"Most Common Words: {most_common_words}")
    print(f"Month with Most Reviews: {most_reviews_month}")

# Example input
reviews = [
    {"id": 1, "rating": 5, "review": "Great app. Love the features. The design is outstanding.", "date": "2023-01-15"},
    {"id": 2, "rating": 4, "review": "Very useful. It's become a daily tool for me.", "date": "2023-01-17"},
    {"id": 3, "rating": 3, "review": "Decent, but some features don't work well.", "date": "2023-02-05"},
    {"id": 4, "rating": 2, "review": "I experienced some bugs. Needs fixing.", "date": "2023-02-11"},
    {"id": 5, "rating": 5, "review": "Outstanding! Everything I wanted in an app.", "date": "2023-02-14"},
    {"id": 6, "rating": 4, "review": "Good app overall, just some minor issues.", "date": "2023-02-20"},
    {"id": 7, "rating": 3, "review": "Average, but the user experience could be better.", "date": "2023-03-05"}
]
reviews1 = [
  {"id": 1, "rating": 5, "review": "The coffee was fantastic.", "date": "2022-05-01"},
  {"id": 2, "rating": 4, "review": "Excellent atmosphere. Love the modern design!", "date": "2022-05-15"},
  {"id": 3, "rating": 3, "review": "The menu was limited.", "date": "2022-05-20"},
  {"id": 4, "rating": 4, "review": "Highly recommend the caramel latte.", "date": "2022-05-22"},
  {"id": 5, "rating": 4, "review": "The seating outside is a nice touch.", "date": "2022-06-01"},
  {"id": 6, "rating": 5, "review": "It's my go-to coffee place!", "date": "2022-06-07"},
  {"id": 7, "rating": 3, "review": "I found the Wi-Fi to be quite slow.", "date": "2022-06-10"},
  {"id": 8, "rating": 3, "review": "Menu could use more vegan options.", "date": "2022-06-15"},
  {"id": 9, "rating": 4, "review": "Service was slow but the coffee was worth the wait.", "date": "2022-06-20"},
  {"id": 10, "rating": 5, "review": "Their pastries are the best.", "date": "2022-06-28"},
  {"id": 11, "rating": 2, "review": "Very noisy during the weekends.", "date": "2022-07-05"},
  {"id": 12, "rating": 5, "review": "Baristas are friendly and skilled.", "date": "2022-07-12"},
  {"id": 13, "rating": 3, "review": "It's a bit pricier than other places in the area.", "date": "2022-07-18"},
  {"id": 14, "rating": 4, "review": "Love their rewards program.", "date": "2022-07-25"},
]

# Call the function with the input
analyze_reviews(reviews)
analyze_reviews(reviews1)