# This program demonstrates type casting expectations

# Define a variable with a string value
user_age_string = "30"

# Try converting the string to an integer (expected behavior)
try:
  user_age_int = int(user_age_string)
  print("Converted age (int):", user_age_int)
except ValueError:
  print("Error: Could not convert age to integer.")

# Try converting a string with non-numeric characters (unexpected behavior)
unexpected_input = "hello"
try:
  # This will raise a ValueError because "hello" cannot be converted to an integer
  unexpected_int = int(unexpected_input)
  print("Converted unexpected input (int):", unexpected_int)  # This line won't execute
except ValueError:
  print("Error: Unexpected input cannot be converted to integer.")
