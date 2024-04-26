# This program demonstrates working with lists in Python

# Lists (mutable ordered collections)

# In Python, lists are used to store collections of items in a specific order. 
# These items can be of various data types (strings, integers, floats, etc.)

# Create a list of groceries
grocery_list = ["apples", "bananas", "milk", "bread", "cheese"]

# Print the entire list
print("Grocery list:", grocery_list)

# Access an element by index (starts from 0)
first_item = grocery_list[0]
print("First item:", first_item)

# Access the last element using negative indexing
last_item = grocery_list[-1]
print("Last item:", last_item)

# Modify an element using index assignment
grocery_list[2] = "yogurt"  # Replace "milk" with "yogurt"
print("Updated grocery list:", grocery_list)

# Add an element to the end of the list using append
grocery_list.append("eggs")
print("List with eggs:", grocery_list)

# Remove an element by value (careful, might not be unique)
grocery_list.remove("apples")
print("List without apples:", grocery_list)

# Check if an element exists in the list
if "bread" in grocery_list:
  print("Bread is on the list!")
else:
  print("Bread is not on the list.")

# Get the length (number of items) of the list
list_length = len(grocery_list)
print("List length:", list_length)
