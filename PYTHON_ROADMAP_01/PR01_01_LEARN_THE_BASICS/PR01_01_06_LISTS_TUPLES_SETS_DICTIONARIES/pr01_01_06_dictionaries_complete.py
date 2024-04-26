# This program demonstrates dictionaries in Python

# Dictionaries (unordered key-value pairs)

# Dictionaries are collections of key-value pairs, similar to real-world dictionaries. 
# Each key acts as a unique identifier for its associated value. Keys must be immutable 
# (like strings or numbers), while values can be of any data type.

# Defining a dictionary
phonebook = {
  "Alice": "123-456-7890",  # Key: Name, Value: Phone number (string)
  "Bob": "987-654-3210",
  "Charlie": {  # You can nest dictionaries! (Key: Name, Value: Another dictionary)
    "home": "555-123-4567",
    "work": "555-789-0123"
  }
}

# Accessing elements by key
alice_number = phonebook["Alice"]
charlies_work_number = phonebook["Charlie"]["work"]

print("Alice's phone number:", alice_number)
print("Charlie's work number:", charlies_work_number)

# Checking for key existence (avoiding errors)
if "David" in phonebook:
  print("David's number:", phonebook["David"])  # This will cause a KeyError if David doesn't exist
else:
  print("David's number is not listed.")

# Modifying values using key
phonebook["Bob"] = "098-765-4321"  # Update Bob's number

print("Updated phonebook (Bob's number changed):", phonebook)

# Adding new key-value pair
phonebook["David"] = "777-345-6789"  # Add David's information

print("Phonebook with David added:", phonebook)

# Removing key-value pair using del
del phonebook["Charlie"]  # Remove Charlie's entry

print("Phonebook without Charlie:", phonebook)

# Looping through key-value pairs
for name, number in phonebook.items():
  print(f"{name}: {number}")  # f-string for formatted printing

# Getting all keys or values as separate lists
keys = list(phonebook.keys())
values = list(phonebook.values())

print("List of names (keys):", keys)
print("List of phone numbers (values):", values)

# Additional methods (demonstrated without modification)
# phonebook.copy() - Creates a shallow copy of the dictionary
# phonebook.pop(key) - Removes the element with the specified key and returns its value
# phonebook.popitem() - Removes and returns an arbitrary key-value pair
# phonebook.get(key, default) - Returns the value for the key if it exists, otherwise returns the default value
