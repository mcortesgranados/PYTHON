# This program demonstrates working with tuples in Python

# Create a tuple of personal information (immutable)
personal_info = ("Alice", 30, "New York")

# Print the entire tuple
print("Personal info:", personal_info)

# Access elements by index (similar to lists)
name = personal_info[0]
age = personal_info[1]
city = personal_info[2]

print("Name:", name)
print("Age:", age)
print("City:", city)

# Tuples are immutable (cannot change elements)
# Trying to modify an element will result in an error
# personal_info[1] = 31  # This will cause a TypeError

# You can create a new tuple with modifications
updated_info = personal_info + ("Developer",)  # Add "Developer" as a new element

print("Updated info (new tuple):", updated_info)
