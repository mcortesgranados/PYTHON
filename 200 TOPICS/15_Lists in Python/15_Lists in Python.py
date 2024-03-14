# Creating a list
my_list = [1, 2, 3, 4, 5]
print("Original list:", my_list)

# Accessing elements
print("First element:", my_list[0])
print("Last element:", my_list[-1])

# Slicing
print("Slice of the list:", my_list[1:4])

# Modifying elements
my_list[2] = 10
print("Modified list:", my_list)

# Appending elements
my_list.append(6)
print("List after appending:", my_list)

# Removing elements
removed_element = my_list.pop(2)
print("Removed element:", removed_element)
print("List after removal:", my_list)

# Iterating over elements
print("Iterating over elements:")
for item in my_list:
    print(item)

# List comprehensions
squared_numbers = [x ** 2 for x in my_list]
print("Squared numbers:", squared_numbers)

# List length
print("Length of the list:", len(my_list))

# Checking if an element exists in the list
print("Is 5 in the list?", 5 in my_list)

# Sorting the list
my_list.sort()
print("Sorted list:", my_list)

# Reversing the list
my_list.reverse()
print("Reversed list:", my_list)
