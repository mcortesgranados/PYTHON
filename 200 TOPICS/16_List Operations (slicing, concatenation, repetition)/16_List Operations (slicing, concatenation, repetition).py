# Creating two lists
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c']

# Slicing
print("Sliced list (list1):", list1[1:4])  # Extract elements from index 1 to 3 (exclusive)
print("Sliced list (list2):", list2[:2])    # Extract elements from the beginning up to index 1
print("Sliced list (list1 reversed):", list1[::-1])  # Reverse the list

# Concatenation
concatenated_list = list1 + list2
print("Concatenated list:", concatenated_list)

# Repetition
repeated_list = list2 * 3
print("Repeated list:", repeated_list)
