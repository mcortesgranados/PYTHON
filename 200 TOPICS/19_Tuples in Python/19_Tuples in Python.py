#  Tuples in Python are similar to lists, but they are immutable, meaning
#  their elements cannot be changed after creation. 
# Below is a Python code sample demonstrating the creation and usage of tuples:

# Creating a tuple
my_tuple = (1, 2, 3, 4, 5)
print("Tuple:", my_tuple)

# Accessing elements
print("First element:", my_tuple[0])
print("Last element:", my_tuple[-1])

# Slicing
print("Slice of the tuple:", my_tuple[1:4])

# Length of the tuple
print("Length of the tuple:", len(my_tuple))

# Iterating over elements
print("Iterating over elements:")
for item in my_tuple:
    print(item)

# Concatenating tuples
tuple1 = (1, 2, 3)
tuple2 = ('a', 'b', 'c')
concatenated_tuple = tuple1 + tuple2
print("Concatenated tuple:", concatenated_tuple)
