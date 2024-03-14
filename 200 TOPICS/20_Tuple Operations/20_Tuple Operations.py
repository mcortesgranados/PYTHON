# Creating tuples
tuple1 = (1, 2, 3)
tuple2 = ('a', 'b', 'c')

# Accessing elements
print("First element of tuple1:", tuple1[0])
print("Last element of tuple2:", tuple2[-1])

# Slicing
print("Slice of tuple1:", tuple1[1:])
print("Slice of tuple2:", tuple2[:2])

# Concatenating tuples
concatenated_tuple = tuple1 + tuple2
print("Concatenated tuple:", concatenated_tuple)

# Unpacking tuples
a, b, c = tuple1
print("Unpacked tuple1:", a, b, c)
