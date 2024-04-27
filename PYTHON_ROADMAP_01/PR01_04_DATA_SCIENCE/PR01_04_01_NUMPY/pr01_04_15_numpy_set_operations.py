import numpy as np

# Define two sets as NumPy arrays
set1 = np.array([1, 2, 3, 4, 5])
set2 = np.array([4, 5, 6, 7, 8])

# Perform set operations
union_set = np.union1d(set1, set2)          # Union of set1 and set2
intersection_set = np.intersect1d(set1, set2)  # Intersection of set1 and set2
diff_set1_set2 = np.setdiff1d(set1, set2)    # Set difference: elements in set1 but not in set2
diff_set2_set1 = np.setdiff1d(set2, set1)    # Set difference: elements in set2 but not in set1

# Display the results
print("Union:", union_set)
print("Intersection:", intersection_set)
print("Difference (set1 - set2):", diff_set1_set2)
print("Difference (set2 - set1):", diff_set2_set1)
