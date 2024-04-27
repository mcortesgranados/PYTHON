"""
Sparse arrays are efficient data structures used to store and manipulate large arrays with mostly zero values. 
They are particularly useful when dealing with large datasets where the majority of elements are zero. 
Here's an example demonstrating sparse arrays using SciPy:

"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

# Create a dense array with mostly zero values
dense_array = np.array([[0, 0, 0, 0],
                        [0, 5, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 3]])

# Convert the dense array to a compressed sparse row (CSR) matrix
csr_sparse = csr_matrix(dense_array)

# Convert the dense array to a compressed sparse column (CSC) matrix
csc_sparse = csc_matrix(dense_array)

# Print the sparse matrices
print("CSR Sparse Matrix:")
print(csr_sparse)

print("\nCSC Sparse Matrix:")
print(csc_sparse)
