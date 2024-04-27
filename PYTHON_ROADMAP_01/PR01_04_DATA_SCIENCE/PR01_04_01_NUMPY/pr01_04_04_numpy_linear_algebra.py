import numpy as np

# Example: Linear Algebra Operations with NumPy

# Create sample matrices
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# 1. Matrix Multiplication (Dot Product)
C = np.dot(A, B)
print("Matrix Multiplication (Dot Product):")
print(C)

# 2. Matrix Decomposition
# a) Matrix Inverse
A_inv = np.linalg.inv(A)
print("\nMatrix Inverse:")
print(A_inv)

# b) Determinant of a Matrix
det_A = np.linalg.det(A)
print("\nDeterminant of Matrix A:")
print(det_A)

# c) Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\nEigenvalues of Matrix A:")
print(eigenvalues)
print("\nEigenvectors of Matrix A:")
print(eigenvectors)

# 3. Solving Linear Equations
# Solve the linear equation Ax = B
x = np.linalg.solve(A, np.array([1, 2]))
print("\nSolution to Ax = B:")
print(x)

# Documenting the Linear Algebra Operations with NumPy:
def numpy_linear_algebra_documentation():
    """
    This function demonstrates various linear algebra operations that can be performed with NumPy.

    Examples:
    - Matrix multiplication (numpy.dot()).
    - Matrix decomposition: inverse (numpy.linalg.inv()), determinant (numpy.linalg.det()), eigenvalues and eigenvectors (numpy.linalg.eig()).
    - Solving linear equations (numpy.linalg.solve()).
    """
    pass

# End of example
