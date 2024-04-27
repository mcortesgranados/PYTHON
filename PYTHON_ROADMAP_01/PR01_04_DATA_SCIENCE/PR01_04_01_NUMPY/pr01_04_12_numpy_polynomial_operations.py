import numpy as np

# Define two polynomials as arrays of coefficients
poly1 = np.array([1, -2, 3])  # Represents: x^2 - 2x + 3
poly2 = np.array([2, 4, 5])    # Represents: 2x^2 + 4x + 5

# Perform polynomial addition
addition_result = np.polyadd(poly1, poly2)
print("Polynomial addition result:", addition_result)

# Perform polynomial subtraction
subtraction_result = np.polysub(poly1, poly2)
print("Polynomial subtraction result:", subtraction_result)

# Perform polynomial multiplication
multiplication_result = np.polymul(poly1, poly2)
print("Polynomial multiplication result:", multiplication_result)

# Perform polynomial division
division_result, remainder = np.polydiv(multiplication_result, poly1)
print("Polynomial division result:", division_result)
print("Remainder:", remainder)
