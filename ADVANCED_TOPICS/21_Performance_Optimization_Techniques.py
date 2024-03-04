# FileName: 21_Performance_Optimization_Techniques.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Performance Optimization Techniques in Python

# Python offers various techniques to optimize the performance of code, including memoization.

# Memoization: Technique of storing the results of expensive function calls and returning the cached result when the same inputs occur again.

# Example: Fibonacci sequence calculation with memoization

def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# Test the fibonacci function
n = 10
print(f"Fibonacci({n}):", fibonacci(n))
