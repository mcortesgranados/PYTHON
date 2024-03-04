# FileName: 12_Performance_Optimization_Memoization.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Performance Optimization Techniques: Memoization in Python
# Memoization is a technique used to improve the performance of functions by caching the results of expensive function calls 
# and returning the cached result when the same inputs occur again.

# This Python code demonstrates memoization, a performance optimization technique. 
# It includes a fibonacci function that calculates Fibonacci numbers using memoization to cache results and improve performance. 
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, location, 
# and a link to the author's LinkedIn profile for context.

# Example of memoization using a dictionary to cache results
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# Example usage
print("Fibonacci(50):", fibonacci(50))
