# FileName: 31_Profiling_and_Debugging_Techniques.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Profiling and Debugging Techniques in Python

# Python provides various tools and techniques for profiling and debugging to identify performance bottlenecks and resolve issues.

import time

# Example: Profiling with timeit

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Profile the execution time of the fibonacci function
start_time = time.time()
fibonacci(30)
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

# Example: Debugging with print statements

def divide(a, b):
    result = a / b
    print("Result:", result)
    return result

# Call the divide function with some inputs
divide(10, 2)

