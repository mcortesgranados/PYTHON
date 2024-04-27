"""

Explanation:

We start by importing the NumPy library as np.
We set the seed using np.random.seed(42) to ensure reproducibility of results.
We generate random numbers from different probability distributions using various NumPy functions:
np.random.rand(5): Generates 5 random numbers from a uniform distribution over [0, 1).
np.random.randn(5): Generates 5 random numbers from a standard normal distribution (mean=0, stddev=1).
np.random.binomial(n=10, p=0.5, size=5): Generates 5 random integers from a binomial distribution with parameters n=10 (number of trials) 
and p=0.5 (probability of success).
Finally, we print out the generated random numbers for each distribution.
Documentation:

np.random.seed(seed): Sets the random seed for reproducibility. The seed parameter is an integer value.
np.random.rand(d0, d1, ..., dn): Generates random numbers from a uniform distribution over [0, 1). The d0, d1, ..., dn parameters specify the shape of the 
output array.
np.random.randn(d0, d1, ..., dn): Generates random numbers from a standard normal distribution (mean=0, stddev=1). The d0, d1, ..., dn parameters 
specify the shape of the output array.
np.random.binomial(n, p, size): Generates random integers from a binomial distribution. The n parameter is the number of trials, p is the probability 
of success, and size is the output shape.

"""

import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# Generate random numbers from a uniform distribution [0, 1)
uniform_random_numbers = np.random.rand(5)
print("Random numbers from a uniform distribution [0, 1):", uniform_random_numbers)

# Generate random numbers from a standard normal distribution (mean=0, stddev=1)
standard_normal_random_numbers = np.random.randn(5)
print("Random numbers from a standard normal distribution:", standard_normal_random_numbers)

# Generate random integers from a binomial distribution with n=10 and p=0.5
binomial_random_numbers = np.random.binomial(n=10, p=0.5, size=5)
print("Random integers from a binomial distribution with n=10 and p=0.5:", binomial_random_numbers)
