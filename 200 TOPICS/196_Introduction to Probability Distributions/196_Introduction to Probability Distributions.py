import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(0)

# 1. Normal (Gaussian) Distribution
mean = 0
std_dev = 1
num_samples = 1000
normal_samples = np.random.normal(mean, std_dev, num_samples)

# 2. Uniform Distribution
low = 0
high = 10
uniform_samples = np.random.uniform(low, high, num_samples)

# 3. Exponential Distribution
scale = 1  # scale parameter (inverse of rate parameter lambda)
exponential_samples = np.random.exponential(scale, num_samples)

# 4. Binomial Distribution
n = 10  # number of trials
p = 0.5  # probability of success
binomial_samples = np.random.binomial(n, p, num_samples)

# 5. Poisson Distribution
lambda_ = 5  # rate parameter
poisson_samples = np.random.poisson(lambda_, num_samples)

# Plot histograms of the generated samples
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.hist(normal_samples, bins=30, color='blue', alpha=0.7)
plt.title('Normal Distribution')

plt.subplot(2, 3, 2)
plt.hist(uniform_samples, bins=30, color='green', alpha=0.7)
plt.title('Uniform Distribution')

plt.subplot(2, 3, 3)
plt.hist(exponential_samples, bins=30, color='orange', alpha=0.7)
plt.title('Exponential Distribution')

plt.subplot(2, 3, 4)
plt.hist(binomial_samples, bins=range(12), color='red', alpha=0.7, align='left')
plt.title('Binomial Distribution')

plt.subplot(2, 3, 5)
plt.hist(poisson_samples, bins=range(15), color='purple', alpha=0.7, align='left')
plt.title('Poisson Distribution')

plt.tight_layout()
plt.show()
