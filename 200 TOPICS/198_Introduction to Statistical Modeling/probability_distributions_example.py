import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Normal Distribution Example
data_normal = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data_normal, bins=30, density=True, alpha=0.6, color='g')

# Plot the probability density function (pdf) of normal distribution
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x), 'k-', lw=2)

plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Poisson Distribution Example
data_poisson = np.random.poisson(lam=3, size=1000)
plt.hist(data_poisson, bins=30, density=True, alpha=0.6, color='b')

# Plot the probability mass function (pmf) of poisson distribution
x = np.arange(0, 15)
plt.plot(x, poisson.pmf(x, mu=3), 'k-', lw=2)

plt.title('Poisson Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
