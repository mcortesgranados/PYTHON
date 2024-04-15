# File name: 134_Bayesian Linear Regression.py
# @author Manuela Cortes Granados - 15 Abril 2024 2:27 AM

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(scale=2, size=len(X))

# Define prior distributions
prior_slope = norm(loc=0, scale=10)
prior_intercept = norm(loc=0, scale=10)

# Define likelihood function
def likelihood(y_obs, y_pred, sigma):
    return np.prod(norm.pdf(y_obs, loc=y_pred, scale=sigma))

# Define proposal distributions for Metropolis-Hastings
def proposal(current):
    return current + np.random.normal(scale=0.5)

# Initialize parameter values
current_slope = 0
current_intercept = 0

# Perform Metropolis-Hastings sampling
num_samples = 10000
burn_in = 1000
samples = np.zeros((num_samples, 2))

for i in range(num_samples + burn_in):
    proposed_slope = proposal(current_slope)
    proposed_intercept = proposal(current_intercept)
    
    likelihood_current = likelihood(y, current_slope * X + current_intercept, 2)
    likelihood_proposed = likelihood(y, proposed_slope * X + proposed_intercept, 2)
    
    prior_current = prior_slope.pdf(current_slope) * prior_intercept.pdf(current_intercept)
    prior_proposed = prior_slope.pdf(proposed_slope) * prior_intercept.pdf(proposed_intercept)
    
    acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current)
    
    if acceptance_ratio >= 1 or np.random.uniform() < acceptance_ratio:
        current_slope = proposed_slope
        current_intercept = proposed_intercept
    
    if i >= burn_in:
        samples[i - burn_in] = [current_slope, current_intercept]

# Plot the posterior distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.5, color='b')
plt.title('Posterior distribution of slope')
plt.xlabel('Slope')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(samples[:, 1], bins=30, density=True, alpha=0.5, color='r')
plt.title('Posterior distribution of intercept')
plt.xlabel('Intercept')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
