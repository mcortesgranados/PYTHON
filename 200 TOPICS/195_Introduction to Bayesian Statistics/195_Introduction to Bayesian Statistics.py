import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
true_mean = 5
true_std = 2
data = np.random.normal(true_mean, true_std, 100)

# Define the model
with pm.Model() as model:
    # Prior distribution for the mean
    mean = pm.Normal('mean', mu=0, sigma=10)
    # Prior distribution for the standard deviation
    std_dev = pm.HalfNormal('std_dev', sigma=10)
    # Likelihood (sampling distribution) of the data
    likelihood = pm.Normal('likelihood', mu=mean, sigma=std_dev, observed=data)

    # Perform sampling
    trace = pm.sample(1000, tune=1000)

# Plot the posterior distributions
pm.traceplot(trace)
plt.show()
