# File name: 1822_Variational Linear Regression.py
# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM
# pip install pyro-ppl

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Generate synthetic data
torch.manual_seed(0)
X = torch.linspace(0, 10, 100)
true_slope = 2.0
true_intercept = 1.0
y = true_slope * X + true_intercept + torch.randn_like(X) * 2.0

# Define the variational linear regression model
def model(X, y):
    slope = pyro.sample("slope", dist.Normal(0, 10))
    intercept = pyro.sample("intercept", dist.Normal(intercept_loc, torch.abs(intercept_scale)))
    with pyro.plate("data", len(X)):
        y_pred = slope * X + intercept
        pyro.sample("obs", dist.Normal(y_pred, 2), obs=y)

# Define the guide (variational distribution) for variational inference
def guide(X, y):
    slope_loc = pyro.param("slope_loc", torch.tensor(0.0))
    slope_scale = pyro.param("slope_scale", torch.tensor(1.0))
    slope = pyro.sample("slope", dist.Normal(slope_loc, torch.abs(slope_scale)))
    
    intercept_loc = pyro.param("intercept_loc", torch.tensor(0.0))
    intercept_scale = pyro.param("intercept_scale", torch.tensor(1.0))
    intercept = pyro.sample("intercept", dist.Normal(intercept_loc, torch.abs(intercept_scale))))

# Setup the optimizer and SVI object
adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Perform variational inference
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(X, y)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss = {loss}")

# Get the posterior distribution parameters
slope_loc = pyro.param("slope_loc").item()
slope_scale = pyro.param("slope_scale").item()
intercept_loc = pyro.param("intercept_loc").item()
intercept_scale = pyro.param("intercept_scale").item()

print(f"Estimated Slope: {slope_loc}, Estimated Intercept: {intercept_loc}")
