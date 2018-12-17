import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDelta
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from util import generate_data, gumbel_sample

"""
This scripts implements a simple Mixture Density network in Pyro. It
uses the same data example as in Bishops's book (Bishop, C. M. (2013).
Pattern Recognition and Machine Learning.) and is based on the pytorch
implementation of David Ha,
https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
"""

n_samples = 1000
n_gaussians = 5


x_data, x_variable, y_data, y_variable = generate_data(n_samples)


"""
Define network model.
"""


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super().__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, data):
        z_h = self.z_h(data.view(-1, 1))
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

    # For inference, we enumerate over the discrete Gaussian mixtures.
    @config_enumerate
    def model(self, x, y):
        """
        Generative model for the data.
        """
        pyro.module("MDN", self)

        pi, sigma, mu = self.forward(y)
        muT = torch.transpose(mu, 0, 1)
        sigmaT = torch.transpose(sigma, 0, 1)

        assert muT.shape == (n_gaussians, n_samples)
        assert sigmaT.shape == (n_gaussians, n_samples)
        with pyro.plate("samples", n_samples):
            assignment = pyro.sample("assignment", dist.Categorical(pi))
            # We need this case distinction for the two different
            # cases of assignment: sampling a random assignment and
            # enumerating over mixtures. See
            # http://pyro.ai/examples/enumeration.html for a tutorial.
            if len(assignment.shape) == 1:
                pyro.sample('obs', dist.Normal(torch.gather(muT, 0, assignment.view(1, -1))[0],
                                               torch.gather(sigmaT, 0, assignment.view(1, -1))[0]),
                            obs=x)
            else:
                pyro.sample('obs', dist.Normal(muT[assignment][:, 0],
                                               sigmaT[assignment][:, 0]),
                            obs=x)


# Create network instance
network = MDN(n_hidden=20, n_gaussians=5)

# Define optimizer
adam_params = {"lr": 0.001, "betas": (0.9, 0.999)}
optimizer = Adam(adam_params)

# Use the AutoDelta guide for MAP estimation of the network parameters
guide = AutoDelta(poutine.block(network.model, hide=['assignment', 'obs']))
# Initialize ELBO loss with enumeration over the discrete mixtures of the GMM model
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(network.model, guide, x_variable, y_variable)
# Initialize SVI instance for stochastic variational inference
svi = SVI(network.model, guide, optimizer, loss=elbo)

"""
Define the main training loop and train the model.
"""


def train_mdn():
    losses = []
    for epoch in range(10000):
        loss = svi.step(x_variable, y_variable)
        if epoch % 100 == 0:
            print(epoch, loss / n_samples)
        losses.append(loss / n_samples)
    return losses


losses = train_mdn()


"""
Sample from trained model.
"""
# evenly spaced samples from -10 to 10
x_test_data = np.linspace(-15, 15, n_samples)
# change data shape, move from numpy to torch
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, 1))


assignment, sigma, mu = network(x_test_tensor)
assignment_data = assignment.data.numpy()
sigma_data = sigma.data.numpy()
mu_data = mu.data.numpy()

k = gumbel_sample(assignment_data)
indices = (np.arange(n_samples), k)
rn = np.random.randn(n_samples)
sampled = rn * sigma_data[indices] + mu_data[indices]


pl.figure(figsize=(8, 8))
pl.scatter(y_data, x_data, alpha=0.2, label='Data')
pl.scatter(x_test_data, sampled, alpha=0.2, color='red', label='Samples')
pl.legend()
pl.show()
