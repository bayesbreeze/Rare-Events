import copy
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from scipy.optimize import bisect
from piecewise_linear import *

import numpy as np
import matplotlib.pyplot as plt
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook
import pytorch_util as ptu
from demo2_helper import *

class MixtureCDFFlow(nn.Module):
    def __init__(self,
                 n_xsize,
                 base_dist='uniform', 
                 mixture_dist='gaussian',
                 n_components=10,
                 plot_bounds=(-3,3),
                 is_slow=False):
        super().__init__()
        self.composition = False
        if base_dist == 'uniform':
            self.base_dist = Uniform(ptu.FloatTensor([0.0]), ptu.FloatTensor([1.0]))
        elif base_dist == 'beta':
            self.base_dist = Beta(ptu.FloatTensor([5.0]), ptu.FloatTensor([5.0])) 
        else:
            raise NotImplementedError

        self.loc = nn.Parameter(ptu.randn(n_components), requires_grad=True)
        self.log_scale = nn.Parameter(ptu.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(ptu.zeros(n_components), requires_grad=True)

        w = ptu.randn(n_components)  # used for linearwise
        self.weight_ = w.view(1, n_components).unsqueeze(0).repeat(n_xsize, 1, 1)
        self.weight_.requires_grad = True

        if mixture_dist == 'gaussian':
            self.mixture_dist = Normal 
        self.n_components = n_components
        self.plot_bounds = plot_bounds
        self.flow_ = self.flow1 if is_slow else self.flow2
        self.sample_ = self.sample1 if is_slow else self.sample2
        self.n_xsize= n_xsize


    def flow(self, x):
        return self.flow_(x)

    def sample(self, n):
        return self.sample_(self.n_xsize)

    def kl(self, x, fx):
        logpx = self.log_prob(x)
        return -(fx / logpx.exp() * logpx).mean()

    def flow2(self, x):
        return piecewise_linear_transform(x, self.weight_)

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return self.base_dist.log_prob(z) + log_det.view(self.n_xsize, 1)

    def sample2(self, n):
        with torch.no_grad():
            n = self.n_xsize
            z = self.base_dist.rsample(torch.Size([n]))
            x, _ = piecewise_linear_inverse_transform(z, self.weight_, compute_jacobian=False)
        return x
    
    def get_density(self):
        x = np.linspace(self.plot_bounds[0], self.plot_bounds[1], self.n_xsize)
        with torch.no_grad():
            x_ = ptu.FloatTensor(x).unsqueeze(1)
            d = self.log_prob(x_).exp().cpu().numpy()
        return x, d

    def plot(self, title):
        density = self.get_density()
        plt.figure()
        plt.plot(density[0], density[1])
        plt.title(title)

    # Compute loss as negative log-likelihood
    def nll(self, x):
        return - self.log_prob(x).mean()

    def sample1(self, n):
        z = self.base_dist.rsample(torch.Size([n]))
        return self.invert(z).squeeze()

    def invert(self, z):
        # Find the exact x via bisection such that f(x) = z
        results = []
        for z_elem in z:
            def f(x):
                return self.flow(torch.tensor(x).unsqueeze(0))[0] - z_elem

            x = bisect(f, -20, 20)
            results.append(x)
        return torch.tensor(results).reshape(z.shape)

    def flow1(self, x):
        # set up mixture distribution
        weights = F.softmax(self.weight_logits, dim=0).unsqueeze(0).repeat(x.shape[0], 1)
        mixture_dist = self.mixture_dist(self.loc, self.log_scale.exp())
        x_repeat = x.unsqueeze(1).repeat(1, self.n_components)

        # z = cdf of x
        z = (mixture_dist.cdf(x_repeat) * weights).sum(dim=1)

        # log_det = log dz/dx = log pdf(x)
        log_det = (mixture_dist.log_prob(x_repeat).exp() * weights).sum(dim=1).log()
        return z, log_det
