import matplotlib.pyplot as plt
import numpy as np
import torch
from demo2.utils_2 import *
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from scipy.optimize import bisect
import torch.nn.functional as F
from tqdm import tqdm_notebook
import torch.optim as optim
import copy
from quadratic import *

SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

n_train, n_test = NN, NN
loader_args = dict(batch_size=NN, shuffle=True)
train_loader, test_loader = load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False)


class MyFlow(nn.Module):
    def __init__(self,
                 base_dist='uniform',
                 mixture_dist='gaussian',
                 n_components=10,
                 tail_bound = 5,
                 plot_bounds=(-3, 3)):
        super().__init__()
        self.composition = False
        self.base_dist = Uniform(torch.FloatTensor([-5.0]),
                                 torch.FloatTensor([5.0]))  # Normal(torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))

        self.tail_bound =  tail_bound
        if mixture_dist == 'gaussian':
            self.mixture_dist = Normal
        self.n_components = n_components
        self.plot_bounds = plot_bounds

        self.num_bins = n_components
        self.shape = [NN, 1]
        self.unnormalized_widths_ = nn.Parameter(torch.randn(self.num_bins), requires_grad = True)
        self.unnormalized_heights_ = nn.Parameter(torch.randn(self.num_bins-1), requires_grad=True)

    def call_spline_fn(self,inputs, inverse=False):
        unnormalized_widths = self.unnormalized_widths_.view(1, 1, self.num_bins).repeat(*self.shape, 1)
        unnormalized_heights = self.unnormalized_heights_.view(1, 1, self.num_bins - 1).repeat(*self.shape, 1)
        return unconstrained_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            tail_bound=self.tail_bound,
            inverse=inverse )

    def flow(self, x):
        z, log_det = self.call_spline_fn(x.unsqueeze(1), inverse=False)
        return z, log_det

    def invert(self, z):
        with torch.no_grad():
            inputs_inv, logabsdet_inv = self.call_spline_fn(z, inverse=True)
        return inputs_inv.reshape(z.shape)

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return self.base_dist.log_prob(z) + log_det

    # def kl(self, x, fx):
    #     logpx = self.log_prob(x)
    #     return -(fx / logpx.exp() * logpx).mean()

    def kl(self, x, fx):
        logpx = self.log_prob(x)
        return (logpx/torch.log(fx) ).mean()

    def sample(self):
        with torch.no_grad():
            z = self.base_dist.rsample(torch.Size([NN]))
            x = self.invert(z)
        return x

    # Compute loss as negative log-likelihood
    def nll(self, x):
        return - self.log_prob(x).mean()

    def get_density(self):
        x = np.linspace(self.plot_bounds[0], self.plot_bounds[1], NN)
        with torch.no_grad():
            y = self.log_prob(torch.FloatTensor(x)).exp().cpu().numpy()
        return x, y

    def plot(self, title):
        density = self.get_density()
        plt.figure()
        plt.plot(density[0], density[1])
        plt.title(title)


m1 = Normal(torch.tensor([0]), torch.tensor([1]))
eventLevel = torch.FloatTensor([-0.5])
rho = lambda x : torch.where(x < eventLevel,
                10*(eventLevel - x), torch.FloatTensor([0]))
def f(x):
    return (m1.log_prob(x)).exp()

# x = torch.arange(-2, 2, 0.01)
# y = f(x)
# plt.plot(x, y)

# applies gradient steps for each mini-batch in an epoch
def train(model, optimizer):
    x = model.sample()
    fx = f(x)
    loss = model.kl(x.squeeze(), fx.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_epochs(model, train_args):
    # training parameters
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # plotting parameters
    plot = train_args.get('plot', True)
    plot_frequency = train_args.get('plot_frequency', 5)
    if 'epochs_to_plot' in train_args.keys():
        plot_frequency = epochs + 1
        epochs_to_plot = train_args['epochs_to_plot']
    else:
        epochs_to_plot = []

    train_losses, test_losses = [], []
    for epoch in tqdm_notebook(range(epochs), desc='Epoch', leave=False):
        model.train()
        train_loss = train(model, optimizer)
        train_losses.append(train_loss)

        if plot and (epoch % plot_frequency == 0 or epoch in epochs_to_plot):
            model.plot(f'Epoch {epoch}')

    if plot:
        plot_train_curves(epochs, train_losses, train_losses, title='Training Curve')
    return train_losses, train_losses

def run():
    cdf_flow_model = MyFlow(base_dist='uniform', mixture_dist='gaussian', n_components=25, tail_bound = 5)
    n_epd = 100
    train_epochs(cdf_flow_model,  dict(epochs=n_epd, lr=5e-1, epochs_to_plot=[0, n_epd-1]))
    # visualize_demo1_flow(train_loader, cdf_flow_model_old, cdf_flow_model)

run()