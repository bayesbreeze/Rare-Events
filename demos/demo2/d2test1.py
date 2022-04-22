import matplotlib.pyplot as plt
import numpy as np
import torch
from demo2.utils import *
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

NN = 3000
n_train, n_test = NN, NN
loader_args = dict(batch_size=NN, shuffle=True)
train_loader, test_loader = load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False)


# inputs = torch.rand(*shape)
# outputs, logabsdet = call_spline_fn(inputs, inverse=False)
# inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)
#
# print(inputs, inputs_inv)
# print(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))


class MyFlow(nn.Module):
    def __init__(self,
                 base_dist='uniform',
                 mixture_dist='gaussian',
                 n_components=10,
                 tail_bound = 5,
                 plot_bounds=(-3, 3)):
        super().__init__()
        self.composition = False
        # if base_dist == 'uniform':
        #     self.base_dist = Uniform(torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
        # elif base_dist == 'beta':
        #     self.base_dist = Beta(torch.FloatTensor([5.0]), torch.FloatTensor([5.0]))
        # else:
        #     raise NotImplementedError

        # self.base_dist = Beta(torch.FloatTensor([5.0]), torch.FloatTensor([5.0]))
        self.tail_bound =  tail_bound
        self.base_dist = Uniform(torch.FloatTensor([-5.0]), torch.FloatTensor([5.0])) #Normal(torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
        if mixture_dist == 'gaussian':
            self.mixture_dist = Normal
        self.n_components = n_components
        self.plot_bounds = plot_bounds

        self.num_bins = n_components
        self.shape = [n_train, 1]
        self.unnormalized_widths_ = nn.Parameter(torch.randn(self.num_bins), requires_grad = True)
        self.unnormalized_heights_ = nn.Parameter(torch.randn(self.num_bins-1), requires_grad=True)

            # nn.Parameter(torch.randn(*self.shape, self.num_bins - 1), requires_grad=True)


    def call_spline_fn(self,inputs, inverse=False):
        unnormalized_widths = self.unnormalized_widths_.view(1, 1, self.num_bins).repeat(*self.shape, 1)
        unnormalized_heights = self.unnormalized_heights_.view(1, 1, self.num_bins - 1).repeat(*self.shape, 1)
        return unconstrained_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            tail_bound=self.tail_bound,
            inverse=inverse
        )
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

# applies gradient steps for each mini-batch in an epoch
def train(model, train_loader, optimizer):
    model.train()
    for x in train_loader:
        x = x.to("cpu").float()
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to("cpu").float()
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args):
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
        train(model, train_loader, optimizer)
        train_loss = eval_loss(model, train_loader)
        train_losses.append(train_loss)

        if test_loader is not None:
            test_loss = eval_loss(model, test_loader)
            test_losses.append(test_loss)

        if plot and (epoch % plot_frequency == 0 or epoch in epochs_to_plot):
            model.plot(f'Epoch {epoch}')

    if plot:
        plot_train_curves(epochs, train_losses, test_losses, title='Training Curve')
    return train_losses, test_losses



def run():
    cdf_flow_model = MyFlow(base_dist='uniform', mixture_dist='gaussian', n_components=20, tail_bound = 3)
    cdf_flow_model_old = copy.deepcopy(cdf_flow_model)
    n_epd = 500
    train_epochs(cdf_flow_model, train_loader, test_loader, dict(epochs=n_epd, lr=2e-2, epochs_to_plot=[0, n_epd-1]))
    visualize_demo1_flow(train_loader, cdf_flow_model_old, cdf_flow_model)


def test():
    num_bins = 10
    shape = [1000, 1]

    unnormalized_widths = torch.randn(*shape, num_bins)
    unnormalized_heights = torch.randn(*shape, num_bins - 1)

    def call_spline_fn(inputs, inverse=False):
        return unconstrained_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse
        )

    inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
    outputs, logabsdet = call_spline_fn(inputs, inverse=False)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)
    # print(inputs, inputs_inv)
    # print(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

run()
# test()