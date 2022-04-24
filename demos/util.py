import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_data(n, loc=0, scale=1, visualize = False):
    y = np.random.uniform(size=n)
    x = np.random.normal(loc=loc, scale=scale, size=n)
    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.title("2d")
        plt.subplot(122)
        plt.hist(x, bins=50, density=True)
        plt.title('1d')
        plt.show()
    return train_data

def load_data2(n, loc=0, scale=1, visualize = False):
    y = np.random.uniform(size=n)
    x1 = np.random.normal(loc=0, scale=1, size=n//2)
    x2 = np.random.normal(loc=loc, scale=scale, size = n - n//2)
    x = np.concatenate([x1,x2])
    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.title("2d")
        plt.subplot(122)
        plt.hist(x, bins=50, density=True)
        plt.title('1d')
        plt.show()
    return train_data


def load_data3(n, loc=0, scale=1, visualize = False):
    x = np.random.normal(loc=0, scale=1, size=n)
    x = x[np.where(x>loc, True, False)]
    n = x.shape[0]
    y = np.random.uniform(size=n)
    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.title("2d")
        plt.subplot(122)
        plt.hist(x, bins=50, density=True)
        plt.title('1d')
        plt.show()
    return train_data



def load_data4(n, loc=0, scale=1, visualize = False):
    x = np.random.normal(loc=0, scale=2, size=n)
    y = np.random.normal(loc=0, scale=2, size=n)

    s = (x + y) > loc
    x = x[s]
    y = y[s]
    n = x.shape[0]
    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.plot(0, 0, 'rp', markersize=5)
        plt.axis('equal')
        plt.xlim(-3, 8)
        plt.title("2d")
        plt.subplot(122)
        plt.hist(x, bins=50, density=True)
        plt.title('1d')
        plt.show()
    return train_data



def load_data5(n, loc=0, scale=1, visualize = False):
    x = np.random.normal(loc=0, scale=2, size=n)
    y = np.random.normal(loc=0, scale=2, size=n)

    s = ((0.5*x + y) > loc)*((0.5*x - y) > loc)
    x = x[s]
    y = y[s]
    n = x.shape[0]

    train_data = np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.scatter(x, y, marker='o', alpha=0.1)
        plt.plot(0, 0, 'rp', markersize=5)
        plt.axis('equal')
        plt.xlim(-3, 8)
        plt.title("2d")
        plt.subplot(122)
        plt.hist(x, bins=50, density=True)
        plt.title('1d')
        plt.show()
    return train_data












import torch
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm

from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn as nn
from nde.flows import realnvp
import tqdm
import matplotlib.pyplot as plt
from nde.flows import autoregressive as ar

# torch.set_default_tensor_type(torch.FloatTensor)

torch.manual_seed(0)
np.random.seed(1)

flow = realnvp.SimpleRealNVP(
    features=2,
    hidden_features=20,
    num_layers=10,
    num_blocks_per_layer=2,
)

# flow = ar.MaskedAutoregressiveFlow(
#             features=2,
#             hidden_features=30,
#             num_layers=20,
#             num_blocks_per_layer=2,
#         )

centre = 1

myNorm1 = Normal(torch.tensor([1]), torch.tensor([1.0]))
myNorm2 = Normal(torch.tensor([1.2]), torch.tensor([1.0]))
def f(x):
    v1 = myNorm1.log_prob(x[:,0]).exp()
    v2 = myNorm2.log_prob(x[:,0]).exp()
    return 0.5 * v1 + 0 * v2

def survey_sample(n, centre, rho = 1):
    x = np.random.randn(n) * rho  + centre
    y = np.random.uniform(size=n, low =0, high=1)
    return torch.FloatTensor(np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1))

def calLoss(inputs, log_prob):
    return -(f(inputs) * log_prob / log_prob.exp()).mean()

optimizer_servy = optim.Adam(flow.parameters(), lr=1e-2)
history = []
for epoch in range(100): # tqdm.notebook.tqdm(range(100), desc='Survey', leave=False):
    with torch.no_grad():
        inputs = survey_sample(300, 1)
    log_prob = flow.log_prob(inputs)
    optimizer_servy.zero_grad()
    loss =  calLoss(inputs,  log_prob)
    loss.backward()
    optimizer_servy.step()
    history.append(loss.item())

print("===>", inputs[:,0].mean(), loss.item())
optimizer_refine = optim.Adam(flow.parameters(), lr=5e-3)
for epoch in range(100): #tqdm.notebook.tqdm(range(1000), desc='Refine', leave=False):
    print("===>", epoch,  loss)
    with torch.no_grad():
        inputs = flow.sample(1000).detach()
        if(epoch % 50 == 49):
            print(inputs[:,0].mean(), loss.item())
    log_prob = flow.log_prob(inputs)
    optimizer_refine.zero_grad()
    loss = calLoss(inputs,  log_prob)
    loss.backward()
    optimizer_refine.step()
    history.append(loss.item())

plt.plot(history)

import matplotlib.pyplot as plt

with torch.no_grad():
    samples = flow.sample(5000)
    s0, s1 = samples[:,0], samples[:,1]
    print(s0.mean())
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(s0, s1, marker='o', alpha=0.1)
    plt.plot(0, 0, 'rp', markersize=5)

    plt.title("2d")
    plt.subplot(122)
    plt.hist((s0).detach().numpy(), bins=100,  range=(centre-10,  centre+20), density=True) #, range=(0,  200)
    plt.title('1d')
    plt.show()
