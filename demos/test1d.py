import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from torch.distributions.normal import Normal
import torch.optim as optim
from nde import distributions
from nde import flows
from nde import transforms
import nn as nn_
from nde.transforms import base, nonlinearities as nl, standard, linear as ll
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tqdm.auto as tqdm

class MyFlow(flows.Flow):
    def __init__(self):
        shape = [1]
        transform = transforms.CompositeTransform([
            nl.Sigmoid(),
            # nl.PiecewiseQuadraticCDF(shape, num_bins=5),
            nl.PiecewiseQuadraticCDF(shape, num_bins=10),
            # nl.PiecewiseLinearCDF(shape),
            # nl.PiecewiseRationalQuadraticCDF(shape),
            nl.Logit()
        ])
        super().__init__(
            transform=transform,
            distribution=distributions.StandardNormal([1]),
        )
flow = MyFlow()
for param in flow.parameters():
    param.requires_grad = True


centre = 0.0
level = -20
xlimits = [-20, 20]

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
print("==run on=> ", device)

# myNorm = Normal(torch.tensor([centre]), torch.tensor([1.0]))
# def logf(x):
#     return myNorm.log_prob(x) + torch.log(x>=level)


def logf(x):
    return -0.5 * x**2  + torch.log(x>level)


def survey_sample(n, centre, rho = 1):
    x = np.random.randn(n) * rho + centre
    return torch.tensor(x).unsqueeze(1).float()

def filterInputs(inputs):
    return inputs[((inputs>xlimits[0]) * (inputs<xlimits[1]))].unsqueeze(1).float()

def calLoss(inputs, log_prob):
    return - ((logf(inputs) - log_prob).exp() * log_prob).mean()

def plotHistory(history, level=10):
    history = np.array(history)
    idx =  np.where(history > level)[0][-1]
    history =  history[idx+1:]
    plt.plot(history)
    plt.show()

optimizer_servy = optim.Adam(flow.parameters(), lr=1e-2)
history = []
loss = torch.zeros([0])
for epoch in range(100): #tqdm.notebook.tqdm(, desc='Survey', leave=False):
    with torch.no_grad():
        inputs = []
        while len(inputs) == 0:
            inputs = survey_sample(1000, 0, 2)
            inputs = filterInputs(inputs)
        # print(inputs.shape)
    log_prob = flow.log_prob(inputs)
    optimizer_servy.zero_grad()
    loss = calLoss(inputs, log_prob)
    loss.backward()
    print("====", loss.item())
    if torch.isnan(loss):
        print("Nan!!!")
        break
    optimizer_servy.step()
    history.append(loss.item())

    # for p in flow.parameters():
    #     print(p.grad)
    # break;

print("===>", inputs[:,0].mean(), loss.item())
optimizer_refine = optim.Adam(flow.parameters(), lr=5e-4)
for epoch in tqdm.trange(1000, desc='Refine', leave=False):
    if(torch.isnan(loss)):
        break;
    # print(epoch)
    with torch.no_grad():
        inputs = []
        while len(inputs) == 0:
            inputs = flow.sample(1000).detach()
            inputs  = filterInputs(inputs)

        if(epoch % 50 == 49):
            print(inputs[:,0].mean(), loss.item())
    log_prob = flow.log_prob(inputs)
    optimizer_refine.zero_grad()
    loss = calLoss(inputs, log_prob)
    loss.backward()
    optimizer_refine.step()
    history.append(loss.item())

# plotHistory(history, 1)

import matplotlib.pyplot as plt

if (not torch.isnan(loss)):
    with torch.no_grad():
        x, loggx = flow.sample_and_log_prob(20000)
        intgral = torch.exp(logf(x) - loggx).mean()
        print("==integral=> %.10f" % intgral)
        print((logf(x) > -10000).sum())
        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.scatter(s0, s1, marker='o', alpha=0.002)
        # plt.plot(0, 0, 'rp', markersize=5)

        plt.title("2d")
        # plt.subplot(122)
        plt.hist((x).detach().numpy(), bins=100,  density=True) #, range=(0,  200)
        plt.title('1d')
        plt.show()
else:
    print("got nan!!")