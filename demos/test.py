import torch
import numpy as np
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn as nn
from nde.flows import realnvp
import tqdm
import matplotlib.pyplot as plt
from nde.flows import autoregressive as ar


flow = realnvp.SimpleRealNVP(
    features=2,
    hidden_features=20,
    num_layers=8,
    num_blocks_per_layer=2,
)

flow = ar.MaskedAutoregressiveFlow(
            features=2,
            hidden_features=20,
            num_layers=8,
            num_blocks_per_layer=2,
        )

centre = 0.0
level = 3
xlimits = [3, 20]
ylimits = [0, 1]

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
print("==run on=> ", device)

myNorm = Normal(torch.tensor([centre]), torch.tensor([1.0]))
def logf(x):
    return myNorm.log_prob(x[:,0])  + torch.log(x[:,0]>level)\
             + torch.log(x[:,1]>0) + torch.log(x[:,1]<1)

def survey_sample(n, centre, rho = 1):
    x = np.random.randn(n) * rho  + centre
    y = np.random.uniform(size=n, low =0, high=1)
    return torch.FloatTensor(np.concatenate([x.reshape(n, 1), y.reshape(n, 1)], axis=1)).to(device)

def filterInputs(inputs):
    return inputs[((inputs[:,0]>xlimits[0]) * (inputs[:,0]<xlimits[1]))
                  * ((inputs[:,1]>ylimits[0]) * (inputs[:,1]<ylimits[1]))]

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
for epoch in range(100): #tqdm.notebook.tqdm(, desc='Survey', leave=False):
    with torch.no_grad():
        inputs = survey_sample(1000, 3, 5)
        inputs  = filterInputs(inputs)
        # print(inputs.shape)
    log_prob = flow.log_prob(inputs)
    optimizer_servy.zero_grad()
    loss = calLoss(inputs, log_prob)
    loss.backward()
    if loss == torch.nan:
        break
    optimizer_servy.step()
    history.append(loss.item())

print("===>", inputs[:,0].mean(), loss.item())
optimizer_refine = optim.Adam(flow.parameters(), lr=5e-4)
for epoch in range(2000): #tqdm.notebook.tqdm(, desc='Refine', leave=False):
    # print(epoch)
    with torch.no_grad():
        inputs = flow.sample(1000).detach()
        inputs  = filterInputs(inputs)

        if(epoch % 50 == 49):
            print(inputs[:,0].mean(), loss.item())
    log_prob = flow.log_prob(inputs)
    optimizer_refine.zero_grad()
    loss = calLoss(inputs, log_prob)
    if loss == torch.nan:
        break
    loss.backward()
    optimizer_refine.step()
    history.append(loss.item())

# plotHistory(history, 1)

import matplotlib.pyplot as plt

def calIntegral(print=False):
    with torch.no_grad():
        x, loggx = flow.sample_and_log_prob(10000)
        # x, loggx = filterInputs_logj(x, loggx)
        # print(x.shape)
        s0, s1 = x[:,0], x[:,1]

        intgral = torch.exp(logf(x) - loggx).mean()

        if(print):
            print("==integral=> %.10f" % intgral)
            # plt.figure(figsize=(10,5))
            # plt.subplot(121)
            # plt.scatter(s0, s1, marker='o', alpha=0.002)
            # plt.plot(0, 0, 'rp', markersize=5)

            plt.title("2d")
            plt.subplot(122)
            plt.hist((s0).detach().numpy(), bins=100,  range=(0,  20), density=True) #, range=(0,  200)
            plt.title('1d')
            plt.show()
        return intgral

v = [calIntegral() for i in range(100)]
real = 0.0013499
print("mean: %.10f, std: %.10f, accuracy: %.3f%%" % (np.mean(v), np.std(v),
                                              (1-np.abs(np.mean(v) - real)/real)*100))