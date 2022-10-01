#exec(open('demo5.py').read())
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.datasets import co2

from nde import distributions, flows, transforms
import nn as nn_

import torch
import numpy as np
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn as nn
from nde.flows import realnvp
import tqdm
import matplotlib.pyplot as plt
from nde.flows import autoregressive as ar
from torch.nn.utils import clip_grad_norm_
from scipy.stats import multivariate_normal
from scipy.stats import norm

n_total_steps = 2000
num_flow_steps=5
dim=10
hidden_features=64
num_transform_blocks=2
dropout_probability=0.1
use_batch_norm=0
num_bins=16
learning_rate = 5e-4

def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask

base_transform_type = 'rq'
def create_base_transform(i, _tail_bound):
    if base_transform_type == 'rq':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(
                features=dim,
                even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features:
            nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_transform_blocks,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm
            ),
            num_bins=num_bins,
            apply_unconditional_transform=False,
            tails='linear',
            tail_bound=_tail_bound,
        )
    elif base_transform_type == 'affine':
        return transforms.AffineCouplingTransform(
            mask=create_alternating_binary_mask(
                features=dim,
                even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features:
            nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_transform_blocks,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm
            )
        )
    else:
        raise ValueError
        
## =================
device = torch.device("cpu")
if torch.cuda.is_available():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    torch.set_default_tensor_type(torch.backends.mps.torch.FloatTensor)
    device = torch.device("mps")
print("==run on=> ", device)

import torch
from scipy.stats import norm
import numpy as np

class MyDistr0():
    def __init__(self, scale):
        self.p = 0.35
        self.q = 1 - self.p
        self.s = scale

    def pdf(self, x):
        val = self.p * norm.pdf(x, loc=1, scale = self.s) \
            + self.q * norm.pdf(x, loc=-1, scale= self.s)
        return torch.tensor(val)
    
    def pdf_log(self, x):
        return torch.log(self.pdf(x))

class MyDistr():
    def __init__(self, scale=0.01):
        self.dd = MyDistr0(scale)

    def pdf(self, x):
        return self.pdf_log(x).exp()
    
    def pdf_log(self, x):
        return self.dd.pdf_log(x).sum(axis=-1)

distr = MyDistr(0.2)

# x = np.linspace(-2, 2, 1000)
# plt.plot(x, distr.pdf(x), label='pdf')
# # plt.plot(x, distr.pdf_log(x), label='log pdf')
# plt.legend()
# plt.show()


# def logf(x):
#     return distr.pdf_log(x) + torch.log(x.sum(axis=-1) > 0)

level = 5

def logf(x):
    return distr.pdf_log(x) + torch.log(x.sum(axis=-1) > level)

def calLoss(inputs, log_prob):
    return - ((logf(inputs) - log_prob).exp() * log_prob).mean() 


distribution = distributions.StandardNormal((dim,))
transform = transforms.CompositeTransform(
    [create_base_transform(i, 2) for i in range(num_flow_steps)])
flow = flows.Flow(transform, distribution).to(device)
next(flow.parameters()).device # check device
##==========

optimizer_refine = optim.Adam(flow.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_refine, n_total_steps)  
for epoch in range(n_total_steps): #tqdm.notebook.tqdm(, desc='Refine', leave=False):
    with torch.no_grad():
        _ = flow.eval()
        inputs = flow.sample(2000)
        # inputs  = filterInputs(inputs)
        if(epoch % 100 == 99):
            print(epoch+1, inputs[:,0].mean().cpu(), loss.item().cpu())

    _ = flow.train()
    # scheduler.step(epoch)
    optimizer_refine.zero_grad()

    log_prob = flow.log_prob(inputs)
    loss = calLoss(inputs, log_prob)
    loss.backward()
    optimizer_refine.step()
    scheduler.step()
    # history.append(loss.item())


# with torch.no_grad():
#     _ = flow.eval()
#     inputs = flow.sample(5000).detach()


import numpy as np
import matplotlib.pyplot as plt

num_bins = 15
plt.hist(inputs.cpu().numpy(), num_bins, density=True)
plt.show()

num_bins = 20
plt.hist(((inputs>0) * 2 - 1).sum(axis=-1).cpu().numpy(), num_bins, density=True)
plt.show()



distr2 = MyDistr(0.05)
def logf0(x):
    return distr2.pdf_log(x) + torch.log(x.sum(axis=-1) > level)


def calIntegral(needprint=False, n_sample = 5000 ):
    with torch.no_grad():
        _ = flow.eval()
        x, loggx = flow.sample_and_log_prob(n_sample)
        x = x.cpu()
        loggx=loggx.cpu()
        intgral = torch.exp(logf0(x) - loggx).mean()

        return intgral


import numpy as np
from math import comb
val = np.sum([comb(10, i)*(0.35**i)*(0.65**(10-i))  
              for i in range(8, 11)])
print("correct %.4e" % val)

N = 100
v = [calIntegral(i==0, 10000)  for i in range(N)]  # 10 is x scale, 5 is y scale
real = val
ell = np.array(v) -  real
print("real: %.5e,\nmean: %.5e,\n std: %.3e, \naccuracy: %.3f%%, \nRE1: %.3e, \nRE2: %.3e" 
      % (real, np.mean(v), np.std(v), (1-np.abs(np.mean(v) - real)/real)*100, 
         np.abs(np.mean(ell))/ real, np.std(ell)/np.mean(ell)/np.sqrt(N)))





# guide
n_sample = 10000
with torch.no_grad():
    _ = flow.eval()
    x, loggx = flow.sample_and_log_prob(n_sample)

mmm = np.mean(x.numpy()>0)
print(mmm)

import numpy as np
q = 0.85
N=10**5; threshold = 5 # this is gamma
n=10; p=0.35;
r0, r1 = (1-p)/(1-q), p/q
ss = np.array([np.random.rand(n) <= q for _ in range(N)]) * 1

LRs = np.prod(r0 * (1-ss) + r1 * ss, axis=1)
Xs = np.sum(ss * 2 - 1, axis=1)
vals = LRs * (Xs > threshold)

ell = np.mean(vals)
RE0 = np.abs(np.mean(ell - real))/ real
RE = np.std(vals) / (np.sqrt(N) * ell)

print("ell: %.4e, Ralative Error: %.4e, %.4e"  % (ell, RE0, RE))