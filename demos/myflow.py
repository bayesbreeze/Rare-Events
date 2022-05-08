"""Implementations of Real NVP."""

import torch
from torch.nn import functional as F
import numpy as np

from nde import distributions
from nde import flows
from nde import transforms
import nn as nn_
from nde.transforms import base, nonlinearities as nl, standard, linear as ll
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MyFlow(flows.Flow):
    def __init__(self):
        shape = [1]
        transform = transforms.CompositeTransform([
            nl.Sigmoid(),
            # nl.PiecewiseQuadraticCDF(shape),
            ll.NaiveLinear(1),
            nl.Logit()
        ])
        super().__init__(
            transform=transform,
            distribution=distributions.StandardNormal([1]),
        )


flow = MyFlow()
inputs = flow.sample(10).detach()
inputs = torch.tensor(np.random.randn(10) * 1).unsqueeze(1).float()
o = flow.log_prob(inputs)
print(o)

#
# def test_forward():
#     batch_size = 10
#     shape = [1]
#     transform = transforms.CompositeTransform([
#         nl.Sigmoid(),
#         nl.PiecewiseQuadraticCDF(shape),
#         nl.PiecewiseQuadraticCDF(shape),
#         nl.PiecewiseQuadraticCDF(shape),
#         nl.Logit()
#     ])
#
#     inputs = torch.randn(batch_size, *shape)
#     print(inputs)
#     outputs, logabsdet = transform(inputs)
#     print(outputs.shape, logabsdet.shape)
#     print(outputs)
#     i, li = transform.inverse(inputs)
#     print(i.shape, li.shape)
# test_forward()

