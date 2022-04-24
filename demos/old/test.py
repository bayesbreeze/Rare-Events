#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:03:45 2022

@author: weijiang
"""
import torch

def calcOut_(h_, x, w1, w2):
    h = w1 * torch.tanh(h_) + w2 * x
    return torch.tanh(h), h

def calcOut(W):
    X = torch.FloatTensor([1, 2, 3, 4])
    Y = torch.FloatTensor([0.2, 0.3, 0.4, 0.5])
    O = torch.zeros(4)
    h = torch.FloatTensor([0])
    O[0], h = calcOut_(h, X[0], W[0], W[1])
    O[1], h = calcOut_(h, X[1], W[0], W[1])
    O[2], h = calcOut_(h, X[2], W[0], W[1])
    O[3], h = calcOut_(h, X[3], W[0], W[1])
    diff = Y-O
    return (diff ** 2).mean()

calcOut(torch.FloatTensor([1,1]))
