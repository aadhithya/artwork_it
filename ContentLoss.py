#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 22:47:25 2018

@author: megamind
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

class ContentLoss(nn.Module):
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        
        #detach from tree
        self.target = target.detach()
        
    def forward(self, inp):
        #Forward Pass for this module
        self.loss = F.mse_loss(inp,self.target)
        return inp
        
    