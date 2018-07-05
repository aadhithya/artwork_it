#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 22:50:22 2018

@author: megamind
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

class StyleLoss(nn.Module):
    
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        
        self.target = self.get_gram_matrix(target.detach())
        
    def get_gram_matrix(self, inp):
        #bs: Batch Size
        #ch: input channels. 3 for RGB image
        #h: image height
        #w= image width
        bs, ch, h, w = inp.data.size()
        
        feats = inp.view(bs*ch,h*w)
        G = torch.mm(feats,feats.t())
        
        #return normalized Gram matrix
        return G.div(bs*ch*h*w)
    
    def forward(self, inp):
        #forward pass for our StyleLoss Module
        inp_gm = self.get_gram_matrix(inp)
        
        self.loss = F.mse_loss(inp_gm, self.target)
        
        return inp
    