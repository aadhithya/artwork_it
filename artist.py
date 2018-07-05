#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 22:04:04 2018

@author: megamind
"""
# The Artist class is responsible to repaint the content image in the style of the style image.  

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
import time

#import our NN modules
from Normalization import Normalization
from StyleLoss import StyleLoss
from ContentLoss import ContentLoss
class Artist:
    
    
    def __init__(self,content_img, style_img,out,params={'transfer_img_type':1,'cnn':'vgg11','im_size':None}):#transfer_img_type = 1, cnn = 'vgg11', im_size=None):
        
        #transfer_img_type = 0 --> use a random gaussian noise image
        #transfer_img_type = 1 --> use the content image
        self.params = params
        
        #check if we have Cuda installed
        self.isCuda = torch.cuda.is_available()
        #set device
        self.device = "cuda" if self.isCuda else "cpu"
        
        #output image path
        self.outpath =  out
        
        #set image size
        if params['im_size'] is None:
            self.im_size = (512,512) if self.isCuda else (300,300)
        else:
            self.im_size = params['im_size']

        #set of transforms to be applied when loading image
        self.img_loader = transforms.Compose([
                transforms.Resize(self.im_size),
                transforms.ToTensor()])
    
        #image unloader. Uselful for plotting the images.
        self.img_unloader = transforms.ToPILImage()
         
        #initialise content and style images
        self.content_img = self.load_image(content_img).unsqueeze(0)
        self.style_img = self.load_image(style_img).unsqueeze(0)

        self.transfer_img = self.init_transfer_img(params['img_init']).unsqueeze(0)
         
        #load CNN and its normalization values
        self.cnn = self.initCNN(params['cnn'])
         
        self.style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']
        self.content_layers = ['conv_4']
         
    #helper function to load the CNN of choice
    def initCNN(self, cnn='vgg11'):
        if cnn is 'vgg13':
            cnn = models.vgg13(pretrained=True).features.to(self.device).eval()
        elif cnn is 'vgg16':
            cnn = models.vgg16(pretrained=True).features.to(self.device).eval()
        elif cnn is 'vgg19':
            cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        elif cnn is 'vgg13_bn':
            cnn = models.vgg13_bn(pretrained=True).features.to(self.device).eval()
        elif cnn is 'vgg16_bn':
            cnn = models.vgg16_bn(pretrained=True).features.to(self.device).eval()
        else:
            cnn = models.vgg11(pretrained=True).features.to(self.device).eval()
        #Initializr normalization values for our CNN
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.norm_Std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        return cnn
    
    #helper function to initialise transfer image
    def init_transfer_img(self, type=0):
        if type is 0:
            img = torch.randn(self.content_img.data.size())
        elif type is 1:
            img = self.content_img.clone()
        else:
            img = self.style_img.clone()
        return img
    
    #helper function to load images as Tensors
    def load_image(self,imagepath, plot=False):
        img = Image.open(imagepath)
        #img.save(imagepath.split("/")[-1])
        if plot:
            print(imagepath)
            self.plot_image(img,imagepath)
        img = self.img_loader(img)
        return img
    
    # helper function to plot images.
    def plot_image(self,image, title="Image plot"):
         plt.title(title)
         plt.figure()
         plt.imshow(image)
    
    #Getter and Setters for our loss layers.
    def set_content_layers(self,layers=None):
        if layers is None:
            self.content_layers = ['conv_4']
        else:
            self.content_layers=layers
    
    def get_layers(self, layers='content'):
        if layers is 'content':
            return self.content_layers
        elif layers is 'style':
            return self.style_layers
        else:
            raise RuntimeError("Error in getting layers: Layers not found. Parameter layers takes values 'content' or 'style' ")
            
    def set_style_layers(self, layers=None):
        if layers is None:
            self.style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']
        else:
            self.style_layers=layers
    
    def make_model(self):
        
        #Lists to keep track of our style and content losses
        content_losses = []
        style_losses = []
        
        #Creating our normalization layer from our CNN model's mean and varience
        normazilation_layer = Normalization(self.norm_mean,self.norm_Std).to(self.device)
        
        #creating our Sequential NN model
        model = nn.Sequential(normazilation_layer)
        
        #Bookkeeping variable
        id = 0
        
        #Lets iterate over the layers in our CNN and add them to our model Network.
        #We'll also add our ContentLoss and StyleLoss modules to the content_layers and style_layers of our model NN.
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                id += 1
                name = 'conv_{}'.format(id)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(id)
                #We don't like inplace
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(id)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(id)
            else:
                raise RuntimeError('Unrecognized layer @ function artist.make_model: {}'.format(layer.__class__.__name__))
            
            #add the layer to our model N/W
            model.add_module(name,layer)
            
            #Add ContentLoss and Style Modules to the content_layers and style_layers of our model NN.
            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(id),content_loss)
                content_losses += [content_loss]
            if name in self.style_layers:
                target = model(self.style_img).detach()
                style_loss = StyleLoss(target)
                model.add_module('style_loss_{}'.format(id),style_loss)
                style_losses += [style_loss]
            
        #now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:i+1]
            
        return model, content_losses, style_losses
    
    #We train our model network here to generate our transfer_image
    def make_artwork(self):#, num_iters=1000, content_weight=1, style_weight=1e7, log_every=250, verbose=True):
        
        if 'iters' in self.params:
            num_iters = self.params['iters']
        else:
            num_iters = 150
            
        if 'content_weight' in self.params:
            content_weight = self.params['content_weight']
        else:
            content_weight = 1
        
        if 'style_weight' in self.params:
            style_weight = self.params['style_weight']
        else:
            style_weight = 1e8
        if 'log_every' in self.params:
            log_every = self.params['log_every']
        else:
            log_every = 50
            
        if 'verbose' in self.params:
            verbose = self.params['verbose']
        else:
            verbose = True
            
        part_out = self.outpath.split('.')[0]
        
        model, content_losses, style_losses = self.make_model()
        
        self.transfer_img = self.init_transfer_img(type=1)
        #We use ADAM/LGBFS as our optimizer for now. 
        optimizer = optim.LBFGS([self.transfer_img.requires_grad_()])
        print("Repainting...")
        run=[0]
        
        while run[0]<=num_iters:
            def closure():
                
                #correct the values of updated input image
                self.transfer_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(self.transfer_img)
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                
                loss.backward()
                
                run[0] += 1
                
                if run[0]%log_every is 0 and verbose:
                    print("Iteration {}, Style Loss: {:4f}, Content Loss: {:4f}".format(run[0],style_score.item(), content_score.item()))
                    temp_img = self.img_unloader(self.transfer_img.squeeze(0))
                    temp_img.save('{}_iter_{}.jpg'.format(part_out,run[0]))
                return style_score + content_score
            
            #take a step
            optimizer.step(closure)
            
        self.transfer_img.data.clamp_(0, 1)
        return self.img_unloader(self.transfer_img.squeeze(0))
        

#Test Modules
#artist = Artist("images/content/d.jpg","images/style/style9.JPG", cnn='vgg13_bn', im_size=(640,640))
#artist.plot_image(artist.content_img,"Original Image")
#artist.plot_image(artist.style_img,"Style Image")
#tic = time.time()
#artwork = artist.make_artwork(num_iters=150,log_every=50)
#artwork.save("images/output.jpg")
#plt.imshow(artwork)
#toc = time.time()
#run_time = toc-tic
#run_time /=60.0
#print("Our Artist took {} minutes to repaint the image!".format(run_time))'''