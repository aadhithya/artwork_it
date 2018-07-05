#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 22:03:14 2018

@author: megamind
"""

import argparse as ap
from artist import Artist
import time
import matplotlib.pyplot as plt

parse = ap.ArgumentParser()
parse.add_argument('--content-image','-ci',help='Path to the content image. type: str',type=str,required=True)
parse.add_argument('--style-image','-si',help='path to style image. type: str',type=str,required=True)
parse.add_argument('--out','-o',help='output image path. type: str',type=str,required=True)
parse.add_argument('--img-init', '-init',help='Image initialisation type. 0: Random Init, 1: Init Content Image, 2: Init Style Image.  Default=1 type: int', choices=[0,1,2],type = int, required = False)
parse.add_argument('--style-layers','-sl',help='Layers where style loss is to be calculated. type: list(string). e.g. conv_2 conv_3',type= str,required=False)
parse.add_argument('--content-layers','-cl',help='Layers where content loss is to be calculated. type: list(string). e.g. conv_2 conv_3',type= str,required=False)
parse.add_argument('--iters','-iters',help='number of iterations the optimizer should be run. Default=150 type: int',type=int,required=False)
parse.add_argument('--verbose','-v',help='To print stuff or not to print. Default=True type: bool',type=bool,required=False)
parse.add_argument('--log-every','-le',help='How often should stuff be printed. Default=50 type: int',type=int,required=False)
parse.add_argument('--style-weight','-sw',help='Weight for the style image. Default=1e8 type: float',type=float,required=False)
parse.add_argument('--content-weight','-cw',help='Weight for the content image. Default=1e-1 type: float',type=float,required=False)
parse.add_argument('--cnn','-nn',help='What base CNN should be used. Default=vgg11 available: vgg11, vgg13, vgg13_bn,vgg16,vgg16_bn,vgg19. type: str',choices=['vgg11', 'vgg13', 'vgg13_bn','vgg16','vgg16_bn','vgg19'] ,type=str, required=False)
parse.add_argument('--im-size','-im-size',help='image size. default=(512,512) if cuda available, else (300,300). type=tuple(int)', type=tuple, required=False)

args = parse.parse_args()

args = vars(args)
#print(args)

content_image = args['content_image']
style_image = args['style_image']
out = args['out']
params=dict()
params={k:v for k,v in args.items() if v is not None}

#set default values for our required values.
if 'im_size' not in params:
    params['im_size'] = None
if 'img_init' not in params:
    params['img_init'] = 1
if 'cnn' not in params:
    params['cnn'] = 'vgg11'
    
artist = Artist(content_image,style_image,out,params)
tic = time.time()
artwork = artist.make_artwork()
toc = time.time()
plt.imshow(artwork)
print("It took our artist {}mts to repaitn your picture!".format((toc-tic)/60))