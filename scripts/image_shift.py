#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shifting objects within grayscale images with black background along x-axis"""

from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch

def shift_image(img, shft_int = 1):
    """
    Shifts the pixels of a grayscale image to the right by shft_int if the shft_int 
    last columns of the tensor have zero entries only (i.e. black). If they cointain 
    any non-zeros, the pixels are shifted to the left by shft_int.
    Abortion if both cases occur.
    
    Input: 'img' tuple with a torch tensor and an integer,
           'shft_int' no. of cols to shift along x-axis
    Output: tuple with a shifted torch tensor and an unmodified integer
        
    """
    no_cols = img[0].shape[1]
    lst_col =  no_cols - 1
    col_sty = no_cols - shft_int 
    col_idx = torch.cat([torch.zeros(col_sty, dtype = torch.bool),
                         torch.ones(shft_int, dtype = torch.bool)])
    cols = torch.reshape(img[0][0,:,col_idx], (no_cols,shft_int))
    cols_sum = torch.sum(cols)
    inval_shft = torch.is_nonzero(cols_sum)

    if inval_shft:
        col_idx = torch.cat([torch.ones(shft_int, dtype = torch.bool),
                             torch.zeros(col_sty, dtype = torch.bool)])
        cols = torch.reshape(img[0][0,:,col_idx], (no_cols,shft_int))
        cols_sum = torch.sum(cols)
        inval_shft = torch.is_nonzero(cols_sum)
        if inval_shft:
            raise ValueError('Consider shifting along another axis.')
        mod_img = torch.cat([img[0][0,:,~col_idx],cols], dim = 1)
        mod_img = torch.reshape(mod_img, (1,mod_img.shape[0], mod_img.shape[1]))
        mod_img = (mod_img,img[1])
        return mod_img
    
    mod_img = torch.cat([cols,img[0][0,:,~col_idx]], dim = 1)
    mod_img = torch.reshape(mod_img, (1,mod_img.shape[0], mod_img.shape[1]))
    mod_img = (mod_img,img[1])
    return mod_img

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# shift all MNIST images by 1 (7 secs)
shft_train_dat = [None] * len(train_data)
for idx in range(len(train_data)):
    shft_train_dat[idx] = shift_image(train_data[idx])

# extract only the ones with label '1'
ones_only = []
for idx in range(len(train_data)):
    if train_data[idx][1] == 1:
        ones_only.append(train_data[idx])
len(ones_only) # 6742

### CHECK

#def plot_it(im_g):
#    return plt.imshow(torch.reshape(im_g[0],(28,28)), cmap='gray', interpolation='none')

#plot_it(ones_only[0])
#plot_it(shift_image(ones_only[0], shft_int = 3))
#plot_it(shift_image(ones_only[0], shft_int = 7)) # shifted to left instead of right
#plot_it(shift_image(ones_only[0], shft_int = 10)) # abortion

## 615 from train_data contains non-zeros on the very right column
#plot_it(train_data[615])
#plot_it(shft_train_dat[615])
