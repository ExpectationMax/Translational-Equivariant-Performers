#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shifting objects within grayscale images with black background"""

from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch

def shift_image(img):
    """
    Shifts the pixels of a grayscale image to the right by one if the last 
    column of pixels is completely zero (i.e. black). If it cointains any non-
    zeros, the pixels are shifted to the left. Abortion if both cases occur.
    
    Input: tuple with a torch tensor and an integer
    Output: tuple with a shifted torch tensor and an unmodified integer
        
    """
    lst_clm_idx = img[0].shape[1] - 1
    lst_clm = torch.reshape(img[0][0,:,lst_clm_idx], (lst_clm_idx + 1,1))
    lst_clm_sum = torch.sum(lst_clm)
    inval_shft = torch.is_nonzero(lst_clm_sum)

    if inval_shft:
        frst_clm = torch.reshape(img[0][0,:,0], (lst_clm_idx + 1,1))
        frst_clm_sum = torch.sum(frst_clm)
        inval_shft = torch.is_nonzero(frst_clm_sum)
        if inval_shft:
            raise ValueError('Consider shifting along another axis.')
        mod_img = torch.cat([img[0][0,:,1:(lst_clm_idx + 1)],frst_clm], dim = 1)
        mod_img = torch.reshape(mod_img, (1,mod_img.shape[0], mod_img.shape[1]))
        mod_img = (mod_img,img[1])
        return mod_img
        
    mod_img = torch.cat([lst_clm,img[0][0,:,0:(lst_clm_idx)]], dim = 1)
    mod_img = torch.reshape(mod_img, (1,mod_img.shape[0], mod_img.shape[1]))
    mod_img = (mod_img,img[1])
    return mod_img

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 7 secs for MNIST
shft_train_dat = [None] * len(train_data)
for idx in range(len(train_data)):
    shft_train_dat[idx] = shift_image(train_data[idx])
    
### CHECK
#
#def plot_it(im_g):
#    return plt.imshow(torch.reshape(im_g[0],(28,28)), cmap='gray', interpolation='none')
#
#plot_it(train_data[0])
#plot_it(shft_train_dat[0])
#
## 615 contains non-zeros on the very right column
#plot_it(train_data[615])
#plot_it(shft_train_dat[615])
