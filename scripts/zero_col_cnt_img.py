#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count no. of zero-entry columns along the x-axis of a object within a grayscale image"""

from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch

def cnt_blck(img):
    """
    Counts the number of columns to the left and right (x-axis) of an object
    within a grayscale image that have zero entries (i.e. black).
    
    Input: 'img' tuple with a torch tensor and an integer
    Output: 
        
    """
    no_cols = img[0].shape[1]
    col_sum = torch.sum(img[0], dim = 1)
    zero_col = (col_sum == 0).type(torch.float)
    
    if zero_col[0][0] == 0:
        very_left = 0
    if zero_col[0][1] == 0 and zero_col[0][0] != 0:
        very_left = 1
    
    if zero_col[0][no_cols - 1] == 0:
        very_right = 0
    if zero_col[0][no_cols - 2] == 0 and zero_col[0][no_cols - 1] != 0:
        very_right = 1
    
    first_diff = (zero_col[0][1:] - zero_col[0][:-1])
    col_idx = (first_diff != 0).nonzero() + 1 # deprecated, replace me
    
    if len(col_idx) == 1 and 'very_left' in locals():
        return (very_left, (no_cols - col_idx[1]).item())
    
    if len(col_idx) == 1 and 'very_right' in locals():
        return (col_idx[0].item(), very_right)
    
    return (col_idx[0].item(), (no_cols - col_idx[1]).item())

### CHECK
#
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
#train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#
#cnt_blck(train_data[0])
#cnt_blck(train_data[615])