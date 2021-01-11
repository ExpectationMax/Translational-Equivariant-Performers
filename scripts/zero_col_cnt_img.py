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
    
    
    # cond. statements are due to insensivity of first_diff to changes in the
    # very first and very last position of the tensor
    if zero_col[0][0] == 0:
        very_left = 0
    if zero_col[0][1] == 0 and zero_col[0][0] != 0:
        very_left = 1 # second left
    
    if zero_col[0][no_cols - 1] == 0:
        very_right = 0
    if zero_col[0][no_cols - 2] == 0 and zero_col[0][no_cols - 1] != 0:
        very_right = 1 # second right
    
    first_diff = (zero_col[0][1:] - zero_col[0][:-1])
    col_idx = (first_diff != 0).nonzero() + 1 # deprecated, replace me
    
    if len(col_idx) == 1 and 'very_left' in locals():
        return (very_left, (no_cols - col_idx[len(col_idx) - 1]).item())
    
    if len(col_idx) == 1 and 'very_right' in locals():
        return (col_idx[0].item(), very_right)
    
    return (col_idx[0].item(), (no_cols - col_idx[1]).item())

### CHECK
#
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#
## train_data[29] # how to handle?
#
#cnt_blck(train_data[0]) # regular
#
#cnt_blck(train_data[1416]) # one left
#cnt_blck(train_data[1322]) # none left
#
#cnt_blck(train_data[109]) # one right
#cnt_blck(train_data[615]) # none right

# extract only the ones with label '1' from test
ones_only = []
for idx in range(len(test_data)):
    if train_data[idx][1] == 1:
        ones_only.append(test_data[idx])
len(ones_only) # 1127

def plot_it(im_g):
    return plt.imshow(torch.reshape(im_g[0],(28,28)), cmap='gray', interpolation='none')

#ones_only_cnt_blck_left = []
#ones_only_cnt_blck_right = []
space_ones = []
for idx in range(len(ones_only)):
    res = cnt_blck(ones_only[idx])
    if res[0] >= 10 and res[1] >= 10: # more than 10 pixels to the left and right
        space_ones.append(ones_only[idx])
    #ones_only_cnt_blck_left.append(res[0])
    #ones_only_cnt_blck_right.append(res[1])
    
len(space_ones) # 63
    
plot_it(space_ones[0])

#fig, ax = plt.subplots(1,2)
#ax[0].hist(ones_only_cnt_blck_left, alpha = 0.5, color = 'r')
#ax[1].hist(ones_only_cnt_blck_right, alpha = 0.5, color = 'g')
#plt.show()


