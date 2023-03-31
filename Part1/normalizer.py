import os
import re
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt
from torchvision import transforms
from dataloaders import *

def get_normalization_factors(train_dataset):

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, collate_fn=custom_collate)

    mean_x, std_x, mean_y, std_y = mean_std(train_loader)

    return mean_x, std_x, mean_y, std_y

def norm_(mean,std):
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    return transform_norm

def inverse_norm(mean_,std_):
    invTrans = transforms.Compose([ transforms.Normalize(mean = 0,
                                                     std = 1/std_),
                                transforms.Normalize(mean = -mean_,
                                                     std = 1),
                               ])
    
    return invTrans

def inv_norm_min_max(x,min_,max_):
    
    return (x*(max_-min_)+min_)/10

def stats_std(loader):
    input, label, scalers, maps = next(iter(loader))
    
    input_ = input.flatten()
    output_ = label.flatten()
    
    mean, std = input_.min(),input_.max()
    mean_, std_ = output_.min(), output_.max()
    
    return mean, std, mean_, std_

def mean_std(loader):
    input, label, scalers, maps = next(iter(loader))
    # shape of input = [b,101,101,1]
    
    mean, std = input.mean([0,1,2,3]), input.std([0,1,2,3])
    mean_, std_ = label.mean([0,1,2]), label.std([0,1,2])
    
    return mean, std, mean_, std_