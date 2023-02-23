import os
import re
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt

def get_dataset(input_path,output_path):
    dataset = FNO_Dataset(input_path, output_path)

    return dataset

def split_into_train_test(dataset, ratio):
    # split into train and test
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size

    total_indices = list(range(len(dataset)))
    
    # take random sample of indices
    np.random.shuffle(total_indices)

    train_indices = total_indices[:train_size]

    test_indices = total_indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, test_dataset

def custom_collate(batch):
    
    inputs = []
    outputs = []

    for i in range(len(batch)):
        inputs.append(batch[i][0])
        outputs.append(batch[i][1])

    inputs = torch.stack(inputs)
    # reshape
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)
    outputs = torch.stack(outputs)

    return inputs, outputs, batch[0][2], batch[0][3]

def mean_std(loader):
    input, label, scalers, maps = next(iter(loader))
    
    mean, std = input.mean([0,1,2,3]), input.std([0,1,2,3])
    mean_, std_ = label.mean([0,1,2]), label.std([0,1,2])
    
    return mean, std, mean_, std_

def get_loaders(train_dataset, test_dataset):
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    return train_loader,test_loader

