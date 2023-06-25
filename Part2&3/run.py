import os
import re
import torch
import numpy as np
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt
from utils import *

def train_one_epoch(train_loader,optimizer,model,scheduler,train_l2_arr, train_rmse_arr):

    model.train()
    rmse = 0
    train_l2 = 0
    for x, y, scalers, maps in train_loader:

        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

        rmse += eval(out.detach().cpu().numpy(), y.detach().cpu().numpy(), maps)

    train_l2_arr.append(train_l2)
    train_rmse_arr.append(rmse)

    return train_l2_arr,train_rmse_arr,model

def test_one_epoch(test_loader,test_l2_arr, test_rmse_arr):

    model.eval()
    test_l2 = 0.0
    trmse = 0
    with torch.no_grad():
        for x, y, scalers, maps in test_loader:
            
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s, s)
            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            trmse += eval(out.detach().cpu().numpy(), y.detach().cpu().numpy(), maps)

    test_l2_arr.append(test_l2)
    test_rmse_arr.append(trmse)

    return test_l2_arr,test_rmse_arr


def plot_errors(train_loss,test_loss,train_rmse,test_rmse):


    if(ep % 30 == 0):
        plt.plot(list(range(0,ep+1)),train_loss)
        plt.plot(list(range(0,ep+1)),test_loss)
        plt.show()
        
        plt.plot(list(range(0,ep+1)),train_rmse)
        plt.plot(list(range(0,ep+1)),test_rmse)
        plt.show()

DATA_PATH = os.getcwd()
input_path = os.path.join(DATA_PATH, 'input')
output_path = os.path.join(DATA_PATH, 'output')
shapes = os.path.join(DATA_PATH, 'shape_coords')

dataset = FNO_Dataset(shapes, input_path, output_path)
# split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

total_indices = list(range(len(dataset)))
 
# take random sample of indices
np.random.shuffle(total_indices)

train_indices = total_indices[:train_size]

test_indices = total_indices[train_size:]

train_dataset = torch.utils.data.Subset(dataset, train_indices)

test_dataset = torch.utils.data.Subset(dataset, test_indices)

# dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

################################################################
# configs
################################################################
ntrain = len(train_dataset)
ntest = len(test_dataset)
# ntest = 100

batch_size = 1
learning_rate = 0.0001

epochs = 100
iterations = epochs*(ntrain//batch_size)

modes = 24
width = 32

r = 5
h = int(((101 - 1)/r) + 1)
s = h

# CHANGE HERE
s = 101

model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)

train_l2_arr = []
test_l2_arr = []

train_rmse_arr = []
test_rmse_arr = []

for ep in range(epochs):
    print(f"Training for epoch : {ep}")
    train_l2_arr,train_rmse_arr,model = train_one_epoch(train_loader,optimizer,model,scheduler,train_l2_arr, train_rmse_arr)
    test_l2_arr,test_rmse_arr = test_one_epoch(test_loader,test_l2_arr, test_rmse_arr)
    # if(ep % 30 == 0):
        
    #     plot_errors(train_l2_arr,test_l2_arr,train_rmse_arr,test_rmse_arr)


PATH = "FNO_Model_Part2_all.pth"

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
            }, PATH)








