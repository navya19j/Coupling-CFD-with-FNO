import os
import re
import torch
import numpy as np
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

def see_preds(test_loader,model):

    i = 0
    rmse = []

    vol_true = []
    vol_pred = []

    for input, true_label, scalers, maps in test_loader:


        out = model(input.cuda()).reshape(batch_size, s, s)

        out = out.cpu().detach().numpy()
        
        out = np.round(out)
        
        mask = np.where(input==0)[1:3]

        true_label = true_label.cpu().detach().numpy()

        i += 1
        
        out_ = out.copy()
        out_ = out_.reshape(101,101)

        out_[mask] = 0
        
        vol_t = np.count_nonzero(true_label == 1)
        vol_p = np.count_nonzero(out_ == 1)
        
        print(vol_t)
        print(vol_p)
        
        vol_true.append(true_label)
        vol_pred.append(out_)
        

        if (abs(vol_p-vol_t)<200 and i<2):
            x,y,u    = get_back(true_label, maps,None)
            x_,y_,u_ = get_back(out, maps,mask)

            x_inp,y_inp,input_ = get_back(input, maps,None)

            cs = plt.contourf(x_inp, y_inp, input.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('Input')
            plt.show()

            cs = plt.contourf(x, y, true_label.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('True Label')
            plt.show()

            cs = plt.contourf(x_, y_, out.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('Predicted Label')
            plt.show()  

            cs = plt.contourf(x, y, abs(true_label.reshape(101,101)-out.reshape(101,101)))

            cbar = plt.colorbar(cs)

            plt.title('Error Plot')
            plt.show()

    return vol_true,vol_pred
# ntest = 100

batch_size = 1
learning_rate = 0.0001

epochs = 100

modes = 24
width = 32

r = 5
h = int(((101 - 1)/r) + 1)
s = h

# CHANGE HERE
s = 101 

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

model = FNO2d(modes, modes, width).cuda()
model.load_state_dict(torch.load("FNO_Model_Part2.pth"))

vol_true,vol_pred = see_preds(test_loader,model)
vol_pred = np.array(vol_pred)
vol_true = np.array(vol_true)

acc = 0
prec = 0
rec = 0
f_1 = 0

for i in range (0,vol_true.shape[0]):
    y_true = vol_true[i,0,:,:]
    y_pred = vol_pred[i,:,:]
    y_true[y_true!=2] = 0
    y_pred[y_pred!=2] = 0
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true.flatten(), y_pred.flatten(),pos_label=2)
    recall = recall_score(y_true.flatten(), y_pred.flatten(),pos_label=2)
    f1 = f1_score(y_true.flatten(), y_pred.flatten(),pos_label=2)
    
    acc += accuracy
    prec += precision
    rec += recall
    f_1 += f1
    
acc = acc/vol_true.shape[0]
prec = prec/vol_true.shape[0]
rec = rec/vol_true.shape[0]
f_1 = f_1/vol_true.shape[0]
    
    
print("Accuracy: %f" % acc)

# precision tp / (tp + fp)

print("Precision: %f" % prec)

# recall: tp / (tp + fn)

print("Recall: %f" % rec)

# f1: 2 tp / (2 tp + fp + fn)

print("F1 score: %f" % f_1)

i = 0
# make df

for input, true_label, scalers, maps in test_loader:
    
    print(input.shape)

    x,y,u    = get_back(true_label, maps,None)

    x_inp,y_inp,input_ = get_back(input, maps,None)
    
    cs = plt.contourf(x_inp, y_inp, input.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('Input')
    plt.show()

    input = x_transformer(input)
    
    cs = plt.contourf(x_inp, y_inp, input.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('Input - Normalized')
    plt.show()
    
    input = x_inverse(input)
    
    cs = plt.contourf(x_inp, y_inp, input.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('Input - De - normalized')
    plt.show()
    
    
    
    cs = plt.contourf(x, y, true_label.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('True Label')
    plt.show() 
    
    true_label = y_transformer(true_label)
    
    cs = plt.contourf(x_inp, y_inp, true_label.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('Label - Normalized')
    plt.show()
    
    true_label = y_inverse(true_label)
    
    cs = plt.contourf(x_inp, y_inp, true_label.reshape(101,101))

    cbar = plt.colorbar(cs)

    plt.title('Label - De Normalized')
    plt.show()
    
    i += 1

        
