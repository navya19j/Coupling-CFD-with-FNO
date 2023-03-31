import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from numpy import array,average
from sklearn.preprocessing import StandardScaler

class FNO_Dataset(Dataset):

    def __init__(self, data_path, label_path):
        self.input, self.input_x, self.input_y, self.scaler_x    = get_data_input(data_path)
        self.output, self.output_x, self.output_y, self.scaler_y = get_data_output(label_path)

    def __getitem__(self, idx):

        input = self.input[idx]
        output  = self.output[idx]

        x_mapping = self.input_x
        y_mapping = self.input_y

        normalizer_x = self.scaler_x[idx]
        normalizer_y = self.scaler_y[idx]

        # convert input output to torch tensor
        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()

        scalers = [normalizer_x, normalizer_y]
        mappings = [x_mapping, y_mapping]

        return (input, output, scalers, mappings)

    def __len__(self):

        return self.input.shape[0]

def preprocess(mat,idx_):
    x = mat[:,0]
    y = mat[:,1]
    z = mat[:,2]
    U = mat[:,3]
    V = mat[:,4]
    P = mat[:,5]

    vals = {}

    for i in range (0,len(x)):
        if (np.round(x[i],2),np.round(y[i],2)) in vals.keys():
            # take average of new and old
            vals[(np.round(x[i],2),np.round(y[i],2))].append([U[i],V[i],P[i]])
            # continue
            # vals[(np.round(x[i],2),np.round(y[i],2))] = ((vals[np.round(x[i],2),np.round(y[i],2)][0] + U[i])/2, (vals[np.round(x[i],2),np.round(y[i],2)][1] + V[i])/2, (vals[np.round(x[i],2),np.round(y[i],2)][2] + P[i])/2)
        else:
            vals[(np.round(x[i],2),np.round(y[i],2))] = [[U[i],V[i],P[i]]]

    for key in vals.keys():
        arr = array(vals[key])
        vals[key] = average(arr,axis=0)

    x = np.round(x, 2)
    y = np.round(y, 2)

    x_to_idx = {}
    uniq = 0
    for i in range(len(x)):
        if x[i] not in x_to_idx:
            x_to_idx[x[i]] = uniq
            uniq += 1

    y_to_idx = {}
    uniq = 0
    for i in range(len(y)):
        if y[i] not in y_to_idx:
            y_to_idx[y[i]] = uniq
            uniq += 1

    # U = 0, V = 1, P = 2

    z_ = []
    for i in range (0,len(x)):
        z_.append((x[i],y[i],vals[(x[i],y[i])][idx_]))

    u_np = np.zeros((len(x_to_idx),len(y_to_idx)))

    for item in z_:
        x_coord = item[0]
        y_coord = item[1]
        u_val   = item[2]

        x_idx = x_to_idx[x_coord]
        y_idx = y_to_idx[y_coord]
        u_np[x_idx][y_idx] = u_val

    return u_np, x_to_idx, y_to_idx, z_

def preprocess_newer(mat,idx_):
    x = mat[:,0]
    y = mat[:,1]
    z = mat[:,2]
    U = mat[:,3]
    V = mat[:,4]
    P = mat[:,5]

    x_to_idx = {}
    uniq = 0
    for i in range(len(x)):
        if x[i] not in x_to_idx:
            x_to_idx[x[i]] = uniq
            uniq += 1

    y_to_idx = {}
    uniq = 0
    for i in range(len(y)):
        if y[i] not in y_to_idx:
            y_to_idx[y[i]] = uniq
            uniq += 1

    # U = 0, V = 1, P = 2
    if(idx_ == 0):
        temp = U
    elif (idx_ == 1):
        temp = V
    else:
        temp = P

    u_np = np.zeros((len(x_to_idx),len(y_to_idx)))

    for i in range (0,len(x)):

        x_coord = x[i]
        y_coord = y[i]
        u_val   = temp[i]

        x_idx = x_to_idx[x_coord]
        y_idx = y_to_idx[y_coord]
        u_np[x_idx][y_idx] = u_val

    return u_np, x_to_idx, y_to_idx, temp

def preprocess_newer_input(mat,idx_):
    x = mat[:,0]
    y = mat[:,1]
    z = mat[:,2]
    U = mat[:,3]
    V = mat[:,4]
    P = mat[:,5]

    x_to_idx = {}
    uniq = 0
    for i in range(len(x)):
        if x[i] not in x_to_idx:
            x_to_idx[x[i]] = uniq
            uniq += 1

    y_to_idx = {}
    uniq = 0
    for i in range(len(y)):
        if y[i] not in y_to_idx:
            y_to_idx[y[i]] = uniq
            uniq += 1

    # U = 0, V = 1, P = 2
    if(idx_ == 0):
        temp = U
    elif (idx_ == 1):
        temp = V
    else:
        temp = P

    u_np = np.zeros((len(x_to_idx),len(y_to_idx)))

    for i in range (0,len(x)):

        x_coord = x[i]
        y_coord = y[i]
        u_val   = temp[i]

        x_idx = x_to_idx[x_coord]
        y_idx = y_to_idx[y_coord]
        u_np[x_idx][y_idx] = u_val
        
    min_element = np.min(u_np)
    u_np[u_np > 0] = 1
    u_np[u_np <= 0] = 0

    return u_np, x_to_idx, y_to_idx, temp


def get_data_input(PATH):
    files = os.listdir(PATH)
    files.sort()
    
    files = files[0:10]

    all_indices = []

    data = []
    for file in files:
        if(file != '.DS_Store'):
            temp = []
            mat = np.loadtxt(os.path.join(PATH, file))
            x = mat[:,0]
            y = mat[:,1]

            for i in range (0,len(x)):
                temp.append((x[i],y[i]))

            all_indices.append(temp)

            u, x_to_idx, y_to_idx, z = preprocess_newer_input(mat,0)
            data.append(u)

    data = np.array(data)
    # normalize 

    return data, x_to_idx, y_to_idx, all_indices

def get_data_output(PATH):
    files = os.listdir(PATH)
    files.sort()
    
    files = files[0:10]

    all_indices = []

    data = []
    for file in files:
        if(file != '.DS_Store'):
            temp = []
            mat = np.loadtxt(os.path.join(PATH, file))
            x = mat[:,0]
            y = mat[:,1]

            for i in range (0,len(x)):
                temp.append((x[i],y[i]))

            all_indices.append(temp)

            u, x_to_idx, y_to_idx, z = preprocess_newer(mat,0)
            data.append(u)

    data = np.array(data)

    return data, x_to_idx, y_to_idx, all_indices
