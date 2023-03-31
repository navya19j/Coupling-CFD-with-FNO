import os
import re
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns

def get_back(u, maps, mask):

    u = u.reshape(u.shape[1],u.shape[2])
    if(mask!=None):
#         add mask to u
            u[mask] = np.min(u)
    
    x_map = maps[0]
    # invert map
    x_map = {v: k for k, v in x_map.items()}
    y_map = maps[1]
    # invert map
    y_map = {v: k for k, v in y_map.items()}

    u_vals = []
    x_vals = []
    y_vals = []

    # round u to 2
    
    for i in range(0, u.shape[0]): 
        temp_x = []
        temp_y = []
        for j in range(0, u.shape[1]):
            temp_x.append(x_map[i])
            temp_y.append(y_map[j])
            u_vals.append(u[i][j])
        x_vals.append(temp_x)
        y_vals.append(temp_y)
        
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    return x_vals, y_vals, u_vals

def reshape_and_normalize(input,label):
    
    input = input.reshape(input.shape[1],input.shape[2])
    label = label.reshape(label.shape[1],label.shape[2])
    
    indices = np.where(input==0)
    label[indices] = -0.5
    
    return input, label

def plot(x,y,u,title_):
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, u, color = "green")
    plt.title(title_)
    
    #XX,YY= np.meshgrid(x,y)
    ax.contourf(x,y,u)
    ax.colorbar()

    # save plot
    fig.savefig(title_ + '.png')

def validate_normalization(test_loader,x_transformer,x_inverse,y_transformer,y_inverse):

    i = 0
    for input, true_label, scalers, maps in test_loader:

        x,y,u    = get_back(true_label, maps)

        x_inp,y_inp,input_ = get_back(input, maps)
        
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
        
        if (i==4):
            break
            

def see_preds(test_loader,model):
    i = 0
    rmse = []

    for input, true_label, scalers, maps in test_loader:

        input = x_transformer(input)

        out = model(input.cuda()).reshape(batch_size, s, s)
        input = x_inverse(input)

    #     take inverse transform
        out = y_inverse(out)
        out = out.cpu().detach().numpy()
        
        mask = np.where(input==0)[1:3]

        true_label = y_transformer(true_label)
        true_label = y_inverse(true_label)
        true_label = true_label.cpu().detach().numpy()

        rmse.append(eval(out,true_label,mask))
        
        i += 1

        if (i<10):
            x,y,u    = get_back(true_label, maps,None)
            x_,y_,u_ = get_back(out, maps,mask)

            x_inp,y_inp,input_ = get_back(input, maps,None)

            cs = plt.contourf(x_inp, y_inp, input.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('Input')
            plt.savefig(f"Input_{i}.png")

            cs = plt.contourf(x, y, true_label.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('True Label')
            plt.savefig(f"True_Label_{i}.png")

            cs = plt.contourf(x_, y_, out.reshape(101,101))

            cbar = plt.colorbar(cs)

            plt.title('Predicted Label')
            plt.savefig(f"Predicted_Label_{i}.png")

            cs = plt.contourf(x, y, abs(true_label.reshape(101,101)-out.reshape(101,101)))

            cbar = plt.colorbar(cs)

            plt.title('Error Plot')
            plt.savefig(f"Error_Plot_{i}.png")

            