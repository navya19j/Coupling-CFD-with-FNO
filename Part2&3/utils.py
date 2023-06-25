import torch
import numpy as np
from torchvision import transforms

def mean_std(loader):
    input, label, scalers, maps = next(iter(loader))
    # shape of input = [b,101,101,1]
    
    mean, std = input.mean([0,1,2,3]), input.std([0,1,2,3])
    mean_, std_ = label.mean([0,1,2]), label.std([0,1,2])
    
    return mean, std, mean_, std_

def get_back(u, maps, mask):

    u = u.reshape(u.shape[1],u.shape[2])
    if(mask!=None):
#         add mask to u
            if(np.min(u)>0):
                u[mask] = np.min(u)
            else:
                u[mask] = 0
    
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

def get_data_input(SHAPE_PATH,PATH):
    files = os.listdir(PATH)
    shapes_files = os.listdir(SHAPE_PATH)
    files.sort()
    shapes_files.sort()
    all_indices = []

    data = []
    for i in range (0,len(files)):
        if(files[i] != '.DS_Store'):
            temp = []
            mat = np.loadtxt(os.path.join(PATH, files[i]))
            mat2 = np.loadtxt(os.path.join(SHAPE_PATH, shapes_files[i]))
            x = mat[:,0]
            y = mat[:,1]

            for i in range (0,len(x)):
                temp.append((x[i],y[i]))

            all_indices.append(temp)

            u, x_to_idx, y_to_idx, z = preprocess_newer_input(mat2,mat,2)
            data.append(u)

    data = np.array(data)
    # normalize 

    return data, x_to_idx, y_to_idx, all_indices

def reshape_and_normalize(input,label):
    
    input = input.reshape(input.shape[1],input.shape[2])
    label = label.reshape(label.shape[1],label.shape[2])
    
    indices = np.where(input==0)
    label[indices] = -0.5
    
    return input, label

def plot(x,y,u):
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x, y, u, color = "green")
    plt.title("simple 3D scatter plot")
    
    # show plot
    plt.show()

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
    
    # # show plot
    plt.show()

    # save plot
    fig.savefig(title_ + '.png')

def eval(y_pred, y_true, maps):
    # evaluate

    x_vals_pred, y_vals_pred, u_vals_pred = get_back(y_pred, maps,None)
    x_vals_true, y_vals_true, u_vals_true = get_back(y_true, maps,None)

    # get where 0's in u_vals_true
    u_vals_true = np.array(u_vals_true)
    u_vals_pred = np.array(u_vals_pred)

    mask = u_vals_true == 0

    u_vals_true = u_vals_true[~mask]
    u_vals_pred = u_vals_pred[~mask]

    # rmse
    rmse = np.sqrt(np.mean((u_vals_true - u_vals_pred)**2))

    # accuracy between two array
    

    return rmse