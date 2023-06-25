import os
import re
from model_fno import *
from dataset import *
from utils import *
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

def train_one_epoch():
    model.train()
    train_l2 = 0
    rmse = 0
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

    model.eval()
    test_l2 = 0.0
    trmse = 0
    with torch.no_grad():
        for x, y, scalers, maps in test_loader:
            
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s, s)
            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            trmse += eval(out.detach().cpu().numpy(), y.detach().cpu().numpy(), maps)

    return rmse,train_l2,test_l2,trmse

DATA_PATH = "/home/user/Desktop/Navya/Part_2"
input_path = os.path.join(DATA_PATH, 'input')
output_path = os.path.join(DATA_PATH, 'output')
shapes = os.path.join(DATA_PATH, 'shape_coords')
today = datetime.now()
MODEL_NAME = f"Part2_New"

print("Initializing dataset.....")
# dataset initialization
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

print("Wrapping dataloaders.....")
# dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

################################################################
# configs
################################################################
ntrain = len(train_dataset)
ntest = len(test_dataset)

batch_size = 1
learning_rate = 0.01

epochs = 100
iterations = epochs*(ntrain//batch_size)

modes = 24
width = 32

epochs = 70
s = 101

r = 5
# h = int(((101 - 1)/r) + 1)
# s = h

model = FNO2d(modes, modes, width).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=True)

# PATH = "FNO_Model_Part2_all.pth"
PATH = None

if(PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])


train_rm = []
test_rm = []
count = 0

train_rmse = []
test_rmse = []

for ep in range(epochs):

    rmse,train_l2,test_l2,trmse = train_one_epoch()

    print(f'Train Loss : {train_l2/ntrain}')

    scheduler.step()

    print("Test Loss: ", test_l2/ntest)

    train_l2/= ntrain
    test_l2 /= ntest
    
    rmse/= ntrain
    trmse /= ntest
    
    train_rm.append(train_l2)
    if(ep>1):
        if(test_l2>test_rm[-1]):
            count += 1
    test_rm.append(test_l2)
    
    train_rmse.append(rmse)
    test_rmse.append(trmse)
    
    print(ep, train_l2, test_l2)
    
    if (count == 10):
        break

    if(ep % 50 == 0):
        plt.plot(list(range(0,ep+1)),train_rm)
        plt.plot(list(range(0,ep+1)),test_rm)
        plt.show()

torch.save(model.state_dict(), MODEL_NAME)

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
    

#     if (abs(vol_p-vol_t)>200):
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

vol_true = np.array(vol_true)
vol_pred = np.array(vol_pred)

acc = 0
prec = 0
rec = 0
f_1 = 0

for i in range (0,vol_true.shape[0]):
    y_true = vol_true[i,0,:,:]
    y_pred = vol_pred[i,:,:]
    
    print(y_true)
    
    y_true[y_true!=2] = 0
    y_pred[y_pred!=2] = 0
    
    print(y_true)
    
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