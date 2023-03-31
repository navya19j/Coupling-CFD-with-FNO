import os
import re
from model_fno import *
from dataset import *
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns
from dataloaders import *
from normalizer import *
from visualize import *
from evaluation import *
from sklearn import preprocessing

def train_one_epoch(train_loader,x_transformer,y_transformer, optimizer, model, scheduler,train_rm,train_rmse):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    rmse = 0
    for x, y, scalers, maps in train_loader:
        
        x = x_transformer(x)
        y = y_transformer(y)
        
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

        rmse += eval(out.detach().cpu().numpy(), y.detach().cpu().numpy(), maps, None)

    print(f'Train Loss : {train_l2/ntrain}')

    train_l2/= ntrain
    rmse/= ntrain

    train_rm.append(train_l2)
    train_rmse.append(rmse)

    scheduler.step()

    return train_rm, train_rmse, model

def test_one_epoch(train_loader,x_transformer,y_transformer, optimizer, model, scheduler,test_rm,test_rmse):
    model.eval()
    test_l2 = 0.0
    trmse = 0
    with torch.no_grad():
        for x, y, scalers, maps in test_loader:
            
            x = x_transformer(x)
            y = y_transformer(y)
            
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s, s)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            trmse += eval(out.detach().cpu().numpy(), y.detach().cpu().numpy(), maps, None)

    print("Test Loss: ", test_l2/ntest)

    
    test_l2 /= ntest
    trmse /= ntest

    t2 = default_timer()
    test_rm.append(test_l2)
    test_rmse.append(trmse)
    
    return test_rm, test_rmse

def plot_errors(train_rm, test_rm, ep):

    if(ep % 30 == 0):
        plt.plot(list(range(0,ep+1)),train_rm)
        plt.plot(list(range(0,ep+1)),test_rm)
        
        plt.plot(list(range(0,ep+1)),train_rmse)
        plt.plot(list(range(0,ep+1)),test_rmse)

        plt.savefig(f"errors_{ep}.png")
        

print("Getting datasets ready.....")
DATA_PATH = os.path.join('/home/user/Desktop/Navya/BTP 2/Part 1/gts_files/')
input_path = os.path.join(DATA_PATH, 'input_processed')
output_path = os.path.join(DATA_PATH, 'output_processed')

dataset = get_dataset(input_path,output_path)
train_dataset, test_dataset = split_into_train_test(dataset, 0.8)

################################################################
# configs
################################################################
ntrain = len(train_dataset)
ntest = len(test_dataset)
# ntest = 100

batch_size = 1
learning_rate = 0.001

epochs = 3
iterations = epochs*(ntrain//batch_size)

modes = 24
width = 32

r = 5
h = int(((101 - 1)/r) + 1)
s = h

# TOCHANGE
s = 101

model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

print("Getting dataloaders ready.....")
train_loader, test_loader = get_loaders(train_dataset, test_dataset)

print("Getting normalization factors ready.....")
mean_x, std_x, mean_y, std_y = get_normalization_factors(train_dataset)
x_transformer = norm_(mean_x,std_x)
y_transformer = norm_(mean_y,std_y)

x_inverse = inverse_norm(mean_x,std_x)
y_inverse = inverse_norm(mean_y,std_y)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)

train_rm = []
test_rm = []

train_rmse = []
test_rmse = []

print("Beginning Training.....")
for ep in range(epochs):
    print(f"Training for Epoch : {ep}")
    train_rm, train_rmse, model = train_one_epoch(train_loader,x_transformer,y_transformer, optimizer, model, scheduler,train_rm,train_rmse)
    test_rm, test_rmse = test_one_epoch(test_loader,x_transformer,y_transformer, optimizer, model, scheduler,test_rm,test_rmse)
    plot_errors(train_rm, test_rm, ep)

    

# save model
torch.save(model.state_dict(),"FNO_Model_23.pkl")



