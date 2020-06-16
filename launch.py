import torch
from torch.utils.data import DataLoader
from dataset import Data
import os
from network import FCNResNet
from train import train
from eval import eval
from transform import *
from logger import get_logger
from utils import get_time



MAX_EPOCH=50
USE_PRETRAIN=True
# set the same random seed
SEED = 100000
DATA_DIR='./dataset'
CHECKPOINTS_DIR='./checkpoints'
LOG_DIR='./log'


log=get_logger(LOG_DIR)

transform_train=Compose([
    RandomFlip(),
    RandomRotate((0,180)),
    RandomCrop((224,224)),
    ToTensor(),
    Normalize(
        mean=[0.45,0.45,0.45],
        std=[0.225,0.225,0.225]
    )
])

transform_val=Compose([
    ToTensor(),
    Normalize(
        mean=[0.45,0.45,0.45],
        std=[0.225,0.225,0.225]
    )
])

trainset=Data(DATA_DIR,transform=transform_train)
valset=Data(DATA_DIR,training=False,transform=transform_val)

train_loader = DataLoader(
    trainset,batch_size=4,shuffle=True,num_workers=1,pin_memory=True
)
val_loader = DataLoader(
    valset,batch_size=1,num_workers=1
)

log.info('Data loaded.')

# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(SEED)
else:
    device = torch.device('cpu')
    torch.manual_seed(SEED)

log.info('Torch Device:{}'.format(device))

# set model and optimizer
net = FCNResNet()
net.to(device)
net.init_weights()

# pretrained resnet weights 
if USE_PRETRAIN:
    net.load_state_dict(torch.load('./pretrained/resnet18.pth'),strict=False)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,patience=2,factor=0.5,min_lr=1e-7,cooldown=1)

log.info('Model loaded.')


min_loss = 1000.
min_val_loss = 10000.
    
# 开始迭代   
for epoch in range(MAX_EPOCH):
    # set to train mode
    train_avg_loss = train(net, optimizer, train_loader, device)
    scheduler.step(train_avg_loss)
    eval_avg_loss = eval(net, val_loader, device)

    log.info('Epoch-{} Training Loss:{}'.format(epoch+1,train_avg_loss))
    log.info('Epoch-{} Validation Loss:{}'.format(epoch+1,eval_avg_loss))
    
    if train_avg_loss < min_loss:
        min_loss = train_avg_loss
        filename = '{}_ep{}.pt'.format(get_time(), epoch+1)
        torch.save(net.state_dict(), f=os.path.join(CHECKPOINTS_DIR,filename))

    elif eval_avg_loss < min_val_loss:
        min_val_loss = eval_avg_loss
        filename = '{}_ep{}_val.pt'.format(get_time(), epoch+1)
        torch.save(net.state_dict(), f=os.path.join(CHECKPOINTS_DIR,filename))
    else:
        continue

