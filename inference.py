import torch
from torch.utils.data import DataLoader
from dataset import Data
import os
from network import FCNResNet
from train import train
from eval import eval
from transform import *
from logger import get_logger
from utils import get_time, iou_loss
from tqdm import tqdm
from PIL import Image


MAX_EPOCH=50
USE_PRETRAIN=True
# set the same random seed
SEED = 100000
DATA_DIR='./dataset/mass_roads'
CHECKPOINTS_DIR='./checkpoints'
LOG_DIR='./log'
OUTPUT_DIR='./outputs_iou'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def main():
    log=get_logger(LOG_DIR)

    transform_val=Compose([
        Scale((224,224)),
        ToTensor(),
        Normalize(
            mean=[0.45,0.45,0.45],
            std=[0.225,0.225,0.225]
        )
    ])

    
    valset=Data(DATA_DIR,training=False,transform=transform_val)

    val_loader = DataLoader(
        valset,batch_size=1,num_workers=0
    )

    log.info('Data loaded.')
    log.info('Val samples:{}'.format(len(val_loader)))

    # set device
    device = torch.device('cpu')
    # torch.manual_seed(SEED)

    log.info('Torch Device:{}'.format(device))

    # set model and optimizer
    net = FCNResNet()
    net.to(device)
    # net.init_weights()

    # pretrained resnet weights 
    net.load_state_dict(torch.load(
        './checkpoints/06.22.23.14.41_ep36_val.pt'),strict=False)

    log.info('Model loaded.')

    for i,data in tqdm(enumerate(val_loader)):
        img,gt=data
        img=img.to(device)
        gt=gt.to(device)
        pred=net(img)
        iou=iou_loss(pred,gt)
        pred[pred<0.5]=0
        print(iou.detach_().item())
        
        pred=pred.detach_().numpy().squeeze(0).squeeze(0)
        pred=Image.fromarray((pred*255).astype(np.uint8))
        pred.save(f'{OUTPUT_DIR}/{i}.png')

if __name__ == "__main__":
    main()
