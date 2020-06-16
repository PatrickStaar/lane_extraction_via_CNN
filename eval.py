import torch
from tqdm import tqdm
from utils import criterion


def eval(network, optimizer, dataloader,device):
    network.eval()
    loss_per_epoch=0
    for i,img,gt in tqdm(enumerate(dataloader)):
        img=img.to(device)
        gt=gt.to(device)
        pred=network(img)
        loss=criterion(pred,gt)
        loss_per_epoch+=loss.detach().cpu().item()

    return loss_per_epoch/len(dataloader)
