import torch
from tqdm import tqdm
from utils import criterion

def train(network, optimizer, dataloader, device):
    network.train()
    loss_per_epoch=0

    for i,img,gt in tqdm(enumerate(dataloader)):
        img=img.to(device)
        gt=gt.to(device)
        pred=network(img)
        loss=criterion(pred,gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_per_epoch+=loss.detach().cpu().item()

    return loss_per_epoch/len(dataloader)
        
