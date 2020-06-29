import torch
from tqdm import tqdm
from utils import iou_loss


def eval(network, dataloader,device):
    network.eval()
    loss_per_epoch=0
    for i,data in tqdm(enumerate(dataloader)):
        img,gt=data
        img=img.to(device)
        gt=gt.to(device)
        pred=network(img)
        #loss=criterion(pred,gt)
        loss=iou_loss(pred,gt)
        loss_per_epoch+=loss.detach_().cpu().item()
    loss_dict=dict(
        loss=loss_per_epoch/len(dataloader)
    )
    return loss_dict
