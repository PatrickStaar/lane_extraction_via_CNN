import torch
from tqdm import tqdm
from utils import criterion,iou_loss

def train(network, optimizer, dataloader, device):
    alpha=0.4
    network.train()
    loss_per_epoch=0
    process=tqdm(enumerate(dataloader))
    for i,data in process:
        optimizer.zero_grad()

        img,gt=data
        img=img.to(device)
        gt=gt.to(device)
        pred=network(img)
        bceloss=criterion(pred,gt)
        iouloss=iou_loss(pred,gt)
        loss=alpha*bceloss+(1-alpha)*iouloss
        
        loss.backward()
        optimizer.step()
        loss.detach_().cpu()
        loss_per_epoch+=loss.item()
        process.set_description('BCELoss:{},IoULoss:{}'.format(
            bceloss.detach_().cpu().item(),iouloss.detach_().cpu().item()))

    return loss_per_epoch/len(dataloader)
        
