import torch
from tqdm import tqdm
from utils import criterion,iou_loss

def train(network, optimizer, dataloader, device):
    alpha=0.6
    network.train()
    loss_per_epoch,loss_bce_per_epoch,loss_iou_per_epoch=0,0,0
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
        
        loss_per_epoch+=loss.detach_().cpu().item()
        loss_bce_per_epoch+=bceloss.detach_().cpu().item()
        loss_iou_per_epoch+=iouloss.detach_().cpu().item()

        # process.set_description('BCELoss:{},IoULoss:{}'.format(
        #     bceloss.detach_().cpu().item(),iouloss.detach_().cpu().item()))
    
    loss_dict=dict(
        loss=loss_per_epoch/len(dataloader),
        loss_bce=loss_bce_per_epoch/len(dataloader),
        loss_iou=loss_iou_per_epoch/len(dataloader)
    )

    return loss_dict
        
