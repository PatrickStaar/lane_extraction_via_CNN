from torch.nn import BCELoss
import time
import torch


def criterion(pred,gt):
    # weight_map=torch.ones_like(gt)
    # weight_map[gt>=0.5]*=100
    # weight_map[gt<0.5]*=0.01
    bceloss=BCELoss()
    loss=bceloss(pred,gt)
    return loss

def iou_loss(pred,gt):
    # B,_,H,W=pred.size()
    
    intersect=(pred*gt).sum()
    union=pred.sum()+gt.sum()-intersect
    loss=1-intersect/union
    return loss


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T


def get_info(head,loss_dict):
    
    for k,v in loss_dict.items():
        head+='{}:{:6f} '.format(k,v)
    return head