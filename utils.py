from torch.nn import BCELoss
import time
import torch


def criterion(pred,gt):
    weight_map=torch.ones_like(gt)
    weight_map[gt>0]*=100
    weight_map[gt==0]*=0.01
    bceloss=BCELoss(weight=weight_map)
    loss=bceloss(pred,gt)
    return loss

def iou_loss(pred,gt):
    intersect=pred*gt
    union=pred+gt-intersect
    loss=1-(intersect/union).mean()
    return loss


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T