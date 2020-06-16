from torch.nn import BCELoss
import time

def criterion(pred,gt):
    bceloss=BCELoss()
    return bceloss(pred,gt)


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T