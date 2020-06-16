import torchvision
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class Data(Dataset):
    def __init__(self,root,training=True,transform=None):
        super().__init__(Data)

        self.training=training
        self.root=root
        self.transform=transform
        self.items=self.extract()


    def __getitem__(self, index):
        img,gt=self.items[index].split()
        img = self.load_as_float(os.path.join(self.root,img))
        gt = self.load_as_float(os.path.join(self.root,gt))
        if self.transform is not None:
            img,gt=self.transform([img,gt])

        return (img,gt)


    def __len__(self):
        return len(self.items)

    
    def extract(self):
        file=os.path.join(self.root,'train.txt') \
            if self.training else os.path.join(self.root,'val.txt')

        items=[]
        with open(file,'r') as f:
            for i in f.readlines():
                items.append(i.strip('\n'))
        return items
    
    def load_as_float(self,dir):
        img=cv2.imread(dir)
        if img.shape[-1]==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img