import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class Data(Dataset):
    def __init__(self,root,training=True,transform=None):
        super().__init__()

        self.training=training
        self.root=root
        self.transform=transform
        self.items=self.extract()


    def __getitem__(self, index):
        img,gt=self.items[index].split()
        img = Image.open(os.path.join(self.root,img))
        gt = Image.open(os.path.join(self.root,gt))
        if self.transform is not None:
            img,gt=self.transform([img,gt])

        return [img,gt]


    def __len__(self):
        return len(self.items)

    
    def extract(self):
        file=os.path.join(self.root,'train.txt') \
            if self.training else os.path.join(self.root,'valid.txt')

        items=[]
        with open(file,'r') as f:
            for i in f.readlines():
                items.append(i.strip('\n'))
        return items
    