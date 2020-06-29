import torch
import torchvision.transforms.functional as func
import torchvision.transforms.transforms as transforms
import random
import numpy as np
import cv2
import math



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        img,gt=data
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class RandomFlip(object):
    """Randomly horizontally and vertically flips 
    the given numpy array with a probability of 0.5
    """

    def __call__(self, img, gt):
        assert gt is not None
        if random.random() < 0.5:
            img = func.hflip(img)
            gt = func.hflip(gt)
        
        if random.random()<0.5:
            img = func.vflip(img)
            gt = func.vflip(gt)

        return img, gt


class RandomRotate(object):

    def __init__(self,degree_range):
        self.degree_range=degree_range

    def __call__(self,img,gt):
        angle=random.uniform(self.degree_range[0],self.degree_range[1])
        img=func.rotate(img,angle)
        gt=func.rotate(gt,angle)
        return img, gt


class RandomCrop(transforms.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), 
            ratio=(3. / 4., 4. / 3.)):
        super(RandomCrop,self).__init__(size,scale,ratio)
        
    def __call__(self, img, gt):
        i,j,h,w=self.get_params(img,self.scale,self.ratio)
        img=func.resized_crop(img,i,j,h,w,self.size)
        gt=func.resized_crop(gt,i,j,h,w,self.size)
        return img, gt


class Scale(object):
    def __init__(self, size):
        self.size=size
    def __call__(self, img, gt):
        img=func.resize(img,self.size)
        gt=func.resize(gt,self.size)
        return img,gt


class ToTensor(object):

    def __call__(self, img, gt):
        img=func.to_tensor(img)
        gt=func.to_tensor(gt)
        # print(gt.max())iu 
        return img,gt


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt):
        img=func.normalize(img,self.mean,self.std)
        return img, gt