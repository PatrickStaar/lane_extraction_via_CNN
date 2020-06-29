import torch
import torch.nn as nn
import resnet
import torch.nn.functional as F


class FCNResNet(nn.Module):
    def __init__(self):
        super(FCNResNet,self).__init__()
        
        self.encoder=resnet.resnet18(no_top=True)

        self.deconv_1=nn.ConvTranspose2d(512, 256, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_2=nn.ConvTranspose2d(256, 128, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_3=nn.ConvTranspose2d(128, 64, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_4=nn.ConvTranspose2d(64, 32, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=False)
        self.deconv_5=nn.ConvTranspose2d(32, 1, kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        features=self.encoder(x)
        
        x=self.deconv_1(features[-1])
        x=F.relu(x)
        x=self.deconv_2(x)
        x=F.relu(x)
        x=self.deconv_3(x)
        x=F.relu(x)
        x=self.deconv_4(x)
        x=F.relu(x)
        x=self.deconv_5(x)       
        x=torch.sigmoid(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        


