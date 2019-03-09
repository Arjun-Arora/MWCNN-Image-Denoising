import torch
import os
import sys
import torch.nn as nn
from MWCNN import WCNN,IWCNN

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class MW_Unet(nn.Module):
    """
    Baseline architecture for Multi-level Wavelet-CNN paper
    Incorporates Unet style concatenation of dims
    input:N,C,H,W
    output: N,C,H,W
    """
    def __init__(self,num_conv=0,in_ch=1,channel_1=16,channel_2=32):
        '''
        :param: num_conv per contraction and expansion layer, how many extra conv-batch-relu layers wanted
        :param in_ch: number of input channels expected
        :return:
        '''
        super(MW_Unet,self).__init__()
        print("channel_1: {}, channel_2: {}".format(channel_1,channel_2))
        self.num_conv = num_conv
        self.in_ch = in_ch
        self.cnn_1 = WCNN(in_ch=in_ch,out_ch=channel_1,num_conv=num_conv) #output N,160,H/2,W/2
        self.cnn_2 = WCNN(in_ch=channel_1,out_ch=channel_2,num_conv=num_conv)
        self.cnn_3 = WCNN(in_ch=channel_2,out_ch=channel_2,num_conv=num_conv)
        self.icnn_3 = IWCNN(in_ch=channel_2,internal_ch=4*channel_2,num_conv=num_conv)
        self.icnn_2 = IWCNN(in_ch=2*channel_2,internal_ch=4*channel_1,num_conv=num_conv) #expecting 2*256 because of skip connection
        self.icnn_1 = IWCNN(in_ch=2*channel_1,internal_ch=self.in_ch*4,num_conv=num_conv) # output N,in_ch,H,W
        self.final_conv = nn.Conv2d(in_channels=self.in_ch,out_channels=self.in_ch,kernel_size=3,padding=1)

    def forward(self,x):
        x1 = self.cnn_1(x)
        x2 = self.cnn_2(x1)
        x3 = self.cnn_3(x2)

        y1 = self.icnn_3(x3)
        y2 = self.icnn_2(torch.cat((y1,x2),dim=1))
        y3 = self.icnn_1(torch.cat((y2,x1),dim=1))
        output = self.final_conv(y3)
        return output
if __name__ == "__main__":
    print("testing MW_Unet")
    X = torch.randn(10, 4, 64, 64)
    N, C, H, W = X.shape

    Unet = MW_Unet(in_ch=C)
    Unet.apply(init_weights)
    Y = Unet(X)


    print("shape of X: ", X.shape)
    print("shape of Y: ", Y.shape)
    print(torch.mean(X - Y))





