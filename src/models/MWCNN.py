import torch
import os
import sys
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn as nn


class WCNN(nn.Module):
    """
    multi-level wavelet CNN transform
    param: in channels: number of input channels for each image (should be 1 for a single grayscale image or stack)
    param: out channels: number of output channels (default: 4*C)
    param: filter_sz: size of filter (should be 3)
    param: num conv: number of conv-batch_norm-relu layers wanted beyond first
    input:(N,in_ch,H,W)
    output:(N,out_ch,H/2,W/2)

    """

    def __init__(self, in_ch, out_ch=None, filter_sz=3, num_conv=3):
        super(WCNN, self).__init__()
        if out_ch is None:
            out_ch = 4 * in_ch
        self.DwT = DWTForward(J=1, wave='haar', mode='zero')
        # 4 * input channels since DwT creates 4 outputs per image
        modules = []
        modules.append(nn.Conv2d(4 * in_ch, out_ch, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(num_features=out_ch))
        modules.append(nn.ReLU())
        for i in range(num_conv):
            modules.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            modules.append(nn.BatchNorm2d(num_features=out_ch))
            modules.append(nn.ReLU())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        Yl, Yh = self.DwT(x)
        Yh = Yh[0]
        Y_cnn = [Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :], Yl]
        Y_cnn = torch.cat(Y_cnn, dim=1)
        output = self.conv(Y_cnn)
        return output


class IWCNN(nn.Module):
    """
    inverse of WCNN
    param: in_ch: number of input channels for each image
    param: internal_ch: number of output channels for last internal cnn layer (ensure evenly divisble by 4)
    param: filter_sz: size of filter (should be 3)
    param: num_conv: number of conv-batch_norm-relu layers wanted beyond first
    input: (N,in_ch,H,W)
    output: (N,internal_ch/4,2*H,2*W)

    """

    def __init__(self, in_ch, internal_ch=None, filter_sz=3, num_conv=3):
        super(IWCNN, self).__init__()
        if internal_ch is None:
            internal_ch = in_ch
        self.IDwT = DWTInverse(wave='haar', mode='zero')
        modules = []
        for i in range(num_conv):
            modules.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
            modules.append(nn.BatchNorm2d(num_features=in_ch))
            modules.append(nn.ReLU())
        modules.append(nn.Conv2d(in_ch, internal_ch, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(num_features=internal_ch))
        modules.append(nn.ReLU())
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        Y_cnn = self.conv(x)
        N, C, H, W = Y_cnn.shape
        Yh_cnn_list = torch.unsqueeze(Y_cnn[:, 0:3 * int(C / 4), :, :], dim=2).view(N, int(C / 4), 3, H, W)
        Yl_cnn = Y_cnn[:, 3 * int(C / 4):, :, :]
        Yh_list = [Yh_cnn_list]
        Y = self.IDwT((Yl_cnn, Yh_list))
        return Y

if __name__ == "__main__":
    print("testing WCNN")
    X = torch.randn(10, 128, 64, 64)
    N, C, H, W = X.shape
    cnn = WCNN(C, out_ch=4 * C, num_conv=1)
    cnn_2 = WCNN(4 * C, out_ch=16 * C)
    inv_cnn = IWCNN(in_ch=16 * C, internal_ch=16 * C, num_conv=1)
    inv_cnn_2 = IWCNN(in_ch=4 * C, internal_ch=4 * C, num_conv=1)
    Y = cnn(X)  # N,4*C,H,W
    Y = cnn_2(Y)  # N,16*C,H,W

    output = inv_cnn(Y)  # N,4*C,H,W
    output = inv_cnn_2(output)  # N,C,H,W
    print("shape of X: ", X.shape)
    print("shape of Y: ", Y.shape)
    print("shape of output: ", output.shape)
    print(torch.mean(X - output))

