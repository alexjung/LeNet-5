import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


# Inspired by
# https://github.com/pytorch/vision/blob/6c2cda6a0eda4c835f96f18bb2b3be5043d96ad2/torchvision/models/inception.py#L202
class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(6, 8, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(6, 6, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(6, 8, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(6, 8, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(8, 12, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(12, 12, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(6, 10, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, img):
        outputs = self._forward(img)
        return torch.cat(outputs, 1)


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(5472, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.Softmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class MinInception(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(MinInception, self).__init__()

        self.c1 = C1()
        self.inc1 = InceptionA()
        self.f4 = F4() 
        self.f5 = F5() 

    def forward(self, img):
        output = self.c1(img)
        output = self.inc1(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output
