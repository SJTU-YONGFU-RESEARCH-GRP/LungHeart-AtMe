import torch
import torch.nn as nn
from PLE.attention_augment_conv import AugmentedConv
# from attention_augment_conv import AugmentedConv
from Args import args


def norm(dim, bs=args.batch_size):
    if bs < 32:
        return nn.GroupNorm(min(32, dim), dim)
    else:
        return nn.BatchNorm2d(dim)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_att=False, relatt=False, shape=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(args.dropout)
        # self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.downsample = downsample
        self.conv1 = AugmentedConv(inplanes, planes, kernel_size=3, dk=40, dv=4, Nh=2, relative=relatt, stride=stride, shape=shape) if use_att else conv3x3(inplanes, planes, stride)
        # self.attconv1 =
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        # out = self.relu(self.norm1(x))
        out = self.tanh(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        # out = self.relu(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.droupout(out)
        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Expert(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Expert, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 7, self.stride, 3, bias=True),
            norm(self.out_channel),

            nn.ReLU(inplace=True),
            # nn.Softmax(),
            # nn.Tanh(),

            nn.MaxPool2d(3, self.stride, 1)
        )
        self.exp_inter_layer = nn.Sequential(
            norm(self.out_channel),
            nn.Tanh(),
            nn.MaxPool2d(3, self.stride, 1)
        )
        self.conv2 = conv1x1(self.out_channel, self.out_channel, stride=1)
        # self.conv3 = conv1x1(self.out_channel, self.out_channel, stride=1)
        # self.conv4 = conv1x1(self.out_channel, self.out_channel, stride=2)
        # self.conv5 = conv1x1(self.out_channel, self.out_channel, stride=1)
        # self.ResNet_I = ResBlock(self.out_channel, self.out_channel, stride=self.stride,
        #                          downsample=conv1x1(self.out_channel, self.out_channel, self.stride))
        # self.pool0 = nn.AdaptiveAvgPool2d((28, 28))

    def forward(self, x):
        # print("[ Expert ] before train_model:", x.shape)
        x = self.conv1(x)
        # print('[ Expert ] after 1 conv:', x.shape)
        x = self.exp_inter_layer(x)
        # print('[ Expert ] after inter layer:', x.shape)
        x = self.conv2(x)
        # print('after 2 conv:', x.shape)
        # x = self.conv3(x)
        # print('after 3 conv:', x.shape)
        # x = self.conv4(x)
        # print('after 4 conv:', x.shape)
        # x = self.conv5(x)
        # print('after 5 conv:', x.shape)
        # x = self.ResNet_I(x)
        # print('after ResNet_0_0:', x.shape)

        return x
