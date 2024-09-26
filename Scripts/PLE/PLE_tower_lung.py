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


class Tower_lung(nn.Module):
    # (self, inplanes, planes, shape, drop_rate, stride=1, v=0.2, k=2, Nh=2, downsample=None, attention=False)
    def __init__(self, num):
        super(Tower_lung, self).__init__()
        # self.ResNet_I_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_I_0 = ResBlock(64, 64, downsample=conv1x1(64, 64, 1))
        self.ResNet_II_0 = ResBlock(64, 64)
        self.ResNet_II_1 = ResBlock(64, 64)
        self.ResNet_II_2 = ResBlock(64, 64)
        self.ResNet_II_3 = ResBlock(64, 64)
        self.ResNet_II_4 = ResBlock(64, 64)
        self.ResNet_II_5 = ResBlock(64, 64, use_att=True, relatt=True, shape=args.shape, downsample=conv1x1(64, 64, 1))
        # self.ResNet_II_5 = ResBlock(64, 64, use_att=True, relatt=True, shape=args.shape)
        # self.ResNet_5 = ResBlock(64, 64)
        # self.ResNet_5 = nn.Sequential(nonLocal(64, 32), nn.ReLU(inplace=True))
        self.ResNet_II_6 = ResBlock(64, 64)
        # self.ResNet_7 = ResBlock(64, 64)
        self.norm0 = norm(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.softmax0 = nn.Softmax()
        self.tanh0 = nn.Tanh()
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool1 = nn.MaxPool2d(7, 2, 3)
        self.classifier = nn.Sequential(nn.Linear(64, 64),
        # self.classifier = nn.Sequential(nn.Linear(3136, 64),
                                        # norm(64),
                                        # nn.ReLU(inplace=True),
                                        nn.Dropout(args.dropout),
                                        # nn.Linear(64, 32),
                                        # norm(32),
                                        # nn.ReLU(inplace=True),
                                        # nn.Dropout(0.5),
                                        nn.Linear(64, num),
                                        nn.Dropout(args.dropout)
                                        )
        self.flat = Flatten()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # print("before train_model:", x.shape)
        x = self.ResNet_I_0(x)
        # print('after ResNet_0_1:', x.shape)
        x = self.ResNet_II_0(x)
        # print('after ResNet_0:', x.shape)
        x = self.ResNet_II_1(x)
        # print('after ResNet_1:', x.shape)
        x = self.ResNet_II_2(x)
        # print('after ResNet_2:', x.shape)
        # x = self.ResNet_II_3(x)
        # print('after ResNet_3:', x.shape)
        # x = self.ResNet_II_4(x)
        # print('after ResNet_4:', x.shape)
        x = self.ResNet_II_5(x)
        # print('after ResNet_5:', x.shape)
        x = self.ResNet_II_6(x)
        x = self.dropout(x)
        # print('after ResNet_6:', x.shape)
        # x = self.ResNet_7(x)
        # print('after ResNet_7:', x.shape)
        x = self.norm0(x)
        # print("after norm:", x.shape)

        # x = self.softmax0(x)
        # x = self.tanh0(x)
        x = self.relu0(x)
        # print("after relu:", x.shape)

        x = self.pool0(x)
        # x = self.pool1(x)
        # x = self.pool0(self.relu0(self.norm0(x)))
        # print('after pool0:', x.shape)
        x = self.flat(x)
        # print('after flat:', x.shape)
        x = self.classifier(x)
        # print('after classifier:', x.shape)

        return x