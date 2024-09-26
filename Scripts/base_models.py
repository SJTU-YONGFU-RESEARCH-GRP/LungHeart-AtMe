# coding=utf-8
import torch.nn as nn
import torch
from Args import args
import torch.nn.functional as F


# 定义ResNet18/34的残差结构，为2个3x3的卷积
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1

    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out


# 定义ResNet50/101/152的残差结构，为1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # expansion是指在每个小残差块内，减小尺度增加维度的倍数，如64*4=256
    # Bottleneck层输出通道是输入的4倍
    expansion = 4

    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构，专门用来改变x的通道数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out


# 定义ResNet类
class ResNet(nn.Module):
    # 初始化函数
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        # maxpool的输出通道数为64，残差结构输入通道数为64
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 浅层的stride=1，深层的stride=2
        # block：定义的两种残差模块
        # block_num：模块中残差块的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 自适应平均池化，指定输出（H，W），通道数不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层
            # self.droupout = nn.Dropout(args.dropout)
            # self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 64),
                                        # norm(64),
                                        # nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        # nn.Linear(64, 32),6
                                        # norm(32),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(64, num_classes)
                                        # nn.Dropout(0.3)
                                        )
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 定义残差模块，由若干个残差块组成
    # block：定义的两种残差模块，channel：该模块中所有卷积层的基准通道数。block_num：模块中残差块的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 如果满足条件，则是虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # Sequential：自定义顺序连接成模型，生成网络结构
        return nn.Sequential(*layers)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 无论哪种ResNet，都需要的静态层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# ResNet()中block参数对应的位置是BasicBlock或Bottleneck
# ResNet()中blocks_num[0-3]对应[3, 4, 6, 3]，表示残差模块中的残差数
# 34层的resnet
def resnet34(num_classes=args.num_classes, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 50层的resnet
def resnet50(num_classes=args.num_classes, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 101层的resnet
def resnet101(num_classes=args.num_classes, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


# 152层的resnet
def resnet152(num_classes=args.num_classes, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


class AlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self, num_classes):
        # super：引入父类的初始化方法给子类进行初始化
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        # 卷积层，输入大小为224*224，输出大小为55*55，输入通道为3，输出为96，卷积核为11，步长为4
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()
        # MaxPool2d：最大池化操作
        # 最大池化层，输入大小为55*55，输出大小为27*27，输入通道为96，输出为96，池化核为3，步长为2
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为27*27，输出大小为27*27，输入通道为96，输出为256，卷积核为5，扩充边缘为2，步长为1
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 最大池化层，输入大小为27*27，输出大小为13*13，输入通道为256，输出为256，池化核为3，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为256，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为256，卷积核为3，扩充边缘为1，步长为1
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 最大池化层，输入大小为13*13，输出大小为6*6，输入通道为256，输出为256，池化核为3，步长为2
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(6 * 6 * 256, 4096)
        self.f7 = nn.Linear(4096, 4096)
        # 全连接层&softmax
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, self.num_classes)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        # x = F.dropout(x, p=0.5)
        x = self.f7(x)
        # x = F.dropout(x, p=0.5)
        x = self.f8(x)
        # x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x


def alexnet(num_classes=args.num_classes):
    return AlexNet(num_classes=num_classes)


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
                nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01 ** 2)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def test_output_shape(self):
        # out_channel input_channel, image_size
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.features:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)


def vgg19(num_classes=args.num_classes):
    return VGG19(num_classes=num_classes)