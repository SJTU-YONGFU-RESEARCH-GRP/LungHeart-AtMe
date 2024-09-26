import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from PLE.PLE_experts import Expert
from PLE.PLE_tower import Tower
from PLE.PLE_tower_lung import Tower_lung
# from PLE_experts import Expert
# from PLE_tower import Tower
# from PLE_tower_lung import Tower_lung
from Args import args

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def norm(dim, bs=args.batch_size):
    if bs < 32:
        return nn.GroupNorm(min(32, dim), dim)
    else:
        return nn.BatchNorm2d(dim)


class PLE(nn.Module):
    def __init__(self, num_specific_experts, num_shared_experts):
        super(PLE, self).__init__()
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts

        # experts layer 1
        self.experts_shared_0 = nn.ModuleList([Expert(3, 64, 2) for i in range(self.num_shared_experts)])
        self.experts_lung_0 = nn.ModuleList([Expert(3, 64, 2) for i in range(self.num_specific_experts)])
        self.experts_heart_0 = nn.ModuleList([Expert(3, 64, 2) for i in range(self.num_specific_experts)])
        # experts layer 2
        # self.experts_shared_1 = nn.ModuleList([Expert(64, 64, 1) for i in range(self.num_shared_experts)])
        # self.experts_lung_1 = nn.ModuleList([Expert(64, 64, 1) for i in range(self.num_specific_experts)])
        # self.experts_heart_1 = nn.ModuleList([Expert(64, 64, 1) for i in range(self.num_specific_experts)])

        # selector layer 1
        self.selec0_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=True),
            norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # self.selec0_conv2 = conv1x1(64, 64, 2)
        self.selec0_pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.selec0_flat = Flatten()
        self.selector_lung_0 = nn.Sequential(
            nn.Linear(64, self.num_specific_experts+self.num_shared_experts), nn.Softmax())
        self.selector_heart_0 = nn.Sequential(
            nn.Linear(64, self.num_specific_experts + self.num_shared_experts), nn.Softmax())
        # self.selector_share_0 = nn.Sequential(
        #     nn.Linear(64, 2 * self.num_specific_experts + self.num_shared_experts), nn.Softmax())
        # selector layer 2
        # self.selec1_pool0 = nn.AdaptiveAvgPool2d((1, 1))
        # self.selec1_flat = Flatten()
        # self.selector_lung_1 = nn.Sequential(
        #     nn.Linear(64, self.num_specific_experts + self.num_shared_experts), nn.Softmax())
        # self.selector_heart_1 = nn.Sequential(
        #     nn.Linear(64, self.num_specific_experts + self.num_shared_experts), nn.Softmax())

        # tower
        self.tower_lung = Tower(num=4)
        self.tower_heart = Tower(num=2)

    def forward(self, x):
        # experts layer 1
        exp_shared_0 = [e(x) for e in self.experts_shared_0]
        print("[ PLE ] after expert:", len(exp_shared_0))
        print("[ PLE ] after expert:", exp_shared_0[0].shape)
        exp_shared_0 = torch.stack(exp_shared_0)
        exp_lung_0 = [e(x) for e in self.experts_lung_0]
        exp_lung_0 = torch.stack(exp_lung_0)
        exp_heart_0 = [e(x) for e in self.experts_heart_0]
        exp_heart_0 = torch.stack(exp_heart_0)

        # gates layer 1
        # x = self.selec_conv1(x)
        # x_lung = self.selec_pool0(x)
        # x_lung = self.selec_flat(x_lung)
        # x_heart = self.selec_conv2(x)
        # x_heart = self.selec_conv3(x_heart)
        # x_heart = self.selec_pool0(x_heart)
        # x_heart = self.selec_flat(x_heart)
        # selector_lung_0 = self.selector_lung_0(x_lung)
        # selector_heart_0 = self.selector_heart_0(x_heart)
        x = self.selec0_conv1(x)
        # x = self.selec0_conv2(x)
        # print("[ PLE ] before AvgPool:", x.shape)
        x = self.selec0_pool0(x)
        # print("[ PLE ] after AvgPool:", x.shape)
        x = self.selec0_flat(x)
        # print("[ PLE ] after Flatten:", x.shape)
        selector_lung_0 = self.selector_lung_0(x)
        selector_heart_0 = self.selector_heart_0(x)
        # selector_share_0 = self.selector_share_0(x)

        # weighted_out layer 1
        weighted_lung_0 = torch.cat((exp_lung_0, exp_shared_0), dim=0)
        print("[ PLE ] after cat:", weighted_lung_0.shape)
        layer_out_lung_0 = torch.einsum('abcde, ba -> bcde', weighted_lung_0, selector_lung_0)
        print("[ PLE ] after weighting:", layer_out_lung_0.shape)

        # print("before tower: ", layer_out_lung_0.shape)

        weighted_heart_0 = torch.cat((exp_shared_0, exp_heart_0), dim=0)
        layer_out_heart_0 = torch.einsum('abcde, ba -> bcde', weighted_heart_0, selector_heart_0)
        print("[ PLE ] before tower: ", layer_out_heart_0.shape)

        # weighted_share_0 = torch.cat((exp_lung_0, exp_shared_0, exp_heart_0), dim=0)
        # layer_out_shared = torch.einsum('abcde, ba -> bcde', weighted_share_0, selector_share_0)
        #
        #
        # # experts layer 2
        # exp_shared_1 = [e(layer_out_shared) for e in self.experts_shared_1]
        # exp_shared_1 = torch.stack(exp_shared_1)
        # exp_lung_1 = [e(layer_out_lung_0) for e in self.experts_lung_1]
        # exp_lung_1 = torch.stack(exp_lung_1)
        # exp_heart_1 = [e(layer_out_heart_0) for e in self.experts_heart_1]
        # exp_heart_1 = torch.stack(exp_heart_1)
        #
        # # selector layer 2
        # selec_lung = self.selec1_pool0(layer_out_lung_0)
        # selec_lung = self.selec1_flat(selec_lung)
        # selec_heart = self.selec1_pool0(layer_out_heart_0)
        # selec_heart = self.selec1_flat(selec_heart)
        # selector_lung_1 = self.selector_lung_1(selec_lung)
        # selector_heart_1 = self.selector_heart_1(selec_heart)
        #
        # # weighted_out layer 2
        # weighted_lung_1 = torch.cat((exp_lung_1, exp_shared_1), dim=0)
        # layer_out_lung_1 = torch.einsum('abcde, ba -> bcde', weighted_lung_1, selector_lung_1)
        #
        # weighted_heart_1 = torch.cat((exp_shared_1, exp_heart_1), dim=0)
        # layer_out_heart_1 = torch.einsum('abcde, ba -> bcde', weighted_heart_1, selector_heart_1)


        # tower
        lung = self.tower_lung(layer_out_lung_0)
        heart = self.tower_heart(layer_out_heart_0)

        out = torch.cat((lung, heart), dim=1)
        # final_lung = torch.cat((torch.unsqueeze(heart[:,0],dim=1),torch.unsqueeze(lung[:,0],dim=1)),dim=1)
        # out = torch.cat((final_lung, lung[:,1:3], heart[:,1:3]),dim=1)

        return out


class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()

        # experts layer 1
        self.experts_shared_0 = Expert(3, 64, 2)

        # tower
        self.tower_lung = Tower(num=6)

    def forward(self, x):
        # experts layer 1
        exp_shared_0 = self.experts_shared_0(x)
        # print("[ CNN ] after expert:", exp_shared_0[0].shape)
        # exp_shared_0 = torch.stack(exp_shared_0)
        # print("[ CNN ] after stack:", exp_shared_0.shape)

        # tower
        out = self.tower_lung(exp_shared_0)
        # print("[ CNN ] after tower:", out.shape)

        # out = torch.cat((lung, heart), dim=1)
        # final_lung = torch.cat((torch.unsqueeze(heart[:,0],dim=1),torch.unsqueeze(lung[:,0],dim=1)),dim=1)
        # out = torch.cat((final_lung, lung[:,1:3], heart[:,1:3]),dim=1)

        return out


def PLE_base(num_special=1, num_share=1):
    model = PLE(num_specific_experts=num_special, num_shared_experts=num_share)
    return model


def CNN():
    model = Simple_CNN()
    return model


if __name__ == '__main__':
    input_tdim = 224
    input_fdim = 224
    bs = 10
    in_c = 3
    num_share = 1
    num_special = 1
    model1 = PLE_base(num_special=num_special, num_share=num_share)
    model2 = CNN()
    # input a batch of 10 spectrogram, each with 1024 time frames and 128 frequency bins
    test_input = torch.rand([bs, in_c, input_tdim, input_fdim])

    spec_p = "/home/kaswary/AICAS_AtMe/Dataset/aug_denoise_stft_mfcc/test/crackle/104_1b1_Ll_sc_Litt32004.png"
    spec = Image.open(spec_p)
    data_trans = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    spec_t = data_trans(spec).unsqueeze(dim=0)
    test_out1 = model1(spec_t)
    test_out2 = model2(spec_t)
    print(test_out1.shape)
    print(test_out2.shape)
    # test_output = model(test_input)
    # output should be in shape [10, 6], i.e., 10 samples, each with prediction of 6 classes.
    # print(test_output.shape)

    # input_tdim = 224
    # input_fdim = 224
    # num_share = 2
    # num_special = 2
    # model = PLE_base(num_special=num_special, num_share=num_share)
    # # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    # test_input = torch.rand([bs, in_c, input_tdim, input_fdim])
    # test_output = model(test_input)
    # # output should be in shape [10, 6], i.e., 10 samples, each with prediction of 6 classes.
    # print(test_output.shape)