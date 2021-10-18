import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ptflops.flops_counter import get_model_complexity_info


# non-causal version
class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size, 1, padding=2*(kernel_size-1)//2, dilation=2),
                                   nn.BatchNorm1d(output_channel),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv1d(output_channel, output_channel, kernel_size, 1, padding=4*(kernel_size-1)//2, dilation=4),
                                   nn.BatchNorm1d(output_channel),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.in_conv = nn.Conv1d(input_channel, output_channel, kernel_size, 1, padding=kernel_size//2)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.conv2(x)

        x = x + self.in_conv(input)

        return x


class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = nn.MaxPool1d(factor, factor)

    def forward(self, x):
        x = self.down_sample(x)
        return x


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=factor, mode='linear', align_corners=True)

    def forward(self, x):
        return self.up_sample(x)


class GLUBlock(torch.nn.Module):
    def __init__(self, dila_rate, causal_flag):
        super(GLUBlock, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1),
                                     nn.BatchNorm1d(64))
        if causal_flag is True:
            self.pad = nn.ConstantPad1d((int(dila_rate * 10), 0), value=0.)     # 前填后不填
        else:
            self.pad = nn.ConstantPad1d((int(dila_rate * 5), int(dila_rate * 5)), value=0.)     # 前后都填

        self.dila_conv_left = self.left_conv = nn.Sequential(
            nn.PReLU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dila_rate),
            nn.BatchNorm1d(64))
        self.dila_conv_right = nn.Sequential(
            nn.PReLU(),
            self.pad,
            nn.Conv1d(64, 64, kernel_size=11, dilation=dila_rate),
            nn.BatchNorm1d(64),
            nn.Sigmoid())
        self.out_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128))
        self.out_prelu = nn.PReLU()

    def forward(self, input):
        x = input
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + input
        x = self.out_prelu(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #Down Blocks
        self.conv_block1 = ConvBlock(1, 16, 11)
        self.conv_block2 = ConvBlock(16, 32, 11)
        self.conv_block3 = ConvBlock(32, 64, 11)
        self.conv_block4 = ConvBlock(64, 128, 11)
        # self.conv_block5 = ConvBlock(128, 256, 15)

        # GLU Blocks
        self.glu_list = nn.Sequential(GLUBlock(dila_rate=1, causal_flag=True),
                                      GLUBlock(dila_rate=2, causal_flag=True),
                                      GLUBlock(dila_rate=4, causal_flag=True),
                                      GLUBlock(dila_rate=8, causal_flag=True),
                                      GLUBlock(dila_rate=16, causal_flag=True))

        #Up Blocks
        self.conv_block6 = ConvBlock(128+128, 128, 7)
        self.conv_block7 = ConvBlock(128+64, 64, 7)
        self.conv_block8 = ConvBlock(64+32, 32, 7)
        self.conv_block9 = ConvBlock(32+16, 16, 7)

        #Last convolution
        self.last_conv = nn.Sequential(nn.ConvTranspose1d(17, 1, 1),  nn.Tanh())

        self.downsample = DownSample()
        self.upsample = UpSample()

    def forward(self, input):
        x = input
        x1 = self.conv_block1(x)
        x = self.downsample(x1)
        x2 = self.conv_block2(x)
        x = self.downsample(x2)
        x3 = self.conv_block3(x)
        x = self.downsample(x3)
        x4 = self.conv_block4(x)
        x = self.downsample(x4)
        # x5 = self.conv_block5(x)

        x = self.glu_list(x)

        x = self.upsample(x)
        x = torch.cat((x4, x), dim=1)
        x = self.conv_block6(x)

        x = self.upsample(x)
        x = torch.cat((x3, x), dim=1)
        x = self.conv_block7(x)

        x = self.upsample(x)
        x = torch.cat((x2, x), dim=1)
        x = self.conv_block8(x)

        x = self.upsample(x)
        x = torch.cat((x1, x), dim=1)
        x = self.conv_block9(x)

        x = torch.cat((x, input), dim=1)
        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([3, 1, 2560])
    '''
    model = Generator()
    x = torch.randn(1, 1, 2560)  # (B, channels, T).
    print(x.shape)

    out = model(x)
    print(out.size())

    # model params
    flops, params = get_model_complexity_info(model, (1, 2560), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))
