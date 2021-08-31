import torch
import torch.nn as nn
import numpy as np


class GLUBlock(torch.nn.Module):
    def __init__(self, input_size, dila_rate):
        super(GLUBlock, self).__init__()
        self.in_conv = nn.Conv1d(input_size, input_size//2, 1)
        self.dila_conv_left = nn.Sequential(nn.LeakyReLU(0.2),
                                            nn.Conv1d(input_size//2, input_size//2, kernel_size=5, stride=1,
                                                      padding=np.int(dila_rate * 2), dilation=dila_rate))
        self.dila_conv_right = nn.Sequential(nn.LeakyReLU(0.2),
                                             nn.Conv1d(input_size//2, input_size//2, kernel_size=5, stride=1,
                                                      padding=np.int(dila_rate * 2), dilation=dila_rate),
                                             nn.Sigmoid())
        self.out_conv = nn.Conv1d(input_size//2, input_size, 1)
        self.out_lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + input
        x = self.out_lrelu(x)
        return x


class TCRBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, dila_rate):
        super(TCRBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_channel, output_channel//2, 11, 2, padding=5),
                                   nn.BatchNorm1d(output_channel//2),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(output_channel//2, output_channel, 11, 1, padding=5),
                                   nn.BatchNorm1d(output_channel),
                                   nn.LeakyReLU(0.2))
        self.glu = GLUBlock(output_channel, dila_rate)
        self.deconv = nn.ConvTranspose1d(output_channel, input_channel, 11, 2, padding=5, output_padding=1)
        self.out_lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.glu(x)
        x = self.deconv(x)
        x = x + input
        # x = torch.cat((x, input), dim=1)
        x = self.out_lrelu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.in_conv = nn.Sequential(nn.Conv1d(1, 16, 11, padding=5), nn.LeakyReLU(0.2))

        self.tcr_block1 = TCRBlock(16, 32, 2)
        self.tcr_block2 = TCRBlock(16, 64, 4)
        self.tcr_block3 = TCRBlock(16, 128, 8)
        # self.tcr_block4 = TCRBlock(16, 256, 16)
        self.tcr_block5 = TCRBlock(16, 128, 8)
        self.tcr_block6 = TCRBlock(16, 64, 4)
        self.tcr_block7 = TCRBlock(16, 32, 2)

        #Last convolution
        self.last_conv = nn.Sequential(nn.ConvTranspose1d(16, 1, 1), torch.nn.Tanh())

    def forward(self, z, x):
        # z = torch.reshape(z, (x.shape[0], 2, 1280))
        z = self.relu(z)

        x = self.in_conv(x)

        x = self.tcr_block1(x)
        x = self.tcr_block2(x)
        x = self.tcr_block3(x)
        # x = self.tcr_block4(x)
        x = self.tcr_block5(x)
        x = self.tcr_block6(x)
        x = self.tcr_block7(x)

        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([3, 1, 2560])
    '''
    model = Generator()
    x = torch.randn(3, 1, 2560)  # (B, channels, T).
    z = nn.init.uniform_(torch.Tensor(x.shape[0], 2560), -1., 1.)
    print(x.shape)

    out = model(z, x)
    print(out.size())

    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))