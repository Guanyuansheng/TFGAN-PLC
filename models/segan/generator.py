import torch
import torch.nn as nn
import numpy as np
# from sru import SRU
from ptflops.flops_counter import get_model_complexity_info

class GLUBlock(torch.nn.Module):
    def __init__(self, input_size, dila_rate=2):
        super(GLUBlock, self).__init__()
        self.in_conv = nn.Conv1d(input_size, input_size//2, 1)
        self.dila_conv_left = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                            nn.Conv1d(input_size//2, input_size//2, kernel_size=5, stride=1,
                                                      padding=np.int(dila_rate * 2), dilation=dila_rate))
        self.dila_conv_right = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                             nn.Conv1d(input_size//2, input_size//2, kernel_size=5, stride=1,
                                                       padding=np.int(dila_rate * 2), dilation=dila_rate),
                                             nn.Sigmoid())
        self.out_conv = nn.Conv1d(input_size//2, input_size, 1, dilation=dila_rate*2)
        # self.out_lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        x = input
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + input
        # x = self.out_lrelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, frame_size=15, stride=2):
        super(Generator, self).__init__()
        padding = frame_size // 2

        # encoder gets a noisy signal as input
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=frame_size, stride=stride,
                                            padding=padding, bias=False),
                                  nn.BatchNorm1d(16))
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(32))
        self.enc3 = nn.Sequential(nn.Conv1d(32, 32, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(32))
        self.enc4 = nn.Sequential(nn.Conv1d(32, 64, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(64))
        self.enc5 = nn.Sequential(nn.Conv1d(64, 64, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(64))
        self.enc6 = nn.Sequential(nn.Conv1d(64, 128, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(128))
        self.enc7 = nn.Sequential(nn.Conv1d(128, 128, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(128))
        self.enc8 = nn.Sequential(nn.Conv1d(128, 256, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(256))
        self.enc9 = nn.Sequential(nn.Conv1d(256, 256, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(256))
        self.enc10 = nn.Sequential(nn.Conv1d(256, 512, frame_size, stride, padding, bias=False),
                                  nn.BatchNorm1d(512))
        self.enc11 = nn.Sequential(nn.Conv1d(512, 1024, frame_size, stride, padding, bias=False),
                                   nn.BatchNorm1d(1024))

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.z_fc = nn.Linear(in_features=100, out_features=8192)
        self.relu = nn.ReLU(inplace=True)

        # decoder generates an enhanced signal, input: [Bx128x20]
        # each decoder output are concatenated with homologous encoder output,
        # so the feature map sizes are doubled
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(2048, 512, 5, stride, 2, 1),
                                  nn.BatchNorm1d(512))
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(1024, 256, 5, stride, 2, 1),
                                  nn.BatchNorm1d(256))
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(512, 256, 5, stride, 2, 1),
                                  nn.BatchNorm1d(256))
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(512, 128, 5, stride, 2, 1),
                                  nn.BatchNorm1d(128))
        self.dec5 = nn.Sequential(nn.ConvTranspose1d(256, 128, 5, stride, 2, 1),
                                  nn.BatchNorm1d(128))
        self.dec6 = nn.Sequential(nn.ConvTranspose1d(256, 64, 5, stride, 2, 1),
                                  nn.BatchNorm1d(64))
        self.dec7 = nn.Sequential(nn.ConvTranspose1d(128, 64, 5, stride, 2, 1),
                                  nn.BatchNorm1d(64))
        self.dec8 = nn.Sequential(nn.ConvTranspose1d(128, 32, 5, stride, 2, 1),
                                  nn.BatchNorm1d(32))
        self.dec9 = nn.Sequential(nn.ConvTranspose1d(64, 32, 5, stride, 2, 1),
                                  nn.BatchNorm1d(32))
        self.dec10 = nn.Sequential(nn.ConvTranspose1d(64, 16, 5, stride, 2, 1),
                                  nn.BatchNorm1d(16))
        self.dec_final = nn.Sequential(nn.ConvTranspose1d(32, 1, 5, stride, 2, 1),
                                       nn.Tanh())

    def forward(self, x):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal), size = [Bx1x2560]
            z: latent vector
        """
        # encoding step
        e1 = self.enc1(x)   # [Bx16x1280]
        e2 = self.enc2(self.activation(e1))  # [Bx32x640]
        e3 = self.enc3(self.activation(e2))  # [Bx32x320]
        e4 = self.enc4(self.activation(e3))  # [Bx64x160]
        e5 = self.enc5(self.activation(e4))  # [Bx64x80]
        e6 = self.enc6(self.activation(e5))  # [Bx128x40]
        e7 = self.enc7(self.activation(e6))   # [Bx128x20]
        e8 = self.enc8(self.activation(e7))
        e9 = self.enc9(self.activation(e8))
        e10 = self.enc10(self.activation(e9))
        e11 = self.enc11(self.activation(e10))
        # c = compressed feature, the 'thought vector'
        c = self.activation(e11)

        # concatenate the thought vector with latent variable
        z = nn.init.normal_(torch.Tensor(1, 100))
        z = self.z_fc(z)
        z = torch.reshape(z, c.shape)
        z = self.relu(z)
        encoded = torch.cat((z, c), dim=1)

        # decoding step,
        d1 = self.dec1(encoded)
        d1_c = self.activation(torch.cat((d1, e10), dim=1))
        d2 = self.dec2(d1_c)  # [Bx128x40]
        d2_c = self.activation(torch.cat((d2, e9), dim=1))
        d3 = self.dec3(d2_c)  # [Bx128x40]
        d3_c = self.activation(torch.cat((d3, e8), dim=1))
        d4 = self.dec4(d3_c)
        d4_c = self.activation(torch.cat((d4, e7), dim=1))  # [Bx256x40]
        d5 = self.dec5(d4_c)  # [Bx64x80]
        d5_c = self.activation(torch.cat((d5, e6), dim=1))  # [Bx128x80]
        d6 = self.dec6(d5_c)  # [Bx64x160]
        d6_c = self.activation(torch.cat((d6, e5), dim=1))  # [Bx128x160]
        d7 = self.dec7(d6_c)  # [Bx32x320]
        d7_c = self.activation(torch.cat((d7, e4), dim=1))  # [Bx64x320]
        d8 = self.dec8(d7_c)  # [Bx32x640]
        d8_c = self.activation(torch.cat((d8, e3), dim=1))  # [Bx64x640]
        d9 = self.dec9(d8_c)  # [Bx16x1280]
        d9_c = self.activation(torch.cat((d9, e2), dim=1))  # [Bx32x1280]
        d10 = self.dec10(d9_c)
        d10_c = self.activation(torch.cat((d10, e1), dim=1))
        out = self.dec_final(d10_c)  # [Bx1x2560]
        return out


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([3, 1, 2560])
    '''
    model = Generator()

    x = torch.randn(1, 1, 16384)  # (B, channels, T).

    print(x.shape)

    out = model(x)
    print(out.shape)
    # model params
    flops, params = get_model_complexity_info(model, (1, 16384), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))