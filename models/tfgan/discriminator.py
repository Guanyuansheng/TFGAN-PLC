import torch
import torch.nn as nn

from .complexnn import ComplexConv2d, ConvSTFT


class timeDiscriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super(timeDiscriminator, self).__init__()
        self.dropout1 = nn.Dropout(dropout_drop)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=5),
                                   nn.BatchNorm1d(16),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv1_e = nn.Sequential(nn.Conv1d(16, 16, 11, 2, 5),
                                     nn.BatchNorm1d(16),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 11, 2, 5),
                                   nn.BatchNorm1d(32),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 32, 11, 2, 5),
                                   nn.BatchNorm1d(32),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv1d(32, 64, 11, 2, 5),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, 11, 2, 5),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv6 = nn.Sequential(nn.Conv1d(64, 128, 11, 2, 5),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv7 = nn.Sequential(nn.Conv1d(128, 128, 11, 2, 5),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.fully_connected = nn.Linear(in_features=2560, out_features=1)
        self.fully_connected_e = nn.Linear(in_features=160, out_features=1)

    def forward(self, x):
        """
        Forward pass of discriminator.
        Args:
            x: input batch (signal), size = [Bx1x2560]
        """
        if x.shape[-1] == 2560:
            x = self.conv1(x)   # [Bx16x1280]
            x = self.conv2(x)   # [Bx32x640]
            x = self.conv3(x)   # [Bx32x320]
            x = self.conv4(x)   # [Bx64x160]
            x = self.conv5(x)   # [Bx64x80]
            x = self.conv6(x)   # [Bx128x40]
            x = self.conv7(x)   # [Bx128x20]
            # Flatten
            x = x.view(-1, 2560)
            x = self.fully_connected(x)
        elif x.shape[-1] == 160:
            x = self.conv1(x)  # [Bx16x80]
            x = self.conv1_e(x)  # [Bx16x40]
            x = self.conv2(x)  # [Bx32x20]
            x = self.conv3(x)  # [Bx32x10]
            x = self.conv3(x)  # [Bx32x5]
            # Flatten
            x = x.view(-1, 160)
            x = self.fully_connected_e(x)
        return x


class stft_Discriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super(stft_Discriminator, self).__init__()
        self.dropout1 = nn.Dropout(dropout_drop)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.outc = nn.Sequential(nn.Conv2d(512, 1, 1),
                                             nn.Sigmoid())

    def forward(self, x):
        # x: input batch (signal), size = [Bx3x11x1024]
        x = self.conv1(x)  # [Bx3x16x1024]
        x = self.conv2(x)  # [Bx3x32x512]
        x = self.conv3(x)  # [Bx3x32x256]
        x = self.conv4(x)  # [Bx3x64x128]
        x = self.pool(x)
        # Flatten
        x = self.outc(x)
        return x


class DC_Discriminator(nn.Module):
    def __init__(self):
        super(DC_Discriminator, self).__init__()
        self.win_len = [320, 600, 1000]
        self.win_inc = [160, 150, 300]
        self.fft_len = [320, 1024, 2048]
        self.win_type = 'hanning'

        self.stft = ConvSTFT(self.win_len[0], self.win_inc[0], self.fft_len[0], self.win_type, 'complex', fix=True)
        # self.stft1 = ConvSTFT(self.win_len[1], self.win_inc[1], self.fft_len[1], self.win_type, 'complex', fix=True)
        # self.stft2 = ConvSTFT(self.win_len[2], self.win_inc[2], self.fft_len[2], self.win_type, 'complex', fix=True)
        # self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)

        self.dcconv1 = nn.Sequential(ComplexConv2d(2, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(16),
                                     nn.PReLU())
        self.dcconv2 = nn.Sequential(ComplexConv2d(16, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(32),
                                     nn.PReLU())
        self.dcconv3 = nn.Sequential(ComplexConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(64),
                                     nn.PReLU())
        self.dcconv4 = nn.Sequential(ComplexConv2d(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.PReLU())
        self.dcconv5 = nn.Sequential(ComplexConv2d(128, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.PReLU())
        self.dcconv6 = nn.Sequential(ComplexConv2d(256, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.PReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.outc = nn.Sequential(nn.Conv2d(256, 1, 1),
                                  nn.Sigmoid())

    def forward(self, inputs):
        specs = self.stft(inputs)
        real = specs[:, :320//2 + 1]
        imag = specs[:, 320//2 + 1:, :]

        # specs1 = self.stft1(inputs)
        # real1 = specs1[:, :1024 // 2].reshape(specs1.shape[0], 256, 40)
        # imag1 = specs1[:, 1024 // 2 + 2:].reshape(specs1.shape[0], 256, 40)
        #
        # specs2 = self.stft2(inputs)
        # real2 = specs2[:, :2048 // 2].reshape(specs2.shape[0], 256, 40)
        # imag2 = specs2[:, 2048 // 2 + 2:].reshape(specs2.shape[0], 256, 40)

        cspecs = torch.stack([real, imag], 1)
        # cspecs = cspecs[:, :, 1:]

        x = cspecs
        x = self.dcconv1(x)
        x = self.dcconv2(x)
        x = self.dcconv3(x)
        x = self.dcconv4(x)
        x = self.dcconv5(x)
        x = self.dcconv6(x)

        x = self.pool(x)
        x = self.outc(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.tdisc = timeDiscriminator()
        self.fdisc = DC_Discriminator()

    def forward(self, x):
        dt = self.tdisc(x)
        df = self.fdisc(x).view(-1, 1)
        y = dt + df
        return y


if __name__ == '__main__':
    model = Discriminator()
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([])
    '''

    x = torch.randn(3, 1, 2560)
    x1 = torch.randn(3, 3, 44, 256)
    print(x1.shape)

    out = model(x)
    print(out.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))

