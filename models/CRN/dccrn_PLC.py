import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .conv_stft import ConvSTFT, ConviSTFT

from .complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm, cPReLU

class DCCRN(nn.Module):
    def __init__(self):
        super(DCCRN, self).__init__()
        self.win_len = 400
        self.win_inc = 100
        self.fft_len = 512
        self.win_type = 'hanning'
        self.rnn_layers = 2
        self.rnn_units = 128

        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)

        self.dcconv1 = nn.Sequential(ComplexConv2d(2, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(16),
                                     cPReLU())
        self.dcconv2 = nn.Sequential(ComplexConv2d(16, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(32),
                                     cPReLU())
        self.dcconv3 = nn.Sequential(ComplexConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(64),
                                     cPReLU())
        self.dcconv4 = nn.Sequential(ComplexConv2d(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(128),
                                     cPReLU())
        self.dcconv5 = nn.Sequential(ComplexConv2d(128, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(256),
                                     cPReLU())
        self.dcconv6 = nn.Sequential(ComplexConv2d(256, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     ComplexBatchNorm(256),
                                     cPReLU())

        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))
        rnns = []
        for idx in range(self.rnn_layers):
            rnns.append(
                NavieComplexLSTM(
                    input_size=hidden_dim * 256 if idx == 0 else self.rnn_units,
                    hidden_size=self.rnn_units,
                    bidirectional=True,
                    batch_first=False,
                    projection_dim=hidden_dim * 256 if idx == self.rnn_layers - 1 else None,
                )
            )
            self.enhance = nn.Sequential(*rnns)

    def forward(self, inputs):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        # spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        # spec_phase = torch.atan2(imag, real)
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]

        x = cspecs
        x = self.dcconv1(x)
        x = self.dcconv2(x)
        x = self.dcconv3(x)
        x = self.dcconv4(x)
        x = self.dcconv5(x)
        x = self.dcconv6(x)


        return x
