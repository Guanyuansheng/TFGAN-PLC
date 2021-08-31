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
        
        self.dcdeconv1 = nn.Sequential(ComplexConvTranspose2d(512, 256, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)),
                                       ComplexBatchNorm(256),
                                       cPReLU())
        self.dcdeconv2 = nn.Sequential(ComplexConvTranspose2d(512, 128, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)),
                                       ComplexBatchNorm(128),
                                       cPReLU())
        self.dcdeconv3 = nn.Sequential(ComplexConvTranspose2d(256, 64, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)),
                                       ComplexBatchNorm(64),
                                       cPReLU())
        self.dcdeconv4 = nn.Sequential(ComplexConvTranspose2d(128, 32, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)),
                                       ComplexBatchNorm(32),
                                       cPReLU())
        self.dcdeconv5 = nn.Sequential(ComplexConvTranspose2d(64, 16, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)),
                                       ComplexBatchNorm(16),
                                       cPReLU())
        self.dcdeconv6 = nn.Sequential(ComplexConvTranspose2d(32, 2, kernel_size=(5, 2), stride=(2, 1),
                                                              padding=(2, 0), output_padding=(1, 0)))
        

    def forward(self, inputs):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        # spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        # spec_phase = torch.atan2(imag, real)
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]

        x = cspecs
        x1 = self.dcconv1(x)
        x2 = self.dcconv2(x1)
        x3 = self.dcconv3(x2)
        x4 = self.dcconv4(x3)
        x5 = self.dcconv5(x4)
        x6 = self.dcconv6(x5)
        out = x6
        
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        r_rnn_in = out[:, :, :channels // 2]
        i_rnn_in = out[:, :, channels // 2:]
        r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
        i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

        r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

        r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
        i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
        out = torch.cat([r_rnn_in, i_rnn_in], 2)
        
        out = out.permute(1, 2, 3, 0)
        
        x = complex_cat([x, x6], 1)
        x = self.dcdeconv1(x)
        x = x[..., 1:]
        x = complex_cat([x, x5], 1)
        x = self.dcdeconv2(x)
        x = x[..., 1:]
        x = complex_cat([x, x4], 1)
        x = self.dcdeconv3(x)
        x = x[..., 1:]
        x = complex_cat([x, x3], 1)
        x = self.dcdeconv4(x)
        x = x[..., 1:]
        x = complex_cat([x, x2], 1)
        x = self.dcdeconv5(x)
        x = x[..., 1:]
        x = complex_cat([x, x1], 1)
        x = self.dcdeconv6(x)
        x = x[..., 1:]
    
        mask_real = x[:, 0]
        mask_imag = x[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )

        mask_mags = torch.tanh(mask_mags)
        # out_mask = torch.zeros_like(mask_mags)
        # out_mask.masked_fill_((mask_mags > 0.15) & (mask_mags < 0.85), 1.0)
        # mask_mags = out_mask * mask_mags
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        # out_wav = torch.squeeze(out_wav, 1)
        # out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp(out_wav, -1, 1)
        return out_spec, out_wav

        return x
    
    
if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([3, 1, 2560])
    '''
    model = DCCRN()
    x = torch.randn(3, 1, 3200)  # (B, channels, T).
    z = nn.init.uniform_(torch.Tensor(x.shape[0], 3200), -1., 1.)
    print(x.shape)

    out_spec, out_wav = model(x)
    print(out_spec.size())
    print(out_wav.size())

    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))
