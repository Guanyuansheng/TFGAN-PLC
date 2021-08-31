import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from conv_stft import ConvSTFT, ConviSTFT
from ptflops.flops_counter import get_model_complexity_info


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ELU())
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ELU())
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU())
        conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ELU())
        conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ELU())
        self.Module_list = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.Module_list)):
            x = self.Module_list[i](x)
            x_list.append(x)
        return x, x_list


class GLSTM(nn.Module):
    def __init__(self, i_num, g_num):
        super(GLSTM, self).__init__()
        self.K = g_num
        self.g_feat = i_num // self.K
        self.glstm_list = nn.ModuleList([nn.LSTM(input_size=self.g_feat, hidden_size=self.g_feat, batch_first=True) for i in range(self.K)])

    def forward(self, x):
        batch_num, seq_len, feat_num = x.size()[0], x.size()[1], x.size()[2]
        x = x.reshape(batch_num, seq_len, self.K, self.g_feat)
        h = Variable(torch.zeros(batch_num, seq_len, self.K, self.g_feat)).to(x.device)
        for i in range(self.K):
            h[:, :, i, :], _ = self.glstm_list[i](x[:, :, i, :])
        h = h.permute(0, 1, 3, 2).contiguous()
        h = h.view(batch_num, seq_len, -1)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ELU())
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU())
        deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ELU())
        deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(3, 1), stride=(2, 1)),
            self.pad,
            nn.BatchNorm2d(16),
            nn.ELU())
        deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(1),
            nn.ELU())
        self.Module_list = nn.ModuleList([deconv1, deconv2, deconv3, deconv4, deconv5])
        self.fc = nn.Linear(161, 161)

    def forward(self, x, x_list):
        for i in range(len(self.Module_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.Module_list[i](x)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.fc(x.squeeze(dim=1))
        x = x.permute(0, 2, 1).contiguous()
        return x


class GCRN(nn.Module):
    def __init__(self):
        super(GCRN, self).__init__()
        self.win_len = 320
        self.win_inc = 80
        self.fft_len = 320
        self.win_type = 'hanning'

        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)

        self.encoder = Encoder()
        self.glstm1 = GLSTM(i_num=1024, g_num=1)
        self.glstm2 = GLSTM(i_num=1024, g_num=1)
        self.decoder_real = Decoder()
        self.decoder_imag = Decoder()

    def forward(self, inputs):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        cspecs = torch.stack([real, imag], 1)
        # cspecs = cspecs[:, :, 1:]

        x = cspecs
        x, x_list = self.encoder(x)

        batch_size, channels, dims, lengths = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, lengths, channels * dims)
        x = self.glstm1(x)
        x = self.glstm2(x)
        x = x.view(batch_size, lengths, dims, channels)
        x = x.permute(0, 3, 2, 1).contiguous()

        x_real = self.decoder_real(x, x_list)
        x_imag = self.decoder_imag(x, x_list)
        del x_list

        #out_spec = torch.stack((x_real, x_imag), dim=1)
        out_spec = torch.cat([x_real, x_imag], 1)
        out_wav = self.istft(out_spec)

        # out_wav = torch.squeeze(out_wav, 1)
        # out_wav = torch.tanh(out_wav)
        # out_wav = torch.clamp(out_wav, -1, 1)
        return out_spec, out_wav


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 3200])
    output: torch.Size([3, 1, 3200])
    '''
    model = GCRN()
    x = torch.randn(1, 1, 3200)  # (B, channels, T).
    z = nn.init.uniform_(torch.Tensor(x.shape[0], 3200), -1., 1.)
    print(x.shape)

    out_spec, out_wav = model(x)
    print(out_spec.size())
    print(out_wav.size())

    # model params
    flops, params = get_model_complexity_info(model, (1, 3200), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))
