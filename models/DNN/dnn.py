import torch
import torch.nn as nn
import os
import sys
import math
import torchaudio.functional as F
from ptflops.flops_counter import get_model_complexity_info


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=320,
        n_hop=160,
        center=False,
        kind_window='hann'
    ):
        super(STFT, self).__init__()

        # choose stft window
        if kind_window == 'bartlett':
            window = torch.bartlett_window(n_fft)
        elif kind_window == 'hamming':
            window = torch.hamming_window(n_fft)
        elif kind_window == 'hann':
            window = torch.hann_window(n_fft)
        else:
            raise NotImplementedError
        self.window = nn.Parameter(window, requires_grad=False)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f

    def istft(self, x):
        return F.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode='reflect'
        )


class BiasLayer(nn.Module):
    def __init__(self, init_bias, init_scale):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(init_bias)
        self.scale = nn.Parameter(init_scale)

    def forward(self, x, norm, rescale=False):
        biased_x = x - self.bias if norm else x + self.bias
        if rescale:
            return biased_x / self.scale
        return biased_x


class AmplitudeEstimator(nn.Module):
    def __init__(
            self,
            phase_features_dim,
            init_bias,
            n_fft=320,
            n_hop=1024,
            context_frames=5,
            seq_duration=6.0,
    ):
        super(AmplitudeEstimator, self).__init__()
        # nb_channels = 2
        # sample_rate = 44100
        # nb_timesteps = int(sample_rate * seq_duration)
        # d_in = nb_channels * nfft//2 + nb_timesteps // (nhop+1) + 2
        self.bias_layer = BiasLayer(*init_bias)

        # amplitude layers
        self.conv_A = nn.Sequential(
            nn.ReflectionPad2d((0, 0, context_frames, context_frames)),
            nn.Conv2d(n_fft // 2 + 1, 2048, (2 * context_frames + 1, 1)),
            # nn.Linear(2049, 2048),
            nn.ReLU()
        )
        self.fc_A = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # phase layers
        self.conv_phi = nn.Sequential(
            nn.Conv3d(n_fft // 2 + 1, 2048, (2 * context_frames + 1, 1, 1), padding=(context_frames, 0, 0)),
            # nn.Linear(2049, 2048),
            nn.ReLU()
        )
        self.fc_phi = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # combining layers
        self.fc_final = nn.Sequential(
            nn.Linear(2048, n_fft // 2 + 1),
            nn.ReLU()
        )

        self.reshape = nn.Linear(phase_features_dim + 1, 1)

    def forward(self, amplitude, phase_features):
        """Estimate the amplitude of the unmixed signal
        Parameters
        ----------
        amplitude: torch.tensor, shape (batch_size, nb_channels, L//(nhop+1)+1, nfft//2+1)
            amplitude of the mixture signal
        phase_features: torch.tensor shape (batch_size, nb_channels, L//(nhop+1)+1, phase_features_dim, nfft//2+1)
            phase of the mixture signal
        Returns
        -------
        torch.tensor
            amplitude of the unmixed signal
        """
        # extract features from amplitude
        A = self.bias_layer(amplitude, True, rescale=True)

        A = self.conv_A(A.transpose(-3, -1)).transpose(-3, -1)
        A = self.fc_A(A)

        # extract features from phase features
        phi = self.conv_phi(phase_features.transpose(-4, -1)).transpose(-4, -1)
        phi = self.fc_phi(phi)
        # print(A.shape, phi.shape)
        A = A.unsqueeze(-3)
        features = torch.cat((A, phi), dim=-3)
        features = self.fc_final(features)
        # print(features.shape)
        features = self.reshape(features.transpose(-3,-1)).squeeze(-1)
        # print(features.shape)
        features = self.bias_layer(features.transpose(-2,-1), False)

        return features



class DNN(nn.Module):
    def __init__(
        self,
        init_bias,
        n_fft=320,
        n_hop=160,
        context_frames=5,
    ):
        super(DNN, self).__init__()

        # self.n_fft = n_fft
        # self.n_hop = n_hop

        # input transformation
        self.stft = STFT(n_fft, n_hop, kind_window='hann')

        # phase preprocessing
        phase_features_dim = 2
        self.phase_shift = nn.Parameter(
            2 * math.pi * n_hop / n_fft * torch.arange(n_fft//2+1),
            requires_grad=False
        )

        self.estimator = AmplitudeEstimator(
            phase_features_dim,
            init_bias,
            n_fft=n_fft,
            n_hop=n_hop,
            context_frames=context_frames
        )

    def forward(self, x):
        """
        Input: (batch_size, nb_channels, nb_timesteps)
        Output:() # TODO: find appropriate output
        """
        X = self.stft(x).transpose(-3, -2)

        A, phi = F.complex_norm(X), F.angle(X)

        phase_features = self.compute_features(phi)

        A_hat = self.estimator(A, phase_features)

        phase = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)

        Y_hat = A_hat.unsqueeze(-1) * phase

        return Y_hat.transpose(-3, -2)

    def compute_features(self, phi):
        """
        Input: (B, C, T, N)
        Output:(B, C, phase_features_dim, T, N)
        """
        dt_phi = self.derivative(phi, -2)
        df_phi = self.derivative(phi, -1)

        # # display dt_phi distribution
        # import matplotlib.pyplot as plt
        # plt.hist(dt_phi.reshape(-1).cpu().numpy(), bins=1000)
        # plt.title('$\\Delta_t \\varphi$ (before preprocessing)')
        # plt.show()
        # plt.hist(df_phi.reshape(-1).cpu().numpy(), bins=1000)
        # plt.title('$\\Delta_f \\varphi$ (before preprocessing)')
        # plt.show()

        # for i in range(5):
        #     print('b', i)
        #     plt.hist(dt_phi[...,i].reshape(-1).cpu().numpy(), bins=1000)
        #     plt.show()
        # display df_phi distribution

        dt_phi -= self.phase_shift
        df_phi -= math.pi

        # for i in range(5):
        #     print('a', i)
        #     plt.hist(dt_phi[...,i].reshape(-1).cpu().numpy(), bins=1000)
        #     plt.show()
        # print('done')
        d_phi = torch.stack((df_phi, dt_phi), dim=-3)
        d_phi = (d_phi + math.pi) % (2*math.pi) - math.pi

        # plt.hist(d_phi[...,1,:].reshape(-1).cpu().numpy(), bins=1000)
        # plt.title('$\\Delta_t \\varphi$ (after preprocessing)')
        # plt.show()
        # plt.hist(d_phi[...,0,:].reshape(-1).cpu().numpy(), bins=1000)
        # plt.title('$\\Delta_f \\varphi$ (after preprocessing)')
        # plt.show()

        return d_phi

    def derivative(self, x, dim):
        """Approximate the derivative of a tensor using finite differences
        Parameters
        ----------
        x: torch.tensor, shape (*)
            tensor to be derived
        dim: int
            dimension along the tensor should be derived
        Returns
        -------
        torch.tensor, shape (*)
            derivative of `x`
        """
        dx = x.narrow(dim, 1, x.size(dim)-1) - x.narrow(dim, 0, x.size(dim)-1)
        return torch.cat((dx, dx.narrow(dim, -1, 1)), dim=dim)


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([3, 1, 2560])
    '''
    length = 320//2+1
    init_bias = (torch.zeros(length), torch.ones(length))
    model = DNN(init_bias)

    x = torch.randn(3, 1, 16000)  # (B, channels, T).
    print(x.shape)

    out = model(x)
    # model params
    flops, params = get_model_complexity_info(model, (1, 16000), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))