import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=0.1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, latent_dim, style_depth, lr_mul=0.1):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear = EqualLinear(in_dim=latent_dim, out_dim=latent_dim, lr_mul=lr_mul)
        self.depth = style_depth

    def forward(self, x):
        x = F.normalize(x, dim=1)
        for i in range(self.depth):
            x = self.linear(x)
            x = self.leaky_relu(x)
        return x


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, filters, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, filters)

        self.conv = Conv2DMod(filters, filters, 1, demod=False)
        self.conv1 = nn.Conv1d(input_channel, filters, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, t = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            prev_rgb = self.conv1(prev_rgb)
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, t = x.shape

        w1 = y[:, None, :, None]
        w2 = torch.cat([self.weight[None, :, :, :] for _ in range(0, b)], 0)
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3), keepdim=True) + 1e-8)
            weights = weights * d

        x = x.reshape(1, -1, t)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(t, self.kernel, self.dilation, self.stride)
        x = F.conv1d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, t)
        return x


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv1d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(nn.Conv1d(input_channels, filters, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv1d(filters, filters, 3, padding=1), nn.LeakyReLU(0.2))
        self.down = nn.Conv1d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose1d(input_channels // 2, filters, 1, stride=2, output_padding=1)
        self.net = nn.Sequential(nn.Conv1d(input_channels, filters, 3, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv1d(filters, filters, 3, padding=1), nn.LeakyReLU(0.2))
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x)
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x


class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 512])
    output: torch.Size([3, 1, 512])
    '''
    model = StyleVectorizer(latent_dim=512, style_depth=8, lr_mul=0.1)

    z = torch.randn(3, 1, 512)  # (B, channels, T).
    print(z.shape)

    out = model(z)

    print(out.shape)
    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))