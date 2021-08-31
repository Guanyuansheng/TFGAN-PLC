import torch
import torch.nn as nn
import torch.nn.functional as F
from util import RGBBlock, Conv2DMod, DownBlock, UpBlock
import numpy as np
from functools import partial


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2)
        self.to_rgb = RGBBlock(latent_dim, input_channels, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        noise1 = F.linear(inoise, torch.Tensor(x.shape[1] * x.shape[2], 100)).reshape(x.shape)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        noise2 = F.linear(inoise, torch.Tensor(x.shape[1] * x.shape[2], 100)).reshape(x.shape)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, rgba=False, no_const=False, fmap_max=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(np.log2(image_size) - 1)

        filters = [network_capacity * (2 ** i) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]

        in_out_pairs = list(zip(filters[:-1], filters[1:]))
        in_out_pairs = list(map(list, in_out_pairs))
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose1d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4)))

        self.initial_conv = nn.Conv1d(filters[0], filters[0], 3, padding=1)

        self.blocks = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            num_layer = self.num_layers - ind

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=True,
                upsample_rgb=True,
                rgba=rgba
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1)

        x = self.initial_conv(x)
        # styles = styles.transpose(0, 1)

        rgb = None
        for style, block in zip(styles, self.blocks):
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 512])
    output: torch.Size([3, 1, 512])
    '''
    model = Generator(image_size=512, latent_dim=512, network_capacity=16, no_const=False, fmap_max=512)

    x = torch.randn(3, 1, 512)  # (B, channels, T).
    z = nn.init.uniform_(torch.Tensor(3, 100), -1., 1.)
    print(x.shape)

    generated_images = model(x, z)
    print(generated_images.shape)
    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))