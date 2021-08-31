import torch
import torch.nn as nn
import torch.nn.functional as F
from util import RGBBlock, Conv2DMod, DownBlock, UpBlock, Flatten
import numpy as np
from functools import partial


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, fmap_max=512):
        super().__init__()
        num_layers = int(np.log2(image_size) - 3)
        num_init_filters = 1

        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample=is_not_last)
            down_blocks.append(block)

        self.down_blocks = nn.ModuleList(down_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.AvgPool1d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = nn.Sequential(nn.Conv1d(last_chan, last_chan, 3, padding=1), nn.LeakyReLU(0.2),
                                  nn.Conv1d(last_chan, last_chan, 3, padding=1), nn.LeakyReLU(0.2))

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Sequential(nn.Conv1d(1, 1, 1),
                                      nn.Tanh())

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for i, down_block in enumerate(self.down_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 512])
    output: torch.Size([3, 1, 512])
    '''
    model = Discriminator(image_size=512, network_capacity=16, fmap_max=512)

    x = torch.randn(3, 1, 512)  # (B, channels, T).
    print(x.shape)

    enc_out, dec_out = model(x)
    print(enc_out.shape, dec_out.shape)
    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))