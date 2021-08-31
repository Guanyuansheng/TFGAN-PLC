import torch
import torch.nn as nn
from util import EqualLinear
import numpy as np

class MappingBlock(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            lr_mul,
    ):
        """
        :param input_dim: Linear layer input dim
        :param output_dim: Linear layer output dim
        :param lr_mul: Learning rate multiplier
        :param activation: Activation function,
        """
        super(MappingBlock, self).__init__()
        self.linear = EqualLinear(
            in_dim=input_dim,
            out_dim=output_dim,
            lr_mul=lr_mul
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        out = x
        if y is not None:
            out = torch.cat((out, y), dim=1)
        return self.activation(self.linear(out))


class MappingNet(nn.Module):

    def __init__(
            self,
            input_dim=512,
            output_dim=512,
            depth=8,
            lr_mul=0.01,
            broadcast_output=None,
    ):
        """
        Mapping network used in the StyleGAN paper
        :param input_dim:  Latent vector (Z) dimensionality.
        :param output_dim: Disentangled latent (W) dimensionality.
        :param depth: Number of hidden layers.
        :param lr_mul: Learning rate multiplier for the mapping layers.
        """
        super(MappingNet, self).__init__()
        self.depth = depth
        self.broadcast_output = broadcast_output

        layers = []

        for layer_ix in range(self.depth):
            layers.append(
                MappingBlock(
                    input_dim,
                    output_dim,
                    lr_mul,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, latent, embedding):
        for layer in self.layers:
            latent = layer(latent, embedding)
        if self.broadcast_output:
            dlatent = latent.unsqueeze(1).expand(-1, self.broadcast_output, -1)
        else:
            dlatent = latent
        return dlatent


if __name__ == '__main__':
    '''
    input: torch.Size([3, 1, 512])
    output: torch.Size([3, 1, 512])
    '''
    model = MappingNet()

    x = torch.randn(3, 1, 512)  # (B, channels, T).
    z = nn.init.uniform_(torch.Tensor(x.shape), -1., 1.)
    print(x.shape)

    out = model(x, None)
    assert out.shape == torch.Size([3, 1, 512])
    # model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))