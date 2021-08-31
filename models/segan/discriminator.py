import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super(Discriminator, self).__init__()
        self.dropout1 = nn.Dropout(dropout_drop)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=5),
                                    nn.LeakyReLU(0.2))
        self.conv1_e = nn.Sequential(nn.Conv1d(16, 16, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 32, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(32, 64, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64, 128, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(128, 128, 11, 2, 5),
                                    nn.LeakyReLU(0.2))
        self.fully_connected = nn.Linear(in_features=2560, out_features=1)

    def forward(self, x):
        """
        Forward pass of discriminator.
        Args:
            x: input batch (signal), size = [Bx1x2560]
        """
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

        return x


if __name__ == '__main__':
    model = Discriminator()
    '''
    input: torch.Size([3, 1, 2560])
    output: torch.Size([])
    '''

    x = torch.randn(3, 1, 2560)
    print(x.shape)

    out = model(x)
    print(out.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {} ({:.2f} MB)'.format(pytorch_total_params, (float(pytorch_total_params) * 4) / (1024 * 1024)))

