import torch
import torch.nn as nn

class GatedConv(nn.Module):
    def __init__(self, input_size, width=11):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=2 * input_size,
                              kernel_size=width, stride=1, padding=5)

    def forward(self, x_var):
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out

class GLU(nn.Module):
    def __init__(self, input_size, width=11):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                              kernel_size=width, stride=1, padding=5)
        self.gateconv = GatedConv(input_size, width)

    def forward(self, x):
        gate = self.gateconv(x)
        gate = self.conv1(gate)
        x = x + gate
        return x