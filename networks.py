import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B', 'Mask type must be A or B'
        super().__init__(*args, **kwargs)

        effective_kernel_size = self.kernel_size[0] + \
                                (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
        if mask_type == 'A':
            self.padding = effective_kernel_size
        else:
            self.padding = effective_kernel_size - 1

    def forward(self, x):
        length = x.shape[2]

        x = super().forward(x)
        return x[:,:,:length]


class WaveNet(nn.Module):
    def __init__(self, out_channels=256):
        super(WaveNet, self).__init__()
        num_filters = 64
        self.convs = nn.ModuleList()

        self.convs.append(DilatedConv('A', 1, num_filters, kernel_size=7, dilation=1))

        for i in range(1,10):
            self.convs.append(DilatedConv('B', num_filters, num_filters, kernel_size=7, dilation=2**(i%10)))
            self.convs.append(nn.ReLU())

        self.convs.append(DilatedConv('B', num_filters, out_channels, kernel_size=7, dilation=1))


    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x
