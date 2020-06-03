import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import pdb

ks=2

class DilatedConv(nn.Conv1d):
    '''
    Apply a 1d temporal convolution, shifted to preserve the autoregressive property
    Always returns data with the same length
    '''

    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B', 'Mask type must be A or B'
        super().__init__(*args, **kwargs)

        # Calculate kernel size, with dilations
        effective_kernel_size = self.kernel_size[0] + \
                                (self.dilation[0] - 1) * (self.kernel_size[0] - 1)

        if mask_type == 'A':
            self.padding = (effective_kernel_size, )
        else:
            self.padding = (effective_kernel_size - 1, )

    def forward(self, x):
        length = x.shape[2]
        x = super().forward(x)
        return x[:,:,:length]

class ResidualBlock(nn.Module):
    '''
    Simple residual version of DilatedConv
    '''
    def __init__(self, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv = DilatedConv(*args, **kwargs)

    def forward(self, x):
        out = self.conv(x)
        return x + out

class ResidualSkipBlock(nn.Module):
    '''
    Residual DilatedConv block with skip connections and 
    Gated Activation Units as non-linearities
    Outputs both the transformed data and the skip output
    '''

    def __init__(self, *args, **kwargs):
        super(ResidualSkipBlock, self).__init__()
        num_filters = args[1]

        self.conv = DilatedConv(*args, **kwargs)
        self.gate = GatedActivationUnit(num_filters, ks)

        # 1x1 applied to the skip connection
        self.skip_conv = DilatedConv('B', num_filters, num_filters, kernel_size=1)

    def forward(self, x):
        out = self.gate(self.conv(x))
        skip = self.skip_conv(out)
        return x + skip, skip    # in paper it's `x + skip, skip`

class GatedActivationUnit(nn.Module):
    '''
    A Gated Activation Unit
    '''
    def __init__(self, num_filters, kernel_size):
        super(GatedActivationUnit, self).__init__()
        self.f = DilatedConv('B', num_filters, num_filters, 
                             kernel_size=kernel_size, bias=False)
        self.g = DilatedConv('B', num_filters, num_filters, 
                             kernel_size=kernel_size, bias=False)

    def forward(self, x):
        filt = torch.tanh(self.f(x))
        gate = torch.sigmoid(self.g(x))
        return filt * gate

class WaveNet(nn.Module):
    def __init__(self, out_channels=256):
        super(WaveNet, self).__init__()
        num_filters = 64

        # Building residual blocks
        self.layers = nn.ModuleList()
        self.layers.append(DilatedConv('A', 1, num_filters, 
                                       kernel_size=ks, dilation=1))
        for i in range(1,2):
            self.layers.append(ResidualSkipBlock('B', num_filters, num_filters, 
                                             kernel_size=ks, dilation=2**(i%10)))
            #self.layers.append(nn.ReLU())

        # Final layer on sum of skip connections
        self.final_layers = nn.ModuleList()
        self.final_layers.append(nn.ReLU())
        self.final_layers.append(DilatedConv('B', num_filters, num_filters, 
                                       kernel_size=1, dilation=1))
        self.final_layers.append(nn.ReLU())
        self.final_layers.append(DilatedConv('B', num_filters, out_channels, 
                                       kernel_size=1, dilation=1))


    def forward(self, x):
        cum_skip = 0
        for layer in self.layers:
            if isinstance(layer, ResidualSkipBlock):
                x, skip = layer(x)
                cum_skip += skip
            else:
                x = layer(x)

        for layer in self.final_layers:
            cum_skip = layer(cum_skip)

        return cum_skip

    def sample(self, n, t=16000, device='cpu'):
        self.eval()

        samples = torch.zeros(n, 1, t).to(device)
        for i in tqdm(range(t)):
            pred = self.forward(samples)
            pred = F.softmax(pred, dim=1)
            samples[:,:,i] = torch.multinomial(pred[:,:,i], 1)

        return self.invert_samples(samples)

    def inverse_mu_law(self, x, mu=255.):
        return np.sign(x) * (np.power(1.+mu, np.abs(x)) - 1.) / mu

    def undiscretize(self, x, bins=256):
        return 2. * x / bins - 1.

    def invert_samples(self, x):
        x = self.undiscretize(x).cpu().detach().numpy()
        x = self.inverse_mu_law(x)
        x = x * (2**15)
        x = x.astype(int)
        x = np.clip(x, -(2**15), (2**15)-1)
        return x
