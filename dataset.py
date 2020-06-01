import os
import pathlib

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class AudioDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.fs = os.listdir(self.datapath)
        self.mu_bounds = self.mu_law(2**15)    # For 16-bit audio

    def mu_law(self, x, mu=255.):
        return np.sign(x) * np.log(1. + mu*np.abs(x))/np.log(1. + mu)

    def discretize(self, x, lower_bound, upper_bound, bins=256):
        discrete = ((x - lower_bound) / (upper_bound - lower_bound) * bins).astype(int)
        return np.clip(discrete, 0, int(bins)-1)

    def label_to_onehot(self, labels, num_classes=256):
        onehot = np.zeros((len(labels), num_classes))
        onehot[np.arange(len(labels)), labels] = 1
        return onehot

    def __getitem__(self, idx):
        data = np.load(pathlib.Path.cwd() / self.datapath / self.fs[idx])
        '''
        target = self.label_to_onehot(
                            self.discretize(
                                self.mu_law(data),
                                -self.mu_bounds,
                                self.mu_bounds)
                            )
        '''
        target = self.discretize(
                        self.mu_law(data),
                        -self.mu_bounds,
                        self.mu_bounds)

        # Normalize data
        data = np.expand_dims(data / 128. - 1., 0)

        return data, target

    def __len__(self):
        return len(self.fs)

#ds = AudioDataset('./data/denero/npy')
