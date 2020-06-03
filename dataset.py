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

    def mu_law(self, x, mu=255.):
        return np.sign(x) * np.log(1. + mu*np.abs(x))/np.log(1. + mu)

    def discretize(self, x, bins=256):
        # Hardcoded for mu_law
        lower_bound = -1
        upper_bound = 1

        discrete = ((x - lower_bound) / (upper_bound - lower_bound) * bins).astype(int)
        return np.clip(discrete, 0, int(bins)-1)

    def label_to_onehot(self, labels, num_classes=256):
        onehot = np.zeros((len(labels), num_classes))
        onehot[np.arange(len(labels)), labels] = 1
        return onehot

    def __getitem__(self, idx):
        data = np.load(pathlib.Path.cwd() / self.datapath / self.fs[idx])
        data = data / float(2**15)  # 16 bit audio

        target = self.discretize(self.mu_law(data))

        # Add channel dim to data
        data = np.expand_dims(data, 0)

        return data, target

    def __len__(self):
        return len(self.fs)

class DummyDataset(Dataset):
    def __init__(self):
        pass

    def discretize(self, x, bins=256):
        # Hardcoded for mu_law
        lower_bound = -1
        upper_bound = 1

        discrete = ((x - lower_bound) / (upper_bound - lower_bound) * bins).astype(int)
        return np.clip(discrete, 0, int(bins)-1)

    def __getitem__(self, idx):
        out = np.arange(0, 4096) + np.random.randint(256)
        out = out % 256
        out = (out - 128) / 128.
        return np.expand_dims(out, 0), self.discretize(out)

    def __len__(self):
        return 10000

#ds = AudioDataset('./data/denero/npy')
