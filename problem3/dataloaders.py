'''
This file contains dataloading classes for SVHN.
'''

import torch
import numpy as np
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_loaders(batch_size=100, valid_split=0.2, transform=None):
    # Download SVHN yourself -- this will avoid internet issues when running on Helios.

    # We need to create a train and a valid dataloader from the training set, so first
    # create datasets for each from the training set
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.SVHN(root='SVHN', split='train', download=False, transform=transform)
    valid = datasets.SVHN(root='SVHN', split='train', download=False, transform=transform)

    # Need to sample indices randomly from the training set
    len_train = len(train)
    indices = list(range(len_train))
    split = int(np.floor(valid_split * len_train))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Now use random indices to assign samples for each dataloader
    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=valid,
                              batch_size=batch_size,
                              sampler=valid_sampler)

    # Straightforwardly create test dataloader
    test = datasets.SVHN(root='SVHN', split='test', download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, valid_loader, test_loader
