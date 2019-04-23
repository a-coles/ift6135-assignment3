'''
Dataloader classes for Binarized MNIST.
'''

import os
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def get_binarized_mnist_loaders(batch_size=512):
    # Load in the data files
    with open(os.path.join('binarized_mnist', 'binarized_mnist_train.amat'), 'r') as fp:
        x_train = fp.readlines()
        x_train = np.array([[int(i) for i in line.split()] for line in x_train])

    with open(os.path.join('binarized_mnist', 'binarized_mnist_valid.amat'), 'r') as fp:
        x_valid = fp.readlines()
        x_valid = np.array([[int(i) for i in line.split()] for line in x_valid])

    with open(os.path.join('binarized_mnist', 'binarized_mnist_test.amat'), 'r') as fp:
        x_test = fp.readlines()
        x_test = np.array([[int(i) for i in line.split()] for line in x_test])

    # Get the dataloaders
    train = TensorDataset(torch.from_numpy(x_train))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    valid = TensorDataset(torch.from_numpy(x_valid))
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    test = TensorDataset(torch.from_numpy(x_test))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
