'''
This file instantiates and trains a VAE
for q2.1.
'''

import sys
sys.path.append('../..')

import torch

from dataloaders import get_binarized_mnist_loaders
from vae import VAE
from eval_methods import elbo_loss



if __name__ == '__main__':
    # Set up dataloaders
    batch_size = 64
    train_loader, valid_loader, _ = get_binarized_mnist_loaders(batch_size)

    # Instantiate and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(batch_size, device=device)
    vae.train(train_loader, valid_loader, loss_fn=elbo_loss)
    vae.log()
    vae.save_model('vae.pt')
