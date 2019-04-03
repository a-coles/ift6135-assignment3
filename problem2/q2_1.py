'''
This file instantiates and trains a VAE
for q2.1.
'''

import sys
sys.path.append('../..')

import torch

from assignment.probelm2.dataloaders import get_binarized_mnist_loaders
from assignment.problem2.vae import VAE
from assignment.problem2.eval_methods import elbo_loss


if __name__ == '__main__':
    # Set up dataloaders
    train_loader, valid_loader, _ = get_binarized_mnist_loaders(batch_size=128)

    # Instantiate and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(device=device)
    vae.train(train_loader, valid_loader, loss_fn=elbo_loss)
    vae.log()
    vae.save_model('vae.pt')
