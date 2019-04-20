'''
This file trains a GAN for q3.1.
'''
import sys
sys.path.append('../..')

import torch

from vae import VAE
from dataloaders import get_loaders
from eval_methods import elbo_loss

if __name__ == '__main__':
    # Get the dataloaders for SVHN
    print('Getting dataloaders...')
    batch_size = 512
    train_loader, valid_loader, test_loader = get_loaders(batch_size)
    # Instantiate and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(batch_size, device=device)
    vae.train(train_loader, valid_loader, loss_fn=elbo_loss)
    vae.log()
    vae.log_learning_curves()
    vae.save_model('vae.pt')

