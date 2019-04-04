'''
This file trains a GAN for q3.1.
'''
import sys
sys.path.append('../..')

import torch

from assignment.problem3.gan import GAN
from assignment.problem3.dataloaders import get_loaders


def wgan_gp(Dx, DGy, grad=None, lamb=10):
    '''
    Maximize the WGAN objective with gradient penalty
    https://arxiv.org/pdf/1704.00028.pdf
    '''
    grad = grad[0]  # Take first item in mysterious tuple
    grad_penalty = lamb * torch.norm((grad - 1), 2).pow(2).mean()
    loss = DGy.mean() - Dx.mean() + grad_penalty
    loss = -1 * loss
    pass


if __name__ == '__main__':
    # Get the dataloaders for SVHN
    train_loader, valid_loader, test_loader = get_loaders(batch_size=128)

    # Instantiate and train GAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = GAN(device=device)
    gan.train(train_loader, valid_loader, loss_fn=wgan_gp)
    gan.log_learning_curves()
    gan.log_d_crossentropy()
    gan.save_model('gan.pt')
