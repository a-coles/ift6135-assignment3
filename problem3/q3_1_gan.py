'''
This file trains a GAN for q3.1.
'''
import sys
sys.path.append('../..')

import torch

from assignment.problem3.gan import GAN
from assignment.problem3.dataloaders import get_loaders


def wgan_gp(Dx, DGy, grad=None, lamb=10, objective='max'):
    '''
    Maximize the WGAN objective with gradient penalty
    https://arxiv.org/pdf/1704.00028.pdf
    '''
    if objective == 'max':
        # We are coming from the discriminator.
        grad = grad[0]  # Take first item in mysterious tuple
        grad_penalty = lamb * (torch.norm(grad, 2) - 1).pow(2).mean()
        print('norm:', torch.norm(grad, 2))
        #print('grad penalty:', grad_penalty)
        #print('DGy:', DGy.mean())
        #print('Dx:', Dx.mean())
        loss = DGy.mean() - Dx.mean() + grad_penalty
        loss = -1 * loss
    elif objective == 'min':
        # We are coming from the generator -- the other terms in the loss don't matter.
        loss = DGy.mean()
    return loss


if __name__ == '__main__':
    # Get the dataloaders for SVHN
    print('Getting dataloaders...')
    train_loader, valid_loader, test_loader = get_loaders(batch_size=8)

    # Instantiate and train GAN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = GAN(device=device)
    print('Training GAN...')
    gan.train(train_loader, valid_loader, loss_fn=wgan_gp)
    gan.log_learning_curves()
    gan.log_d_crossentropy()
    gan.save_model('gan.pt')
