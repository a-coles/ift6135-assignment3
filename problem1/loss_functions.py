'''
Contains loss functions/discriminator objectives.
'''

import torch
import math


def jsd_loss(Dx, Dy, grad=None):
    '''
    Objective function for the Jensen-Shannon Divergence.
    '''
    Ex = 0.5 * torch.log(Dx).mean()
    Ey = 0.5 * torch.log(1 - Dy).mean()
    loss = math.log(2) + Ex + Ey
    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss


def wd_loss(Dx, Dy, grad=None, lamb=10):
    '''
    Objective function for the Wasserstein Distance.
    '''
    grad = grad[0]  # Take first item in mysterious tuple
    grad_penalty = lamb * (torch.norm(grad, 2) - 1).pow(2).mean()
    print('grad penalty:', grad_penalty)
    loss = Dx.mean() - Dy.mean() - grad_penalty
    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss
