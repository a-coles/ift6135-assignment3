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
    #grad = grad[0]  # Take first item in mysterious tuple
    #print('norm:', torch.norm(grad))
    #print('norm-1 :', (torch.norm(grad, dim=1) - 1)[0:10])
    #print('norm-1 pow:', (torch.norm(grad, dim=1) - 1).pow(2)[0:10])

    # print('grad:', grad.size())
    # print('grad norm:', torch.norm(grad).size())
    # print('grad norm 0:', torch.norm(grad, dim=0).size())
    # print('grad norm 0:', torch.norm(grad, dim=1).size())
    # grad_penalty = lamb * torch.pow((torch.norm(grad, dim=1) - 1), 2).mean()
    grad_penalty = lamb * (torch.norm(grad, dim=1) - 1).pow(2).mean()
    # grad_penalty = lamb * math.pow((grad - 1), 2)

    # print('grad_penalty:', grad_penalty)
    

    loss = Dx.mean() - Dy.mean() - grad_penalty
    #loss = Dy.mean() - Dx.mean() - grad_penalty
    #loss = Dy.mean() - Dx.mean() + grad_penalty
    # loss = Dx.mean() - Dy.mean() + grad_penalty



    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss
