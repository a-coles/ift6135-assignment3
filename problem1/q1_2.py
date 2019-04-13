'''
This file trains an MLP to learn/estimate the
Wasserstein Distance (WD), as in q1.2. We will
save the weights and reload them for q1.3.
'''

import sys
sys.path.append("../..")

import torch
import json

from assignment.problem1.mlp import MLP
from assignment.samplers import distribution2, distribution3


def wd_loss(Dx, Dy, grad=None, lamb=10):
    '''
    Objective function for the Wasserstein Distance.
    '''
    grad = grad[0]  # Take first item in mysterious tuple
    grad_penalty = lamb * (torch.norm(grad, 2) - 1).pow(2).mean()
    loss = Dx.mean() - Dy.mean() - grad_penalty
    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss


if __name__ == '__main__':
    # Set up data - batch size is fixed at 512
    # Question: which distributions?
    p = iter(distribution2(512))
    q = iter(distribution2(512))

    # Set up MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('wd_config.json', 'r') as fp:
        config = json.load(fp)
    mlp = MLP(config, device=device)

    # Train and save
    mlp.train(p, q, loss_fn=wd_loss, dist_type='wd')
    mlp.save_model('wd.pt')
