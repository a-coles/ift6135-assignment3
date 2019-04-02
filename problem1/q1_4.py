'''
This file estimates the density of an unknown distribution
as in q1.4. Here we just do the training and save the model.
The idea is that the density will be plotted locally using
density_estimation.py (TA file) by loading the trained model
and passing a range of values into it.
Note that running matplotlib on Helios leads to stack
smashing errors.
'''

import sys
sys.path.append("../..")

import torch
import json

from assignment.problem1.mlp import MLP
from assignment.samplers import gaussian_1d, distribution4


def unk_loss(x, y, grad=None, lamb=10):
    '''
    Objective function the value function in q1.4.
    '''
    loss = x.log().mean() + y.lov().mean()
    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss


if __name__ == '__main__':
    # Set up data - batch size is fixed at 512
    # Distribution 4 and standard Gaussian (from comment in density_estimation.py)
    p = iter(distribution4(512))
    q = iter(gaussian_1d(512))

    # Set up MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('unk_config.json', 'r') as fp:
        config = json.load(fp)
    mlp = MLP(config, device=device)

    # Train and save
    mlp.train(p, q, loss_fn=unk_loss)
    mlp.save_model('unk.pt')
