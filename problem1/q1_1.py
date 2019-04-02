'''
This file trains an MLP to learn/estimate the
Jensen-Shannon Divergence (JSD), as in q1.1. We will
save the weights and reload them for q1.3.
'''

import sys
sys.path.append("../..")

import math
import torch
import json

from assignment.problem1.mlp import MLP
from assignment.samplers import distribution2, distribution3


def jsd_loss(x, y, grad=None):
    '''
    Objective function for the Jensen-Shannon Divergence
    '''
    loss = math.log(2) + (0.5 * torch.log(x).mean()) + (0.5 * torch.log(1 - y).mean())
    # We want to maximize this objective, not minimize it, so we negate it
    loss = -1 * loss
    return loss


if __name__ == '__main__':
    # Set up data - batch size is fixed at 512
    # Question: which distributions?
    p = iter(distribution2(512))
    q = iter(distribution3(512))

    # Set up MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('jsd_config.json', 'r') as fp:
        config = json.load(fp)
    mlp = MLP(config, device=device)

    # Train and save
    mlp.train(p, q, loss_fn=jsd_loss)
    mlp.save_model('jsd.pt')
