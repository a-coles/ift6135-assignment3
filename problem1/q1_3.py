'''
This file saves a numpy array JSD vs WD as in q1.3.
Note that running matplotlib on Helios to actually plot will
lead to stack smashing errors.
'''
import sys
sys.path.append("../..")

import numpy as np
import json
import torch
import matplotlib.pyplot as plt

from assignment.samplers import distribution1
from assignment.problem1.mlp import MLP
from assignment.problem1.loss_functions import jsd_loss, wd_loss


def plot(phi, jsds, wds):
    plt.plot(phi, jsds, label='JSD')
    plt.plot(phi, wds, label='WD')
    plt.title('Jensen-Shannon Divergence vs Wasserstein Distance')
    plt.xlabel('phi')
    plt.ylabel('Distance between distributions')
    plt.legend()
    plt.savefig('q1_3.png')
    plt.clf()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Let p be the distribution of (0, Z)
    p = iter(distribution1(0, 512))

    # Let q_theta be the distribution of (phi, Z)
    phi = np.linspace(-1, 1, 21)
    q_theta = {}
    for val in phi:
        q_theta[val] = iter(distribution1(val, 512))

    # Now we need to train discriminators between the fixed p and
    # all 21 of the q's.
    jsds, wds = [], []
    with open('jsd_config.json', 'r') as fp:
        jsd_config = json.load(fp)
    with open('wd_config.json', 'r') as fp:
        wd_config = json.load(fp)

    for ph, q in q_theta.items():
        print('PHI:', ph)

        # Train discriminators
        print('Training JSD...')
        jsd = MLP(jsd_config, device=device)
        jsd.train(p, q, loss_fn=jsd_loss, dist_type='jsd', num_epochs=800)

        print('Training WD...')
        wd = MLP(wd_config, device=device)
        wd.train(p, q, loss_fn=wd_loss, dist_type='wd', num_epochs=400)

        # Sample from p and q
        x = next(p)
        y = next(q)

        # Call JSD and WD predictions
        jsd_xy = jsd.estimate_jsd(x, y).cpu().detach().numpy()
        jsds.append(jsd_xy)
        wd_xy = wd.estimate_wd(x, y).cpu().detach().numpy()
        wds.append(wd_xy)

    # Then the WD:

    '''# Load JSD and WD models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('jsd_config.json', 'r') as fp:
        jsd_config = json.load(fp)
    jsd = MLP(jsd_config, device=device)
    jsd.load_model('jsd.pt')

    with open('wd_config.json', 'r') as fp:
        wd_config = json.load(fp)
    wd = MLP(wd_config, device=device)
    wd.load_model('wd.pt')

    # Calculate JSD and WD distances over all theta
    jsds, wds = [], []
    for _, q in q_theta.items():
        x = next(p)
        y = next(q)
        jsd_xy = jsd.estimate_jsd(x, y).cpu().detach().numpy()
        print('jsd:', jsd_xy)
        jsds.append(jsd_xy)
        wd_xy = wd.estimate_wd(x, y).cpu().detach().numpy()
        wds.append(wd_xy)'''

    # Save a numpy array for JSD and WD over all theta
    jsds = np.array(jsds)
    wds = np.array(wds)
    print('phi:', phi)
    print('jsds:', jsds)
    print('wds:', wds)
    data = np.array([phi, jsds, wds])
    np.save('q1_3.npy', data)

    plot(phi, jsds, wds)
