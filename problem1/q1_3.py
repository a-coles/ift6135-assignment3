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

from assignment.samplers import distribution1
from assignment.problem1.mlp import MLP


if __name__ == '__main__':
    # Let p be the distribution of (0, Z)
    p = iter(distribution1(0, 512))

    # Let q_theta be the distribution of (theta, Z)
    theta = np.arange(-1, 1, 0.1)
    q_theta = {}
    for val in theta:
        q_theta[val] = iter(distribution1(val, 512))

    # Load JSD and WD models
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
        wds.append(wd_xy)

    # Save a numpy array for JSD and WD over all theta
    print('theta:', theta)
    print('jsds:', len(jsds))
    print('wds:', len(wds))
    data = np.array([theta, jsds, wds])
    np.save('q1_3.npy', data)
