'''
This file plots JSD vs WD as in q1.3.
Note that running matplotlib on Helios to actually plot will
lead to stack smashing errors.
'''
import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt


def plot(phi, jsds, wds):
    plt.plot(phi, jsds, label='JSD')
    plt.plot(phi, wds, label='WD')
    plt.yscale('log') 
    plt.title('Jensen-Shannon Divergence vs Wasserstein Distance')
    plt.xlabel('phi')
    plt.ylabel('Distance between distributions')
    plt.legend()
    plt.savefig('q1_3.png')
    plt.clf()


if __name__ == '__main__':
    # Open the numpy array
    data = np.load('q1_3.npy')
    phi, jsds, wds = data[0], data[1], data[2]
    print('phi:', phi)
    print('jsds:', jsds)
    print('wds:', len(wds))

    # Plot
    plot(phi, jsds, wds)
