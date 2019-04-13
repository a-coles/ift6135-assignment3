'''
This file plots JSD vs WD as in q1.3.
Note that running matplotlib on Helios to actually plot will
lead to stack smashing errors.
'''
import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt


def plot(theta, jsds, wds):
    plt.plot(theta, jsds, label='JSD')
    #plt.plot(theta, wds, label='WD')
    plt.title('Jensen-Shannon Divergence vs Wasserstein Distance')
    plt.xlabel('phi')
    plt.ylabel('Distance between distributions')
    plt.legend()
    plt.savefig('q1_3.png')
    plt.clf()


if __name__ == '__main__':
    # Open the numpy array
    data = np.load('q1_3.npy')
    theta, jsds, wds = data[0], data[1], data[2]
    print('theta:', theta)
    print('jsds:', jsds)
    print('wds:', len(wds))

    # Plot
    plot(theta, jsds, wds)
