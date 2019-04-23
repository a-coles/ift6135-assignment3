#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function
import sys
sys.path.append("../..")

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from assignment.samplers import distribution4, gaussian_1d, gaussian_1d_density
from assignment.problem1.mlp import MLP

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 


# Load the trained "unk" model for q1.4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('unk_config.json', 'r') as fp:
    unk_config = json.load(fp)
unk = MLP(unk_config, device=device)
unk.load_model('unk.pt')

# Pass samples through the model and estimate
# the unknown density using the identity from q5 theoretical:
# 	f1 = f0 D*(x) / (1 - D*(x))
# 	where D*(x) = argmax_D E(log(Dx)) + E(log(1-Dx))
p0 = iter(gaussian_1d(512))
p1 = iter(distribution4(512))
x = next(p1)	# Samples from unknown distribution
y = next(p0)	# Samples from known Gaussian distribution
f0_x = gaussian_1d_density(x)
f1_x = unk.estimate_unk(x, f0_x)


############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


xx = xx.reshape(1000, 1)
xx = torch.from_numpy(xx).float().to(device)
r = unk.model(xx) # evaluate xx using your discriminator; replace xx with the output
xx = xx.cpu().numpy()
r = r.cpu().detach().numpy()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

estimate = unk.estimate_unk(xx, gaussian_1d_density(xx)).cpu().detach().numpy()
#estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')

plt.savefig('q1_4_estimated_density.png')











