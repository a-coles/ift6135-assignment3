'''
This file contains evaluation methods for the VAE.
'''

import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np


def elbo_loss(x, x_hat, mu, logvar):
    '''
    The ELBO loss is the reconstruction loss + the KL divergence,
    summed over the minibatch.
    '''
    recon_loss = nn.BCELoss()

    # Note: have we made a supposition that p and q are underlyingly
    # gaussian? If so, we can solve the KL divergence like this
    # (see appendix B here for derivation: https://arxiv.org/abs/1312.6114):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss(x_hat.float(), x.float()) + kld
    return loss


def log_likelihood_estimate(model, loader):
    '''
    Estimates the log likelihood by importance sampling.
    '''
    k = 200
    for i, x in enumerate(loader):
        # Sample k times from encoder distribution
        ziks = []
        for xi in x:
            mu, logvar = model.encoder(xi)
            stdev = torch.exp(0.5 * logvar)
            encoder_dist = dists.Normal(mu, stdev)
            zik = encoder_dist.sample(k)
            ziks.append(zik)
        ziks = torch.from_numpy(np.array(ziks))
        estimates = importance_sample(model, x, ziks)
    estimate = estimates.sum()
    return estimate


def importance_sample(model, xis, ziks):
    '''
    Implements importance sampling for q2.2.
    We are asked to provide this function in the report,
    so it must have these specific arguments and return value.
    Basically, this does importance sampling for one minibatch.
    '''
    estimates = []

    # We basically have q(zik | xi) as input, as ziks. 
    # Next, get the p(xi | zik) term:
    likelihoods = []
    for zik in ziks:
        likelihood = model.decoder(zik)
        likelihoods.append(likelihood)
    likelihoods = torch.from_numpy(np.array(likelihoods))

    # Then get the p(zik) term:
    prior_dist = dists.Normal(torch.zeros(784), torch.ones(784)) # ?????


    return estimates
