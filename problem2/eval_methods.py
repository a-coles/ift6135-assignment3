'''
This file contains evaluation methods for the VAE.
'''

import torch
import torch.nn as nn


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

    loss = recon_loss(x.float(), x_hat.float()) + kld
    return loss


def log_likelihood_estimate(model, xis, ziks):
    '''
    Estimates the log likelihood by importance sampling.
    '''
    estimates = importance_sample(model, xis, ziks)
    estimate = estimates.sum()
    return estimate


def importance_sample(model, xis, ziks):
    '''
    Implements importance sampling for q2.2.
    We are asked to provide this function in the report,
    so it must have these specific arguments and return value.
    '''
    return estimates
