'''
This file contains evaluation methods for the VAE.
'''

import torch
import torch.nn as nn
import torch.distributions as dists
from scipy.stats import norm
import numpy as np


def elbo_loss(x, x_hat, batch_size, mu, logvar):
    '''
    The ELBO loss is the reconstruction loss + the KL divergence,
    summed over the minibatch.
    '''
    recon_loss = nn.BCELoss(reduction='sum')
    BCE = recon_loss(x_hat.float(), x.float())
    # Note: have we made a supposition that p and q are underlyingly
    # gaussian? If so, we can solve the KL divergence like this
    # (see appendix B here for derivation: https://arxiv.org/abs/1312.6114):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # remove batch size and leave x.shape[0]
    loss = (BCE + kld) / x.shape[0]
    return loss


def log_likelihood_estimate(model, loader, device, batch_size):
    '''
    Estimates the log likelihood by importance sampling.
    '''
    k = 50
    k_loss = []
    model.eval()
    # per batch

    for i, x in enumerate(loader):
        # per xi
        for j in range(len(x[0])):
            xi = x[0][j]
            # duplicate k times to avoid loop
            xi_l = []
            xi_l.extend([xi] * k)
            xi_m = torch.stack(xi_l)
            xi_m = xi_m.to(device)
            # calc input for loss
            output, mu, logvar = model(xi_m)
            stdev = torch.exp(0.5 * logvar)
            #calco input for q
            mu_ = mu[0,:]
            stdev_ = stdev[0,:]
            # generate and sample q
            q_gauss = dists.Normal(mu_, stdev_)
            q_samp = q_gauss.sample_n(k)

            # generate and sample p(zik)
            mu_0 = torch.zeros(100)
            sig_1 = torch.ones(100)
            p_gauss = dists.Normal(mu_0, sig_1)
            p_samp = p_gauss.sample_n(k)
            # send to device
            q = norm.pdf(q_samp.cpu())
            p = norm.pdf(p_samp.cpu())
            p_t = torch.from_numpy(p)
            q_t = torch.from_numpy(q)
            q_t = q_t.to(device)
            p_t = p_t.to(device)
            # take sample
            DKL = DKL_ksample(q_t, p_t, k)
            # calc loss
            recon_loss = nn.BCELoss(reduction='sum')
            lo = recon_loss(output.float(), xi_m.float())
            loss = lo + DKL
            # loss = loss_fn(xi_m, output, batch_size=1, mu=mu, logvar=logvar)
            # print('log sum exp loss is',loss)
            k_loss.append(logSumExp(loss/k))
    return k_loss


def DKL_ksample(q, p, k):
    # find a fast way to sum this
    dkl = -1 * (q * torch.log(q) - q * torch.log(p))
    # check axis
    dkl = torch.mean(dkl)
    return dkl

def logSumExp(input):
    inp = input.cpu().detach().numpy()
    max = np.max(inp)
    n = inp - max
    exp = np.exp(n).sum()
    return max + np.log(exp)
