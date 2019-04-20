'''
This file contains evaluation methods for the VAE.
'''

import torch
import torch.nn as nn
import torch.distributions as dists
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


def log_likelihood_estimate(model, loader, device, batch_size, loss_fn):
    '''
    Estimates the log likelihood by importance sampling.
    '''
    k = 5
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
            output, mu, logvar = model(xi_m)

            # generate and sample q
            stdev = torch.exp(0.5 * logvar)
            q_gauss = torch.dists.normal(mu, stdev)
            q_samp = q_gauss.sample(xi_m)

            # generate and sample p(zik)
            p_gauss = torch.dists.normal(0, 1)
            p_samp = p_gauss.sample(xi_m)

            DKL = DKL_sample(q_samp, p_samp, k)
            loss =


            # loss = loss_fn(xi_m, output, batch_size=1, mu=mu, logvar=logvar)
            # print('log sum exp loss is',loss)
            k_loss.append(logSumExp(loss))
    return k_loss


def DKL_sample(q, p, k):
    # find a fast way to sum this
    dkl = -1 * (q * torch.log(q) - q * torch.log(p))
    # check axis
    dkl = torch.mean(dkl)


def logSumExp(input):
    inp = input.cpu().detach().numpy()
    max = np.max(inp)
    d = inp - max
    sumOfExp = np.exp(d).sum()
    return max + np.log(sumOfExp)
