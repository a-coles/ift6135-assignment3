'''
This file evaluates a trained VAE using importance
sampling for q2.2.
'''

import sys
sys.path.append('../..')

import torch

from assignment.probelm2.dataloaders import get_binarized_mnist_loaders
from assignment.problem2.vae import VAE
from assignment.problem2.eval_methods import elbo_loss, log_likelihood_estimate


def log_eval(eval_method, dataset, num):
    filename = 'vae_{}_{}.log'
    with open(filename, 'w') as fp:
        fp.write('{} evaluation on {} set: {}'.format(eval_method, dataset, num))


if __name__ == '__main__':
    # Set up dataloaders
    _, valid_loader, test_loader = get_binarized_mnist_loaders(batch_size=128)

    # Load trained VAE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(device=device, model_path='vae.pt')

    # Evaluate valid and test sets on ELBO loss
    elbo_valid = vae.valid_epoch(valid_loader, loss_fn=elbo_loss)
    log_eval('elbo', 'valid', elbo_valid)

    elbo_test = vae.valid_epoch(test_loader, loss_fn=elbo_loss)
    log_eval('elbo', 'test', elbo_test)

    # Evaluate valid and test sets on log-likelihood
    loglike_valid = vae.eval_log_likelihood(valid_loader)
    log_eval('loglikelihood', 'valid', loglike_valid)

    loglike_test = vae.eval_log_likelihood(test_loader)
    log_eval('loglikelihood', 'test', loglike_test)
