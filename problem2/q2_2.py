'''
This file evaluates a trained VAE using importance
sampling for q2.2.
'''

import sys
sys.path.append('../..')

import torch

from dataloaders import get_binarized_mnist_loaders
from vae import VAE
from eval_methods import elbo_loss, log_likelihood_estimate


def log_eval(eval_method, dataset, num):
    filename = 'vae_{}_{}.log'
    with open(filename, 'w') as fp:
        fp.write('{} evaluation on {} set: {}'.format(eval_method, dataset, num))


if __name__ == '__main__':
    # Set up dataloaders
    batch_size = 128
    _, valid_loader, test_loader = get_binarized_mnist_loaders(batch_size)

    # Load trained VAE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device ='cpu'
    vae = VAE(batch_size, device=device, model_path='vae.pt')
    # Evaluate valid and test sets on ELBO loss

    elbo_valid = vae.valid_epoch(valid_loader, loss_fn=elbo_loss)
    log_eval('elbo', 'valid', elbo_valid)
    print('elbo_valid is' ,elbo_valid)

    elbo_test = vae.valid_epoch(test_loader, loss_fn=elbo_loss)
    log_eval('elbo', 'test', elbo_test)
    print('elbo_test is', elbo_test)

    # Evaluate valid and test sets on log-likelihood
    loglike_valid = vae.eval_log_likelihood(valid_loader, device, batch_size, loss_fn=elbo_loss)
    print('loglike value imp sampling is ',sum(loglike_valid)/len(loglike_valid))
    log_eval('loglikelihood', 'valid', loglike_valid)


    loglike_test = vae.eval_log_likelihood(test_loader, device, batch_size, loss_fn=elbo_loss)
    print('loglike value imp sampling is ',sum(loglike_test)/len(loglike_test))
    log_eval('loglikelihood', 'test', loglike_test)
