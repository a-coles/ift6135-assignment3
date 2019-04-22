'''
This file contains the qualitative evaluations
for q3.2. Note: it should be run locally, or matplotlib
will cause stack smashing errors on Helios.
'''
import sys
sys.path.append('../..')

import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from gan import GAN
from vae import VAE
from torch.autograd import Variable


def plot_sample(sample, filename):
    # sample = sample.reshape(3, 32, 32)
    # sample = np.moveaxis(sample, 0, 2)
    # sample = ((sample * 255).astype(np.uint8))
    # plt.imshow(sample)
    # path = os.path.join('eval', filename)
    # plt.savefig(path)
    # plt.clf()

    sample = sample.view(3, 32, 32)
    path = os.path.join('eval', filename)
    torchvision.utils.save_image(sample, path, normalize=True)



def provide_samples(nn, num_samples=6, device='cpu'):
    '''
    Generate samples. For GANs, this means generating from
    the generator.
    '''
    if nn.name == 'gan':
        # Note: first argument is batch size of trained model,
        # second argument is size of latent space (don't change that!)
        noise = Variable(torch.randn(num_samples, 100)).to(device)
        samples = nn.model.generator(noise)
        # samples = samples.detach().cpu().numpy()
        print('sample shape is' ,samples.shape)
        for i in range(num_samples):
            sample = samples[i]
            filename = '{}_sample{}.png'.format(nn.name, i)
            plot_sample(sample, filename)

    if nn.name == 'vae':
        noise = Variable(torch.randn(num_samples, 100)).to(device)
        # or model.VAECarc.decoder?
        samples = nn.model.decoder(noise)
        samples = samples.detach().cpu().numpy()
        print('sample shape is' ,samples.shape)
        filename = '{}_sample.png'.format(nn.name)
        plot_sample(samples, filename)


def disentangle(nn, epsilon=1e-1, device='cpu'):
    '''
    Sample from the prior, make small perturbations for each dimension,
    and see if there are interesting changes.
    NOTE: Is this the right idea for "sample from the prior"?
    '''
    if nn.name == 'gan':
        noise = Variable(torch.randn(1, 100)).to(device)
        # Just take one sample
        non_perturbed = nn.model.generator(noise)
        non_perturbed = non_perturbed.detach().cpu().numpy()

    if nn.name == 'vae':
        noise = Variable(torch.randn(1, 100)).to(device)
        # Just take one sample
        non_perturbed = nn.model.decoder(noise)
        non_perturbed = non_perturbed.detach().cpu().numpy()

    # Plot the non-perturbed version
    non_perturbed_name = '{}_perturb_none.png'.format(nn.name)
    plot_sample(non_perturbed, non_perturbed_name)

    # Perturb some dimensions. We have 100, and saving 100 images is excessive,
    # and we just have to report what's 'interesting', so let's look on each quarter
    dims = [0, 25, 50, 75, 99]
    for dim in dims:
        pert_noise = noise
        print(pert_noise.size())
        pert_noise[0][dim] += epsilon
        pert = nn.model.generator(pert_noise)
        pert = pert.detach().cpu().numpy()
        pert_name = '{}_perturb_{}.png'.format(nn.name, dim)
        plot_sample(pert, pert_name)


def interpolate1(nn, device='cpu'):
    '''
    First interpolation scheme.
    Picks two random points in the latent space sampled from the prior,
    then for alpha = [0, 0.1, ..., 1], compute z'a = az0 + (1-a)z1, and
    plot resulting samples x'a = g(z'a).
    '''
    alphas = np.linspace(0, 1, 11)
    if nn.name == 'gan':
        z0 = Variable(torch.randn(1, 100)).to(device)
        z1 = Variable(torch.randn(1, 100)).to(device)

    if nn.name == 'vae':
        z0 = Variable(torch.randn(1, 100)).to(device)
        z1 = Variable(torch.randn(1, 100)).to(device)

    for alpha in alphas:
        z_alpha = (alpha * z0) + ((1 - alpha) * z1)
        if nn.name == 'gan':
            x_alpha = nn.model.generator(z_alpha).detach().cpu().numpy()
            filename = '{}_interp1_{}.png'.format(nn.name, alpha)
            plot_sample(x_alpha, filename)
        if nn.name == 'vae':
            x_alpha = nn.model.generator(z_alpha).detach().cpu().numpy()
            filename = '{}_interp1_{}_VAE.png'.format(nn.name, alpha)
            plot_sample(x_alpha, filename)


def interpolate2(nn, device='cpu'):
    '''
    Second interpolation scheme.
    Picks two random points in the latent space sampled from the prior,
    then generates x0 and x1; then for range of alpha,
    plots resulting samples of x_hat_a = ax0 + (1 - a)x1
    '''
    alphas = np.linspace(0, 1, 11)
    if nn.name == 'gan':
        z0 = Variable(torch.randn(1, 100)).to(device)
        z1 = Variable(torch.randn(1, 100)).to(device)
        x0 = nn.model.generator(z0).detach().cpu().numpy()
        x1 = nn.model.generator(z1).detach().cpu().numpy()
    if nn.name == 'vae':
        z0 = Variable(torch.randn(1, 100)).to(device)
        z1 = Variable(torch.randn(1, 100)).to(device)
        x0 = nn.model.encoder(z0).detach().cpu().numpy()
        x1 = nn.model.encoder(z1).detach().cpu().numpy()


    for alpha in alphas:
        x_hat_alpha = (alpha * x0) + ((1 - alpha) * x1)
        filename = '{}_interp2_{}.png'.format(nn.name, alpha)
        plot_sample(x_hat_alpha, filename)


if __name__ == '__main__':
    print('Plotting qualitatively...')

    # Parse arguments
    parser = argparse.ArgumentParser(description='Do qualitative evaluation of a trained model.')
    parser.add_argument('model_path', type=str,
                        help='The path to the trained model you want to evaluate.')
    parser.add_argument('eval_type', type=int,
                        help='The type of qualitative evaluation to do, as numbered in the question (1-3).')
    args = parser.parse_args()

    vae_batch_size = 32
    gan_batch_size = 128
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'gan' in args.model_path:
        nn = GAN(batch_size=gan_batch_size, device=device, model_path=args.model_path)
    if 'vae' in args.model_path:
        nn = VAE(batch_size=vae_batch_size, device=device, model_path=args.model_path)

    # Do qualitative examination
    if args.eval_type == 1:
        provide_samples(nn, device=device)
    elif args.eval_type == 2:
        disentangle(nn, device=device)
    elif args.eval_type == 3:
        interpolate1(nn, device=device)
    elif args.eval_type == 4:
        interpolate2(nn, device=device)
    else:
        raise ValueError('Unsupported evaluation type.')
