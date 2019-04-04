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
import matplotlib.pyplot as plt

from assignment.problem3.gan import GAN
from torch.autograd import Variable
from torchvision import transforms


def provide_samples(nn, num_samples=4):
    '''
    Generate samples. For GANs, this means generating from
    the generator.
    '''
    if nn.name == 'gan':
        # Note: first argument is batch size of trained model,
        # second argument is size of latent space (don't change that!)
        noise = Variable(torch.randn(128, 100))
        samples = nn.model.generator(noise)[:num_samples]

    for i in range(num_samples):
        sample = transforms.ToPILImage()(samples[i])
        plt.imshow(sample)
        path = os.path.join('eval', '{}_sample{}.png'.format(nn.name, i))
        plt.savefig(sample, path)
        plt.clf()


def disentangle(nn, epsilon=1e-1):
    '''
    Sample from the prior, make small perturbations for each dimension,
    and see if there are interesting changes.
    '''
    if nn.name == 'gan':
        noise = Variable(torch.randn(128, 100))
        # Just take one sample
        z = nn.model.generator(noise)[0]

    # Plot the non-perturbed version
    non_perturbed = transforms.ToPILImage()(z)
    plt.imshow(non_perturbed)
    plt.savefig(non_perturbed, os.path.join('eval', '{}_perturb_none.png'.format(nn.name)))
    plt.clf()

    # Perturb some dimensions. We have 100, and saving 100 images is excessive,
    # and we just have to report what's 'interesting', so let's look on each quarter
    dims = [0, 25, 50, 75, 99]
    for dim in dims:
        


def interpolate():
    pass


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Do qualitative evaluation of a trained model.')
    parser.add_argument('model_path', type=str,
                        help='The path to the trained model you want to evaluate.')
    parser.add_argument('eval_type', type=int,
                        help='The type of qualitative evaluation to do, as numbered in the question (1-3).')
    args = parser.parse_args()

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'gan' in args.model_path:
        nn = GAN(device=device, model_path=args.model_path)

    # Do qualitative examination
    if args.eval_type == 1:
        provide_samples()
    elif args.eval_type == 2:
        disentangle()
    elif args.eval_type == 3:
        interpolate()
    else:
        raise ValueError('Unsupported evaluation type.')
