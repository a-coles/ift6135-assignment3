'''
This file implements an MLP for problem 1 as specified:
3 layers, with SGD, learning rate 1e-3, batch size 512.
'''

import sys
sys.path.append("../..")

import torch
import torch.nn as nn
import math

from torch.autograd import Variable
from assignment.problem2.eval_methods import log_likelihood_estimate


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
        self.fc_mu = nn.Linear(in_features=256, out_features=100)
        self.fc_logvar = nn.Linear(in_features=256, out_features=100)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.avg_pool(self.elu(out))
        out = self.conv2(out)
        out = self.avg_pool(self.elu(out))
        out = self.conv3(out)
        out = self.elu(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=100, out_features=256)
        self.elu = nn.ELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2)

    def forward(self, inp):
        out = self.fc(inp)
        out = self.elu(out)
        out = self.conv1(out)
        out = self.up(self.elu(out))
        out = self.conv2(out)
        out = self.up(self.elu(out))
        out = self.conv3(out)
        out = self.elu(out)
        out = self.conv4(out)
        return out


class VAEArch(nn.Module):
    def __init__(self):
        super(VAEArch, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inp):
        # Encode the input into distributional parameters
        mu, logvar = self.encoder(inp)

        # Reparamaterize and sample from the distribution
        stdev = torch.exp(0.5 * logvar)
        sample = torch.rand_like(stdev) + mu

        # Decode the sample
        out = self.decoder(sample)
        return out, mu, logvar


class VAE():
    def __init__(self, config=None, device='cpu', model_path=None):
        # Set up model
        self.device = device
        self.model = VAEArch()
        if model_path:
            self.load_model(model_path)
        self.model = self.model.to(self.device)

        # Set up logging variables
        self.train_losses = []
        self.valid_losses = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log(self):
        '''
        Logs the learning curve info to a csv.
        '''
        header = 'epoch,train_loss,valid_loss\n'
        num_epochs = len(self.train_losses)
        with open('vae_learning_curves.log', 'w') as fp:
            fp.write(header)
            for e in range(num_epochs):
                fp.write('{},{},{}\n'.format(e, self.train_losses[e], self.valid_losses[e]))

    def train(self, train_loader, valid_loader, loss_fn=None, lr=3e-4, num_epochs=20):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        '''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, loss_fn=loss_fn, optimizer=optimizer)
            valid_loss = self.valid_epoch(valid_loader, loss_fn=loss_fn)

            # For logging
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            print('Epoch {}: \t train_loss: {} \t valid_loss: {}'.format(epoch, train_loss, valid_loss))

    def train_epoch(self, train_loader, loss_fn=None, optimizer=None):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        epoch_loss = 0.0
        for i, (x, y) in enumerate(train_loader):   # Possibly change this call with dataloader
            # Forward pass
            optimizer.zero_grad()
            output, mu, logvar = self.model(x)

            # Backward pass
            loss = loss_fn(x, output, mu=mu, logvar=logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        return epoch_loss

    def valid_epoch(self, valid_loader, loss_fn=None):
        '''
        Does evaluation on the validation set for one epoch.
        '''
        self.model.eval()
        epoch_loss = 0.0
        for i, (x, y) in enumerate(valid_loader):
            output, mu, logvar = self.model(x)
            loss = loss_fn(x, output, mu=mu, logvar=logvar)
            epoch_loss += loss
        return epoch_loss

    def eval_log_likelihood(self, loader):
        '''
        Approximates the log-likelihood via importance sampling.
        '''
        self.model.eval()
        log_likelihood = 0.0
        for i, x in enumerate(loader):
            estimate = log_likelihood_estimate()
            log_likelihood += estimate
        return log_likelihood
