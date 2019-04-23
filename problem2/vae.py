'''
This file implements an MLP for problem 1 as specified:
3 layers, with SGD, learning rate 1e-3, batch size 512.
'''


import sys
sys.path.append("../..")

import torch
import torch.nn as nn
import math
import numpy as np

from torch.autograd import Variable
# from eval_methods import log_likelihood_estimate, importance_sample
from eval_methods import log_likelihood_estimate

from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, device='cpu'):
        super(Encoder, self).__init__()
        # self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
        self.fc_mu = nn.Linear(in_features=256, out_features=100)
        self.fc_logvar = nn.Linear(in_features=256, out_features=100)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inp):
        # if isinstance(inp, list):
        #     inp = inp[0].to(self.device)    # Weird hack
        inp = inp.float().view(-1, 28, 28).unsqueeze(1)
        # inp = inp.view(-1, 28, 28).unsqueeze(1)
        out = self.conv1(inp)
        out = self.avg_pool(self.elu(out))
        out = self.conv2(out)
        out = self.avg_pool(self.elu(out))
        out = self.conv3(out)
        out = self.elu(out)
        out = out.squeeze(2).squeeze(2)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, device='cpu'):
        super(Decoder, self).__init__()
        # self.device = device
        self.fc = nn.Linear(in_features=100, out_features=256)
        self.elu = nn.ELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = self.fc(inp)
        out = self.elu(out)
        out = out.unsqueeze(2).unsqueeze(2)
        out = self.conv1(out)
        out = self.up(self.elu(out))
        out = self.conv2(out)
        out = self.up(self.elu(out))
        out = self.conv3(out)
        out = self.elu(out)
        out = self.conv4(out)
        out = self.sigmoid(out)
        return out



class VAEArch(nn.Module):
    def __init__(self, device='cpu'):
        super(VAEArch, self).__init__()
        # self.device = device
        # self.encoder = Encoder(device=self.device)
        # self.decoder = Decoder(device=self.device)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inp):
        # Encode the input into distributional parameters
        mu, logvar = self.encoder(inp)
        # Reparamaterize and sample from the distribution
        stdev = torch.exp(0.5 * logvar)
        sample = stdev * torch.randn_like(stdev) + mu
        # Decode the sample
        out = self.decoder(sample)#.detach()
        out = out.squeeze(1).view(-1, 28 * 28)    # Re-flatten for loss calc
        return out, mu, logvar


class VAE():
    def __init__(self, batch_size, device='cpu', model_path=None):
        # Set up model
        self.device = device
        self.batch_size = batch_size
        self.model = VAEArch(device=self.device)
        print(torch.cuda.is_available())
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

    def train(self, train_loader, valid_loader, loss_fn=None, lr=1e-3, num_epochs=20):
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
            # print('Epoch {}: \t train_loss: {} \t'.format(epoch, train_loss))

    def train_epoch(self, train_loader, loss_fn=None, optimizer=None):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        for i, x in enumerate(tqdm(train_loader)):
            # Forward pass
            x = x[0].to(self.device)
            optimizer.zero_grad()
            output, mu, logvar = self.model(x)

            # Backward pass
            loss = loss_fn(x, output, self.batch_size, mu=mu, logvar=logvar)
            # print('train loss is' ,loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            num_batches += 1
        return epoch_loss/num_batches

    def valid_epoch(self, valid_loader, loss_fn=None):
        '''
        Does evaluation on the validation set for one epoch.
        '''
        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            num_batches = 0
            for i, x in enumerate(valid_loader):
                x = x[0].to(self.device)
                output, mu, logvar = self.model(x)
                loss = loss_fn(x, output, self.batch_size, mu=mu, logvar=logvar)
                num_batches += 1
                epoch_loss += loss
            return epoch_loss/num_batches

    def eval_log_likelihood(self, loader, device, batch_size, loss_fn=None):
        '''
        Approximates the log-likelihood via importance sampling.
        '''
        estimates = log_likelihood_estimate(self.model, loader, device, batch_size)
        return estimates

