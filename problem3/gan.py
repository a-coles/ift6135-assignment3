'''
This file implements an MLP for problem 1 as specified:
3 layers, with SGD, learning rate 1e-3, batch size 512.
'''

import sys

import torch
import torch.nn as nn
import os
import random

from torch.autograd import Variable
from tqdm import tqdm
from memory_management_utils import dump_tensors


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        channels = 3
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)

            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(p=0.1),

            # State (256x16x16)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(p=0.1),

            # State (512x8x8)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(p=0.1),

            )

            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, inp):
        x = self.main_module(inp)  # .view(-1, 3, 32, 32)
        return self.output(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Analogous to the decoder in VAE paradigm.
        # The question specifies that the VAE decoder and GAN generator
        # should be the same (architecture and all).
        # NOTE: be careful changing the numbers here -- we MUST have an output
        # of [batch_size, 3 (channels), 32, 32] since SVHN is 32x32 and we are trying
        # to generate fake SVHN data.
        channels = 3

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=16, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        # self.output = nn.Tanh()
        self.output = nn.Sigmoid()

    def forward(self, inp):
        
        x = self.main_module(inp)
        return self.output(x)


class GANArch(nn.Module):
    def __init__(self):
        super(GANArch, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()

    def forward(self, inp):
        # Unused as of now
        pass


class GAN():
    def __init__(self, config=None, device='cpu', batch_size=None, model_path=None):
        # Set up model
        self.device = device
        self.model = GANArch()
        if model_path:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.name = 'gan'

        # Set up logging variables
        self.d_train_losses, self.d_valid_losses = [], []
        self.g_train_losses, self.g_valid_losses = [], []
        self.d_train_ce, self.d_valid_ce = [], []   # Cross-entropy

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def log_learning_curves(self):
        '''
        Logs the learning curve info to a csv.
        '''
        header = 'epoch,d_train_loss,d_valid_loss,g_train_loss,g_valid_loss\n'
        num_epochs = len(self.d_train_losses)
        with open(os.path.join('log', 'gan_learning_curves.log'), 'w') as fp:
            fp.write(header)
            for e in range(num_epochs):
                fp.write('{},{},{}\n'.format(e,
                                             self.d_train_losses[e],#, self.d_valid_losses[e],
                                             self.g_train_losses[e]))#, self.g_valid_losses[e]))

    def log_d_crossentropy(self):
        '''
        Logs the discriminator cross-entropy loss to a csv.
        '''
        header = 'epoch,d_train_ce,d_valid_ce\n'
        num_epochs = len(self.d_train_ce)
        with open(os.path.join('log', 'gan_d_crossentropy.log'), 'w') as fp:
            fp.write(header)
            for e in range(num_epochs):
                fp.write('{},{},{}\n'.format(e, self.d_train_ce[e], self.d_valid_ce[e]))

    def get_noise(self, batch_size):
        return Variable(torch.randn(batch_size, 100, 1, 1)).to(device=self.device)

    def train(self, train_loader, valid_loader, loss_fn=None, num_epochs=20, d_update=5):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        '''
        d_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-3)
        g_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            d_train_loss, g_train_loss = self.train_epoch(train_loader,
                                                          loss_fn=loss_fn,
                                                          d_optimizer=d_optimizer,
                                                          g_optimizer=g_optimizer,
                                                          d_update=d_update)
            # For logging
            self.d_train_losses.append(d_train_loss)
            self.g_train_losses.append(g_train_loss)

            print('Epoch {}:'.format(epoch))
            print(' \t d_train_loss: {}'.format(d_train_loss))
            print(' \t g_train_loss: {}'.format(g_train_loss))

            self.save_model('gan.pt')

            del d_train_loss
            del g_train_loss

    def train_epoch(self, train_loader, loss_fn=None, d_optimizer=None, g_optimizer=None, d_update=None):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        d_loss, g_loss = 0.0, 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):
            real = x.to(self.device)
            noise = self.get_noise(real.size(0))
            fake = self.model.generator(noise)
            fake_detach = self.model.generator(noise).detach().to(self.device)

            # Possibly update discriminator several times before updating generator.
            for j in range(d_update):
                # DISCRIMINATOR TRAINING
                d_err, ce = self.train_discriminator(real, fake_detach, loss_fn=loss_fn, d_optimizer=d_optimizer)
            d_loss += d_err

            # GENERATOR TRAINING
            g_err = self.train_generator(fake, loss_fn=loss_fn, g_optimizer=g_optimizer)
            g_loss += g_err

        self.d_train_ce.append(ce)
        return d_loss / self.batch_size, g_loss / self.batch_size

    def valid_epoch(self, valid_loader, loss_fn=None):
        '''
        Does evaluation on the validation set for one epoch.
        '''
        self.model.eval()
        
        d_loss, g_loss = 0.0, 0.0
        for i, (x, y) in enumerate(valid_loader):
            real = x.to(self.device)

            # Generate fake data and pass through discriminator
            noise = self.get_noise(real.size(0))
            fake = self.model.generator(noise).detach().to(self.device)

            # Train on the real and fake data.
            real = real.view(real.size(0), -1).to(self.device)
            fake = fake.view(fake.size(0), -1).to(self.device)
            d_real = self.model.discriminator(real)
            d_fake = self.model.discriminator(fake)

            # Get the WGAN-GP error.
            grad = self.get_gpgrad(real, fake)
            d_err = loss_fn(d_real, d_fake, grad=grad, objective='max')
            d_loss += d_err

            # Also get the cross-entropy error.
            ce = self.discriminator_ce(d_real, d_fake)

            # Pass through generator
            noise = self.get_noise(real.size(0))
            fake = self.model.generator(noise).detach().to(self.device)

            fake = fake.view(fake.size(0), -1).to(self.device)
            d_fake = self.model.discriminator(fake)
            # target = Variable(torch.ones(fake.size(0), 1)).to(self.device)
            grad = self.get_gpgrad(real, fake)
            g_err = loss_fn(d_real, d_fake, grad=grad)
            g_loss += g_err.item()

        self.d_valid_ce.append(ce.item())
        return d_loss / self.batch_size, g_loss / self.batch_size


    def get_gpgrad(self, real, fake):
        '''
        Gets the gradient needed for the gradient penalty.
        '''
        # For gradient penalty, need to sample a t uniformly from [0, 1], get x_hat,
        # and pass x_hat through the discriminator
        t = random.uniform(0, 1)
        x_hat = (t * fake.data) + ((1 - t) * real.data)
        x_hat.requires_grad = True
        d_x_hat = self.model.discriminator(x_hat)
        # Let us take gradient from vector, not scalar
        ones = torch.ones(d_x_hat.size()).to(self.device)
        grad = torch.autograd.grad(d_x_hat, x_hat, 
                                   grad_outputs=ones,
                                   retain_graph=True, create_graph=True)[0]
        # print('grad:', grad.size())
        #print('grad[0]', grad[0])
        #print('grad[1',  grad[1])
        return grad

    def discriminator_ce(self, d_real, d_fake):
        # Let real = class 1 and fake = class 0.
        real_target = Variable(torch.ones(d_real.size(0), 1)).to(self.device)
        fake_target = Variable(torch.zeros(d_fake.size(0), 1)).to(self.device)
        d_real = d_real.view(-1, 1)
        d_fake = d_fake.view(-1, 1)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        ce = bce_loss_fn(d_real, real_target) + bce_loss_fn(d_fake, fake_target)
        return ce

    def train_discriminator(self, real, fake, loss_fn=None, d_optimizer=None):
        '''
        Does a training update to the discriminator.
        '''
        # Set gradients to 0
        self.model.discriminator.train()
        d_optimizer.zero_grad()

        # Train on the real and fake data.
        # real = real.view(real.size(0), -1).to(self.device)
        # fake = fake.view(fake.size(0), -1).to(self.device)
        d_real = self.model.discriminator(real)
        d_fake = self.model.discriminator(fake)
        
        # Get the WGAN-GP error.
        grad = self.get_gpgrad(real, fake)
        d_optimizer.zero_grad()
        err = loss_fn(d_real, d_fake, grad=grad, objective='max')
        err.backward(retain_graph=False)

        # Also get cross-entropy error. 
        ce = self.discriminator_ce(d_real, d_fake)

        # Update weights.
        d_optimizer.step()

        del real
        del fake
        return err.item(), ce.item()

    def train_generator(self, fake, loss_fn=None, g_optimizer=None):
        '''
        Does a training update to the generator.
        '''
        # Set gradients to 0
        self.model.generator.train()
        g_optimizer.zero_grad()

        # Generate fake data.
        # Note: right now the discriminator is a simple MLP, so we need to flatten our images.
        # fake = fake.view(fake.size(0), -1).to(self.device)
        d_fake = self.model.discriminator(fake)

        # Update.
        err = loss_fn(None, d_fake, objective='min')
        err.backward()
        g_optimizer.step()
        return err.item()
