'''
This file implements an MLP for problem 1 as specified:
3 layers, with SGD, learning rate 1e-3, batch size 512.
'''

import sys
sys.path.append("../..")

import torch
import torch.nn as nn
import os

from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Several-layer MLP for now
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        # Question: need to take softmax?

    def forward(self, inp):
        # Save for gradient penalty
        self.d_inp = Variable(inp, requires_grad=True)
        self.d_inp.retain_grad()

        # Forward prop
        out = self.fc1(self.inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Analogous to the decoder in VAE paradigm.
        # The question specifies that the VAE decoder and GAN generator
        # should be the same (architecture and all).
        self.fc = nn.Linear(in_features=100, out_features=256)
        self.elu = nn.ELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.deconv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4)
        self.deconv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2)
        self.deconv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2)
        self.deconv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2)

    def forward(self, inp):
        print('inp size:', inp.size())
        out = self.fc(inp)
        print('fc size:', out.size())
        out = self.elu(out)
        print('elu size:', out.size())

        #out = out.view(-1, 16, 16)
        #out = out.unsqueeze(1)
        print('unsqueeze size:', out.size())
        out = self.deconv1(out)
        print('conv1 size:', out.size())
        out = self.up(self.elu(out))
        out = self.deconv2(out)
        out = self.up(self.elu(out))
        out = self.deconv3(out)
        out = self.elu(out)
        out = self.deconv4(out)
        return out


class GANArch(nn.Module):
    def __init__(self):
        super(GANArch, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()

    def forward(self, inp):
        # Unused as of now
        pass


class GAN():
    def __init__(self, config=None, device='cpu', model_path=None):
        # Set up model
        self.device = device
        self.model = GANArch()
        if model_path:
            self.load_model(model_path)
        self.model = self.model.to(self.device)
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
        num_epochs = len(self.train_losses)
        with open(os.path.join('log', 'gan_learning_curves.log'), 'w') as fp:
            fp.write(header)
            for e in range(num_epochs):
                fp.write('{},{},{}\n'.format(e,
                                             self.d_train_losses[e], self.d_valid_losses[e],
                                             self.g_train_losses[e], self.g_valid_losses[e]))

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
        return Variable(torch.randn(batch_size, 100, device=self.device))

    def train(self, train_loader, valid_loader, loss_fn=None, num_epochs=20, d_update=1):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        '''
        d_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        g_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        for epoch in range(num_epochs):
            d_train_loss, g_train_loss = self.train_epoch(train_loader,
                                                          loss_fn=loss_fn,
                                                          d_optimizer=d_optimizer, g_optimizer=g_optimizer,
                                                          d_update=d_update)
            d_valid_loss, g_valid_loss = self.valid_epoch(valid_loader,
                                                          loss_fn=loss_fn)

            # For logging
            self.d_train_losses.append(d_train_loss)
            self.d_valid_losses.append(d_valid_loss)
            self.g_train_losses.append(g_train_loss)
            self.g_valid_losses.append(g_valid_loss)

            print('Epoch {}:'.format(epoch))
            print(' \t d_train_loss: {} \t d_valid_loss: {}'.format(epoch, d_train_loss, d_valid_loss))
            print(' \t g_train_loss: {} \t g_valid_loss: {}'.format(epoch, g_train_loss, g_valid_loss))

    def train_epoch(self, train_loader, loss_fn=None, d_optimizer=None, g_optimizer=None, d_update=1):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        d_loss, g_loss = 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):   # Possibly change this call with dataloader
            real = x.to(self.device)

            # Possibly update discriminator several times before updating generator.
            for j in range(d_update):
                # DISCRIMINATOR TRAINING
                # Generate some fake data for the discriminator. Detach so that
                # we don't calculate gradients for the generator.
                noise = self.get_noise(real.size(0))
                fake = self.model.generator(noise).detach()

                # Train the discriminator.
                d_err, d_pred_real, d_pred_fake = self.model.train_discriminator(real, fake, d_optimizer=d_optimizer)
                d_loss += d_err

            # GENERATOR TRAINING
            # Generate some fake data, don't detach this time.
            noise = Variable(torch.randn(real.size(0), 100))
            fake = self.model.generator(noise)

            # Train the generator.
            g_err = self.model.train_generator(fake, g_optimizer=g_optimizer)
            g_loss += g_err

        return d_loss, g_loss

    def valid_epoch(self, valid_loader, loss_fn=None):
        '''
        Does evaluation on the validation set for one epoch.
        '''
        self.model.eval()
        d_loss, g_loss = 0.0
        for i, (x, y) in enumerate(valid_loader):
            real = x

            # Pass through discriminator
            noise = Variable(torch.randn(real.size(0), 100))
            fake = self.model.generator(noise).detach()

            pred_real = self.model.discriminator(real)
            target_real = Variable(torch.ones(real.size(0), 1))     # Let real = class 1
            grad_real = torch.autograd.grad(pred_real.mean(), self.model.d_inp, retain_graph=True)
            err_real = d_loss(pred_real, target_real, grad=grad_real)

            pred_fake = self.model.discriminator(fake)
            target_fake = Variable(torch.zeros(fake.size(0), 1))    # Let fake = class 0
            grad_fake = torch.autograd.grad(pred_fake.mean(), self.model.d_inp, retain_graph=True)
            err_fake = d_loss(pred_fake, target_fake, grad=grad_fake)

            d_err = err_real + err_fake
            d_loss += d_err

            bce_loss = nn.BCELoss()     # Track cross-entropy loss too
            ce = bce_loss(pred_real, target_real) + bce_loss(pred_fake, target_fake)
            self.d_valid_ce.append(ce)

            # Pass through generator
            noise = Variable(torch.randn(real.size(0), 100))
            fake = self.model.generator(noise)

            pred = self.model.discriminator(fake)
            target = Variable(torch.ones(fake.size(0), 1))
            grad = torch.autograd.grad(pred.mean(), self.model.d_inp, retain_graph=True)
            g_err = g_loss(pred, target, grad=grad)
            g_loss += g_err

        return d_loss, g_loss

    def train_discriminator(self, real, fake, d_loss=None, d_optimizer=None):
        '''
        Does a training update to the discriminator.
        '''
        d_optimizer.zero_grad()

        # Train on the real data.
        pred_real = self.model.discriminator(real)
        target_real = Variable(torch.ones(real.size(0), 1))     # Let real = class 1
        grad_real = torch.autograd.grad(pred_real.mean(), self.model.d_inp, retain_graph=True)
        err_real = d_loss(pred_real, target_real, grad=grad_real)
        err_real.backward()

        # Train on the fake data.
        pred_fake = self.model.discriminator(fake)
        target_fake = Variable(torch.zeros(fake.size(0), 1))    # Let fake = class 0
        grad_fake = torch.autograd.grad(pred_fake.mean(), self.model.d_inp, retain_graph=True)
        err_fake = d_loss(pred_fake, target_fake, grad=grad_fake)
        err_fake.backward()

        # Update weights.
        d_optimizer.step()

        # Update cross-entropy log.
        bce_loss = nn.BCELoss()
        ce = bce_loss(pred_real, target_real) + bce_loss(pred_fake, target_fake)
        self.d_train_ce.append(ce)

        err = err_real + err_fake
        return err, pred_real, pred_fake

    def train_generator(self, fake, g_loss=None, g_optimizer=None):
        '''
        Does a training update to the generator.
        '''
        g_optimizer.zero_grad()

        # Generate fake data.
        pred = self.model.discriminator(fake)
        # We want to fool the discriminator -- we hope to generate data that
        # look to be of class 1 (real).
        target = Variable(torch.ones(fake.size(0), 1))
        grad = torch.autograd.grad(pred.mean(), self.model.d_inp, retain_graph=True)

        # Update.
        err = g_loss(pred, target, grad=grad)
        err.backward()
        g_optimizer.step()

        return err
