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
        # Several-layer MLP for now
        '''
        self.fc1 = nn.Linear(3072, 4)
        self.fc4 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        '''
        
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1

        channels = 3
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, inp):
        '''
        # Save for gradient penalty
        self.d_inp = Variable(inp, requires_grad=True)
        self.d_inp.retain_grad()

        # Forward prop
        out = self.fc1(self.d_inp)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.tanh(out)
        '''
        # self.d_inp = Variable(inp, requires_grad=True)
        # self.d_inp = self.d_inp.view(-1, 3, 32, 32)

        x = self.main_module(inp.view(-1, 3, 32, 32))
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

        
        # self.fc = nn.Linear(in_features=100, out_features=16)
        # self.bn_fc = nn.BatchNorm1d(16)
        # self.elu = nn.ELU()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=4)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=4, padding=2)
        # self.bn2 = nn.BatchNorm2d(3)
        # self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=2)
        # self.bn3 = nn.BatchNorm2d(3)
        # self.tanh = nn.Tanh()
        # # self.sig = nn.Sigmoid()
        # torch.nn.init.xavier_uniform_(self.fc.weight)
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        # torch.nn.init.xavier_uniform_(self.conv3.weight)
        channels = 3

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, inp):
        
        # out = self.fc(inp)
        # out = self.bn_fc(out)
        # # print('     fc:', out.size())
        # out = self.elu(out)
        # out = out.unsqueeze(2).unsqueeze(2)
        # # print('     reshape:', out.size())
        # out = self.conv1(out)
        # out = self.bn1(out)
        # # print('     conv1:', out.size())
        # out = self.up(self.elu(out))
        # # print('     up1:', out.size())
        # out = self.conv2(out)
        # out = self.bn2(out)
        # # print('     conv2:', out.size())
        # out = self.up(self.elu(out))
        # # print('     up2:', out.size())
        # out = self.conv3(out)
        # out = self.tanh(out)

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
        return Variable(torch.randn(batch_size, 100, 1, 1)).to(device=self.device)

    def train(self, train_loader, valid_loader, loss_fn=None, num_epochs=50, d_update=1):
        '''
        Wrapper function for training on training set + evaluation on validation set.
        '''
        d_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-3)
        g_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            d_train_loss, g_train_loss = self.train_epoch(train_loader,
                                                          loss_fn=loss_fn,
                                                          d_optimizer=d_optimizer, 
                                                          g_optimizer=g_optimizer,
                                                          d_update=d_update)
            # with torch.no_grad():   
            d_valid_loss, g_valid_loss = self.valid_epoch(valid_loader,
                                                          loss_fn=loss_fn)

            # For logging
            # already floats

            self.d_train_losses.append(d_train_loss)
            self.d_valid_losses.append(d_valid_loss)
            self.g_train_losses.append(g_train_loss)
            self.g_valid_losses.append(g_valid_loss)

            print('Epoch {}:'.format(epoch))
            # print('  \t d_train_loss: {}'.format(d_train_loss))
            # print('  \t g_train_loss: {}'.format(g_train_loss))
            print(' \t d_train_loss: {} \t d_valid_loss: {}'.format(d_train_loss, d_valid_loss))
            print(' \t g_train_loss: {} \t g_valid_loss: {}'.format(g_train_loss, g_valid_loss))

            self.save_model('gan.pt')

    def train_epoch(self, train_loader, loss_fn=None, d_optimizer=None, g_optimizer=None, d_update=None):
        '''
        Does training for one epoch.
        '''
        self.model.train()
        d_loss, g_loss = 0.0, 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):   # Possibly change this call with dataloader
            real = x.to(self.device)
            noise = self.get_noise(real.size(0))
            fake = self.model.generator(noise)
            fake_detach = self.model.generator(noise).detach().to(self.device)

            # Possibly update discriminator several times before updating generator.
            for j in range(d_update):
                # DISCRIMINATOR TRAINING
                d_err, ce = self.train_discriminator(real, fake_detach, loss_fn=loss_fn, d_optimizer=d_optimizer)
            # print("train_epoch d_err", type(d_err))

            d_loss += d_err
            # print("train_epoch d_loss", type(d_loss))

            # GENERATOR TRAINING
            g_err = self.train_generator(fake, loss_fn=loss_fn, g_optimizer=g_optimizer)
            g_loss += g_err

        self.d_train_ce.append(ce)
        return d_loss, g_loss

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
        return d_loss, g_loss


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
        grad = torch.autograd.grad(d_x_hat.mean(), x_hat, retain_graph=True, create_graph=True)[0]
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
        real = real.view(real.size(0), -1).to(self.device)
        fake = fake.view(fake.size(0), -1).to(self.device)
        d_real = self.model.discriminator(real)
        d_fake = self.model.discriminator(fake)

        # copy and paste the whole func def to avid deep copy problems
        
        t = random.uniform(0, 1)
        x_hat = (t * fake.data) + ((1 - t) * real.data)
        x_hat.requires_grad = True
        d_x_hat = self.model.discriminator(x_hat)
        grad = torch.autograd.grad(d_x_hat.mean(), x_hat, retain_graph=True, create_graph=True)[0]
        # print('grad', grad)
        #        grad_fn=<CudnnConvolutionBackwardBackward>
        (-d_real).mean().backward(retain_graph=True)
        d_fake.mean().backward(retain_graph=True)
        lamb = 20
        grad_penalty = lamb * (torch.norm(grad, dim=1) - 1).pow(2).mean()
        # print("grad_penalty", grad_penalty)
        grad_penalty.backward()
        
        # Get the WGAN-GP error.
        # grad = self.get_gpgrad(real, fake)
        # d_optimizer.zero_grad()
        # err = loss_fn(d_real, d_fake, grad=grad, objective='max')
        # err.backward(retain_graph=False)

 
        ce = self.discriminator_ce(d_real, d_fake)

        # Update weights.
        d_optimizer.step()

        return grad_penalty.item(), ce.item()

    def train_generator(self, fake, loss_fn=None, g_optimizer=None):
        '''
        Does a training update to the generator.
        '''
        # Set gradients to 0
        self.model.generator.train()
        g_optimizer.zero_grad()

        # Generate fake data.
        # Note: right now the discriminator is a simple MLP, so we need to flatten our images.
        fake = fake.view(fake.size(0), -1).to(self.device)
        d_fake = self.model.discriminator(fake)

        # Update.
        err = loss_fn(None, d_fake, objective='min')
        err.backward()
        g_optimizer.step()
        return err.item()
