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
from assignment.samplers import get_z


class MLP3Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes=1):
        super(MLP3Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.output = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp, dist_type='jsd'):
        # Keep gradient at this point for WD gradient penalty
        #self.inp = Variable(inp, requires_grad=True)
        #self.inp.retain_grad()

        # Begin fprop
        out = self.fc1(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.output(out)
        if dist_type == 'jsd':
            # WGAN-GP discriminator, importantly, JUST does regression, no squishing
            out = self.sigmoid(out)
        return out


class MLP():
    def __init__(self, config, device='cpu', model_path=None):
        # Set up model
        self.device = device
        self.model = MLP3Layer(config['input_size'], config['hidden_size1'],
                               config['hidden_size2'], config['hidden_size3'])
        if model_path:
            self.load_model(model_path)
        self.model = self.model.to(self.device)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def get_grad(self, x, y):
        '''
        Gets the gradient needed for gradient penalty.
        '''
        z = get_z(x, y)
        #z = Variable(z, requires_grad=True)
        z.requires_grad = True
        Dz = self.model(z, dist_type='wd')
        #print('Dz mean:', Dz.mean())
        #Dz.mean().backward()
        Dz.backward(torch.ones_like(Dz))
        grad = z.grad
        return grad

    def train(self, p, q, loss_fn=None, lr=1e-3, num_epochs=60, dist_type='jsd'):
        '''
        This function trains to get D_theta or T_theta in the Latex.
        '''
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.model.train()

        for epoch in range(num_epochs):
            # Sample from the distribution as "input" data.
            # x and y here are the notation used in the Latex.
            x = torch.from_numpy(next(p)).float().to(self.device)
            y = torch.from_numpy(next(q)).float().to(self.device)

            # Forward pass
            optimizer.zero_grad()
            Dx = self.model(x, dist_type=dist_type)
            Dy = self.model(y, dist_type=dist_type)
            grad = None

            # Account for WD gradient penalty
            if dist_type == 'wd':
                '''z = get_z(x, y)
                z = Variable(z, requires_grad=True)
                Dz = self.model(z, dist_type='wd')
                #grad = torch.autograd.grad(Dz.mean(), self.model.inp, retain_graph=True)
                Dz[0].backward()
                grad = z.grad
                #print('grad:', grad)'''
                grad = self.get_grad(x, y)
                optimizer.zero_grad()

            # Backward pass
            loss = loss_fn(Dx, Dy, grad=grad)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print('Epoch {}: \t Loss: {}'.format(epoch, loss))

    def estimate_jsd(self, x, y):
        '''
        This function returns the actual JS divergence estimate.
        '''
        self.model.eval()
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        Dx = self.model(x, dist_type='jsd')
        Dy = self.model(y, dist_type='jsd')

        Ex = 0.5 * torch.log(Dx).mean()
        Ey = 0.5 * torch.log(1 - Dy).mean()
        jsd = math.log(2) + Ex + Ey
        return jsd

    def estimate_wd(self, x, y, lamb=10):
        '''
        This function returns the actual Wasserstein distance estimate.
        '''
        self.model.eval()
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        Dx = self.model(x, dist_type='wd')
        Dy = self.model(y, dist_type='wd')
        '''z = get_z(x, y)
        Dz = self.model(z, dist_type='wd')
        grad = torch.autograd.grad(Dz.mean(), self.model.inp, retain_graph=True)'''
        grad = self.get_grad(x, y)

        '''grad = grad[0]  # Take first item in mysterious tuple
        grad_penalty = lamb * (torch.norm(grad, 2) - 1).pow(2).mean()
        wd = Dx.mean() - Dy.mean() - grad_penalty'''
        loss = self.loss_fn(Dx, Dy, grad=grad)
        #wd = -1 * loss  # The loss function flips sign to maximize, so flip it back
        wd = loss
        return wd

    def estimate_unk(self, x, f0_x):
        '''
        This function returns the density estimate from the unknown distribution (q1.4).
            f1 = f0 D*(x) / (1 - D*(x))
            where D*(x) = argmax_D E(log(Dx)) + E(log(1-Dx))
        '''
        self.model.eval()
        x = torch.from_numpy(x).float().to(self.device)
        f0_x = torch.from_numpy(f0_x).float().to(self.device)
        Dx = self.model(x)
        f1_x = f0_x * (Dx / (1 - Dx))
        return f1_x
