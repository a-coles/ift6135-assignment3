import numpy as np
import random
import math


def distribution1(x, batch_size=512):
    # Distribution defined as (x, U(0,1)). Can be used for question 3
    while True:
        yield(np.array([(x, random.uniform(0, 1)) for _ in range(batch_size)]))


def distribution2(batch_size=512):
    # High dimension uniform distribution
    while True:
        yield(np.random.uniform(0, 1, (batch_size, 2)))


def distribution3(batch_size=512):
    # 1D gaussian distribution
    while True:
        yield(np.random.normal(0, 1, (batch_size, 1)))

def gaussian_1d(batch_size=1):
    # One dimensional gaussian distribution (for q1.4)
    while True:
        yield(np.random.normal(0, 1, (batch_size, 1)))

e = lambda x: np.exp(x)
tanh = lambda x: (e(x) - e(-x)) / (e(x)+e(-x))
def distribution4(batch_size=1):
    # arbitrary sampler
    f = lambda x: tanh(x*2+1) + x*0.75
    while True:
        yield(f(np.random.normal(0, 1, (batch_size, 1))))


def get_z(x, y):
    '''
    For Q1.2
    z = ax + (1 - a)y, where a is Uniform[0,1]
    '''
    #x = x.detach()
    #y = y.detach()
    a = random.uniform(0, 1)
    z = (a * x) + ((1 - a) * y)
    return z


def gaussian_1d_density(x):
    # Density function for a 1D standard gaussian (for q1.4)
    f_x = (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
    return f_x


if __name__ == '__main__':
    # Example of usage
    dist = iter(distribution1(0, 100))
    samples = next(dist)
