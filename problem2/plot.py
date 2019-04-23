'''
Plot VAE learning curves.
'''

import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('vae_learning_curves.log') as fp:
        lines = fp.readlines()
    train_losses, valid_losses = [], []
    for line in lines[1:-1]:    # Skip header and blank line footer
        line_split = line.split(',')
        line_split = [float(x) for x in line_split]
        train_losses.append(line_split[1])
        valid_losses.append(line_split[2])
    epochs = list(range(len(train_losses)))
    epochs = [float(epoch) for epoch in epochs]

    # Plot
    plt.plot(epochs, train_losses, color='blue', label='Train loss')
    plt.plot(epochs, valid_losses, color='orange', label='Valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('vae_learning_curves.png')
    plt.clf()
