'''
Script to plot GAN learning curves.
'''

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read the log file
    with open('gan_learning_curves.log', 'r') as fp:
        lines = fp.readlines()[1:]  # Exclude header
    epochs = []
    d_train_loss, d_valid_loss = [], []
    g_train_loss, g_valid_loss = [], []
    for line in lines[:-1]:         # Exclude final blank line
        line_split = line.split(',')
        line_split = [float(item) for item in line_split]
        epochs.append(line_split[0])
        d_train_loss.append(line_split[1])
        d_valid_loss.append(line_split[2])
        g_train_loss.append(line_split[3])
        g_valid_loss.append(line_split[4])

    # Plot curves
    plt.title('WGAN-GP Losses')
    plt.plot(epochs, d_train_loss, color='green', linestyle='-', label='Disc. train')
    plt.plot(epochs, d_valid_loss, color='green', linestyle='--', label='Disc. valid')
    plt.plot(epochs, g_train_loss, color='orange', linestyle='-', label='Gen. train')
    plt.plot(epochs, g_valid_loss, color='orange', linestyle='--', label='Gen. valid')
    plt.xlabel('Epochs')
    plt.ylabel('WGAN-GP loss')
    plt.legend()
    plt.savefig('gan_learning_curves.png')
    plt.clf()

    # Now read the cross-entropy log file
    with open('gan_d_crossentropy.log', 'r') as fp:
        lines = fp.readlines()[1:]  # Exclude header
    d_train_ce, d_valid_ce = [], []
    for line in lines[:-1]:         # Exclude final blank line
        line_split = line.split(',')
        line_split = [float(item) for item in line_split]
        d_train_ce.append(line_split[1])
        d_valid_ce.append(line_split[2])

    # Plot it too
    plt.title('Discriminator Cross-Entropy Loss')
    plt.plot(epochs, d_train_ce, color='green', linestyle='-', label='Disc. train')
    plt.plot(epochs, d_valid_ce, color='green', linestyle='--', label='Disc.valid')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    plt.savefig('gan_d_crossentropy.png')
    plt.clf()
