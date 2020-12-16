import matplotlib.pyplot as plt
import torch
from numpy import sqrt

def show_image(flat_image_tensor, color_channels=None):
    if color_channels is None:
        plt.imshow(flat_image_tensor.view(int(sqrt(len(flat_image_tensor))), int(sqrt(len(flat_image_tensor)))))
    else:
        plt.imshow(
            flat_image_tensor.view(
                int(sqrt(len(flat_image_tensor) / color_channels)),
                int(sqrt(len(flat_image_tensor) / color_channels)),
                color_channels
            )
        )
    plt.show()

def plot_learning_curves(losses=None, gradients=None):
    fig, ax1 = plt.subplots()

    if losses is not None:
        color = 'b'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.set_ylim(0, max(losses)*1.1)
        ax1.plot(range(len(losses)), losses, 'b.')
        ax1.tick_params(axis='y', labelcolor=color)

    if gradients is not None:
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis

        color = 'r'
        ax2.set_ylabel('2-norm of gradients', color=color)
        ax2.set_ylim(0, max(gradients)*1.1)
        ax2.plot(range(len(gradients)), gradients, 'r.')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.show()

# The calculations behind this plotting function can be found at
# https://drive.google.com/file/d/1rUxEoPjFjYMCNAowvP-4dq0JlYXmMzE4/view?usp=sharing
def plot_loss_function(loss_function):
    probs = torch.arange(0.01, 1, 0.01).repeat(2)
    acts = torch.cat((torch.zeros(probs.shape[0], 1), (probs / (1 - probs)).log().unsqueeze(1)), 1)
    targs = torch.cat((torch.ones(int(len(probs) / 2), 1), torch.zeros(int(len(probs) / 2), 1)), 0)

    diffs = probs - targs.view(-1)
    loss_values = [loss_function(act.unsqueeze(0), targ.unsqueeze(1)).item() for act, targ in zip(acts, targs)]

    plt.plot(diffs, loss_values, 'b.')
    plt.xlabel('prediction - target')
    plt.ylabel('loss')
    plt.show()
