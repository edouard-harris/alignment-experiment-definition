import torch
import torch.nn as nn
from numpy import sqrt

def cross_entropy_loss(activations, targets):
    return nn.CrossEntropyLoss()(activations, targets.long().view(-1))

def accuracy_loss(activations, targets):
    return (1. - (-nn.CrossEntropyLoss(reduction='none')(activations, targets.long().view(-1))).exp()).mean()

def calculate_accuracy(activations, targets):
    return torch.where(activations.argmax(1) == targets.long().view(-1), 1, 0).sum() / activations.shape[0]

def gradient_2_norm(raw_gradients):
    return sqrt(sum([(layer_grads**2).sum().item() for layer_grads in raw_gradients]))

# This lets us build binary loss functions by connecting loss functions piecewise
# on the (predictions - targets) domain. domain_interval is a list of floats that
# defines the boundaries between loss functions in piecewise_func_list.
# See https://drive.google.com/file/d/1rUxEoPjFjYMCNAowvP-4dq0JlYXmMzE4/view?usp=sharing
# for calculations related to this function factory.
# Example:
# lf = metrics.loss_function_factory([-1, 0, 1], [lambda x: x**2, lambda x: x])
# lib.plot_loss_function(lf)
def loss_function_factory(domain_interval, piecewise_func_list):
    epsilon = 1e-15

    if not all(domain_interval[i] < domain_interval[i + 1] for i in range(len(domain_interval) - 1)):
        raise Exception('The domain_interval list needs to be ordered from lowest to highest.')

    if max(domain_interval) != 1. or min(domain_interval) != -1.:
        raise Exception('The domain_interval list should start with -1 and end with 1.')

    if len(domain_interval) != len(piecewise_func_list) + 1:
        raise Exception('The domain_interval list should have one more element than the piecewise_func_list.')

    if not all((
        abs(piecewise_func_list[i](domain_interval[i + 1]) - piecewise_func_list[i + 1](domain_interval[i + 1])) < epsilon
    ) for i in range(len(domain_interval) - 2)):
        raise Exception('Function values need to be equal at the boundaries between intervals.')

    domains_list = [[domain_interval[i], domain_interval[i + 1]] for i in range(len(domain_interval) - 1)]
    base_loss_func = lambda acts, targs: nn.Softmax(dim=1)(acts)[:,1] - targs.view(-1)

    def custom_loss_func(activations, targets):
        if activations.shape[1] != 2:
            raise Exception('Custom loss functions are only designed to work with binary classification problems.')

        losses_, diffs = base_loss_func(activations, targets), base_loss_func(activations, targets)

        for domain, loss_func in zip(domains_list, piecewise_func_list):

# We need to cover the edge case of (predictions - targets) = 1 below, because
# PyTorch calculates the gradient incorrectly if we leave in the `& (diffs < domain[1])`
# in the conditional statement. Note that adding a buffer like `& (diffs < domain[1] + 0.01)`
# (in this edge case ONLY!) would also solve the problem.
            if domain[1] == 1:
                losses_ = torch.where(
                    domain[0] <= diffs,
                    loss_func(losses_),
                    losses_
                )

            else:
                losses_ = torch.where(
                    (domain[0] <= diffs) & (diffs < domain[1]),
                    loss_func(losses_),
                    losses_
                )

        return losses_.mean()

    return custom_loss_func
