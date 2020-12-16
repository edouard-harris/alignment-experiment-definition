import torch.nn as nn
import torch

import lib
import data
import utils
import metrics
import viz

def test():

    inputs = {
        'model': nn.Sequential(nn.Linear(28*28, 32), nn.ReLU(), nn.Linear(32, 2)),
        'learning_rate': 0.1,
        'dataset_size': 512,
        'batch_size': 512,
        'epochs': 10000,
        'parameters_seed': 0,
        'loss_function': metrics.cross_entropy_loss,
        'classification_labels': ['3', '7'],
        'whiten_images': False,
        'data_path': utils.download_dataset('mnist_sample', 'data'),
        'randomize_dataset_seed': 0,
        'experiment_nickname': 'TEST-EXPERIMENT'
    }

    experiment = lib.run_one_experiment(**inputs)

    return experiment
