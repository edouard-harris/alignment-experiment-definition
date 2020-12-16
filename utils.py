import fastai.data.external as ext
import torch
import torch.nn as nn
import pathlib as path
import uuid
import pickle as pk
from numpy import sqrt

def download_dataset(dataset_name, target_folder):
    dataset_dict = {
        'cifar10': ext.URLs.CIFAR,
        'mnist_sample': ext.URLs.MNIST_SAMPLE
    }
    return ext.untar_data(dataset_dict[dataset_name], dest=path.Path()/target_folder)

# Doc reference: https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/tf.image.per_image_whitening.md
def whiten_images(images_tensor):
    return (
        images_tensor - images_tensor.mean(1).unsqueeze(1)
    ) / (
        torch.cat(
            (images_tensor.std(1, unbiased=False).unsqueeze(1), torch.ones(images_tensor.shape[0], 1) / sqrt(images_tensor.shape[1])),
            1
        ).max(1)[0].unsqueeze(1)
    )

def crop_image(image_tensor, crop_size):
    return image_tensor[
        int((image_tensor.shape[0] - crop_size) / 2):int((image_tensor.shape[0] + crop_size) / 2),
        int((image_tensor.shape[1] - crop_size) / 2):int((image_tensor.shape[1] + crop_size) / 2)
    ]

def get_permutations(dataset_size, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return torch.randperm(dataset_size)

def build_experiment_name(experiment, experiment_nickname=''):
    return '{0}-lr_{1}-bs_{2}-eps_{3}_{4}'.format(
        experiment_nickname,
        'variable' if (type(experiment['inputs']['learning_rate']) == list) else experiment['inputs']['learning_rate'],
        experiment['inputs']['dataset_size'],
        experiment['inputs']['epochs'],
        uuid.uuid4().hex
    ).replace('.', '&').replace(',', '&').replace(' ', '')

def make_parameters_deterministic_(torch_model, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for layer in torch_model:
        try:
            layer.reset_parameters()
        except nn.modules.module.ModuleAttributeError:
            pass
