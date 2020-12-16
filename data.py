import fastai.torch_core as tc
import fastai.data.load as load
import torch
import PIL as pil
import pathlib as path
import dill

from numpy import prod

import utils

################################################################################

TARGET_FOLDER = 'data'
DATA_PATH = utils.download_dataset('cifar10', TARGET_FOLDER)
CLASSIFICATION_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
EXPERIMENT_FOLDER = 'experiments'

################################################################################

def build_dataset(
    dataset_size=None,
    classification_labels=CLASSIFICATION_LABELS,
    data_path=DATA_PATH,
    randomize_dataset_seed=None,
    crop_size=28
):
    if dataset_size % len(classification_labels) != 0:
        raise Exception('dataset_size must be divisible by {}.'.format(len(classification_labels)))

    if crop_size % 2 != 0:
        raise Exception('crop_size must be an even integer.')

    raw_x_data = []
    raw_y_data = []

    for label, i in zip(classification_labels, range(len(classification_labels))):
        print('Building raw data for label "{}"...'.format(label))
        im_files = (data_path/'train'/label).ls()

        selected_images = range(len(im_files)) if (
            randomize_dataset_seed is None
        ) else utils.get_permutations(len(im_files), randomize_dataset_seed)

        raw_x_data += [[
            utils.crop_image(
                tc.tensor(pil.Image.open(im_file)), crop_size
            ) for im_file in im_files.sorted()[selected_images][:int(dataset_size / len(classification_labels))]
        ]]

        raw_y_data += [tc.tensor(float(i))]*len(raw_x_data[-1])

    x_data = torch.cat([torch.stack(image_class) for image_class in raw_x_data]).view(-1, prod(raw_x_data[0][0].shape)).float() / 255
    y_data = torch.stack(raw_y_data).view(-1, 1)

    return (
        x_data,
        y_data
    )

def transform_dataset(dataset, randomize_labels_seed=None, whiten_images=True):
    x_data, y_data = dataset
    return (
        x_data if (not whiten_images) else utils.whiten_images(x_data),
        y_data if (randomize_labels_seed is None) else y_data[utils.get_permutations(len(y_data), randomize_labels_seed)]
    )

def build_data_loader(dataset, batch_size=128):
    return load.DataLoader(list(zip(*dataset)), batch_size=batch_size, shuffle=True)

def save_experiment(experiment, experiment_nickname='', experiment_folder=EXPERIMENT_FOLDER):
    path.Path.mkdir(path.Path()/experiment_folder, exist_ok=True)
    experiment_name = utils.build_experiment_name(experiment, experiment_nickname)

    with open(path.Path()/experiment_folder/'{}.dill'.format(experiment_name), 'wb') as f:
        dill.dump(experiment, f)

    return experiment_name

def load_experiment(file_name, experiment_folder=EXPERIMENT_FOLDER):
    with open(path.Path()/experiment_folder/file_name, 'rb') as f:
        experiment = dill.load(f)

    return experiment
