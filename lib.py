import copy as cp

import utils
import data
import metrics
import viz

################################################################################

class Optimizer:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr

    def step(self):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

################################################################################

def train_epoch(model, data_loader=None, optimizer=None, loss_function=None):
    gradients = []

    for x_batch, y_batch in data_loader:
        loss = loss_function(model(x_batch), y_batch)
        loss.backward()
        gradients += [[params.grad.clone().detach() for params in model.parameters()]]
        optimizer.step()
        optimizer.zero_grad()

# Note that we are discarding all but the most recent set of gradients, so we should expect gradients to be noisier
# than they "should" be when batch_size != dataset_size.
    return gradients[-1]

def validate_epoch(
    model,
    data_loader,
    raw_gradients,
    loss_function=metrics.cross_entropy_loss,
    metric_function=metrics.calculate_accuracy,
    epoch=None
):
    losses, accuracies = [], []

    for x_batch, y_batch in data_loader:
        y_preds = model(x_batch)
        losses += [loss_function(y_preds, y_batch).item()]
        accuracies += [metric_function(y_preds, y_batch).item()]

    print(
        '{0}Loss: {1}\t Accuracy: {2}\t Gradients: {3}'.format(
            'Epoch: {}\t '.format(epoch) or '',
            sum(losses) / len(losses),
            sum(accuracies) / len(accuracies),
            metrics.gradient_2_norm(raw_gradients)
        )
    )

    return (
        sum(losses) / len(losses),
        sum(accuracies) / len(accuracies),
        metrics.gradient_2_norm(raw_gradients)
    )

def train_model(
    model,
    epochs,
    learning_rate,
    data_loader,
    loss_function=metrics.cross_entropy_loss,
    optimizer=Optimizer,
    metric_function=metrics.calculate_accuracy,
    train_epoch=train_epoch,
    compute_metrics_every=1
):

    lr_schedule = learning_rate if (type(learning_rate) == list) else [learning_rate]*epochs

    if len(lr_schedule) != epochs:
        raise Exception('Custom learning rate schedules must have the same size as the number of epochs.')

    losses, gradients = [], []

    for i, lr in zip(range(epochs), lr_schedule):
        raw_gradients = train_epoch(
            model,
            data_loader=data_loader,
            optimizer=Optimizer(model.parameters(), lr),
            loss_function=loss_function
        )

        if i % compute_metrics_every == 0:
            loss, accuracies, gradient = validate_epoch(
                model,
                data_loader,
                raw_gradients,
                loss_function=loss_function,
                metric_function=metric_function,
                epoch=i
            )
            losses += [loss]
            gradients += [gradient]

        else:
            print('Epoch: {}'.format(i))

    return (
        losses,
        accuracies,
        gradients
    )

def run_one_experiment(
    model=None,
    learning_rate=None,
    dataset_size=None,
    batch_size=None,
    epochs=None,
    loss_function=metrics.cross_entropy_loss,
    parameters_seed=None,
    data_path=data.DATA_PATH,
    randomize_labels_seed=None,
    randomize_dataset_seed=None,
    whiten_images=True,
    classification_labels=data.CLASSIFICATION_LABELS,
    crop_size=28,
    compute_metrics_every=1,
    experiment_nickname='',
    beep=True,
    plot=True,
    save=True
):
    if parameters_seed is not None:
        utils.make_parameters_deterministic_(model, parameters_seed)

    original_model = cp.deepcopy(model)

    dataset = data.transform_dataset(
        data.build_dataset(
            dataset_size=dataset_size,
            classification_labels=classification_labels,
            data_path=data_path,
            randomize_dataset_seed=randomize_dataset_seed,
            crop_size=crop_size
        ),
        randomize_labels_seed=randomize_labels_seed,
        whiten_images=whiten_images
    )

    losses, accuracies, gradients = train_model(
        model,
        epochs,
        learning_rate,
        data.build_data_loader(dataset, batch_size),
        loss_function=loss_function,
        compute_metrics_every=compute_metrics_every
    )

    experiment = {
        'inputs': {
            'model': original_model,
            'learning_rate': learning_rate,
            'dataset_size': dataset_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'loss_function': loss_function,
            'parameters_seed': parameters_seed,
            'data_path': data_path,
            'randomize_labels_seed': randomize_labels_seed,
            'whiten_images': whiten_images,
            'classification_labels': classification_labels,
            'crop_size': crop_size
        },
        'outputs': {
            'model': model,
            'losses': losses,
            'accuracies': accuracies,
            'gradients': gradients
        }
    }

    if save:
        experiment_name = data.save_experiment(experiment, experiment_nickname)
        print()
        print(experiment_name)

    if beep:
        print('\a')

    if plot:
        viz.plot_learning_curves(losses, gradients)

    return experiment
