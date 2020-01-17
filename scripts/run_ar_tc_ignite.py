import argparse
import os

import numpy as np
import sklearn
import torch
from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.metrics import EpochMetric, Loss
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from deepsphere.data.datasets.dataset import ARTCDataset
from deepsphere.data.transforms.transforms import Normalize, Permute, ToTensor
from deepsphere.layers.samplings.icosahedron_pool_unpool import Icosahedron
from deepsphere.models.spherical_unet.unet_model import SphericalUNet
from deepsphere.utils.parser import create_parser, parse_config
from deepsphere.utils.stats_extractor import stats_extractor


def average_precision_compute_fn(y_pred, y_true):
    """Attached function to the custom ignite metric AveragePrecisionMultiLabel

    Args:
        y_pred (:obj:`torch.Tensor`): model predictions
        y_true (:obj:`torch.Tensor`): ground truths

    Raises:
        RuntimeError: Indicates that sklearn should be installed by the user.

    Returns:
        :obj:`numpy.array`: average precision vector.
                            Of the same length as the number of labels present in the data
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise RuntimeError("This metric requires sklearn to be installed.")

    ap = average_precision_score(y_true.numpy(), y_pred.numpy(), None)
    return ap


def validate_output_transform(x, y, y_pred):
    """A transform to format the output of the supervised evaluator before calculating the metric

    Args:
        x (:obj:`torch.Tensor`): the input to the model
        y (:obj:`torch.Tensor`): the output of the model
        y_pred (:obj:`torch.Tensor`): the ground truth labels

    Returns:
        (:obj:`torch.Tensor`, :obj:`torch.Tensor`): model predictions and ground truths reformatted
    """
    output = y_pred
    labels = y
    B, V, C = output.shape
    B_labels, V_labels, C_labels = labels.shape
    output = output.view(B * V, C)
    labels = labels.view(B_labels * V_labels, C_labels)
    return output, labels


def add_tensorboard(engine_train, optimizer, model, log_dir):
    """Creates an ignite logger object and adds training elements such as weight and gradient histograms

    Args:
        engine_train (:obj:`ignite.engine`): the train engine to attach to the logger
        optimizer (:obj:`torch.optim`): the model's optimizer
        model (:obj:`torch.nn.Module`): the model being trained
        log_dir (string): path to where tensorboard data should be saved
    """
    # Create a logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach(
        engine_train, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {"loss": loss}), event_name=Events.EPOCH_COMPLETED
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach(engine_train, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(engine_train, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(engine_train, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)

    tb_logger.close()


def get_dataloaders(parser_args):
    """Creates the datasets and the corresponding dataloaders

    Args:
        parser_args (dict): parsed arguments

    Returns:
        (:obj:`torch.utils.data.dataloader`, :obj:`torch.utils.data.dataloader`): train, validation dataloaders
    """

    path_to_data = parser_args.path_to_data
    partition = parser_args.partition
    seed = parser_args.seed
    means_path = parser_args.means_path
    stds_path = parser_args.stds_path

    data = ARTCDataset(path=path_to_data, download=parser_args.download)

    train_indices, temp = train_test_split(data.indices, train_size=partition[0], random_state=seed)
    val_indices, test_indices = train_test_split(temp, test_size=partition[2] / (partition[1] + partition[2]), random_state=seed)

    if (means_path is None) or (stds_path is None):
        transform_data_stats = transforms.Compose([ToTensor()])
        train_set_stats = ARTCDataset(path=path_to_data, indices=train_indices, transform_data=transform_data_stats)
        means, stds = stats_extractor(train_set_stats)
    else:
        try:
            means = np.load(means_path)
            stds = np.load(stds_path)
        except ValueError:
            print("No means or stds were provided. Or path names incorrect.")

    transform_data = transforms.Compose([ToTensor(), Permute(), Normalize(mean=means, std=stds)])
    transform_labels = transforms.Compose([ToTensor(), Permute()])
    train_set = ARTCDataset(path=path_to_data, indices=train_indices, transform_data=transform_data, transform_labels=transform_labels)
    validation_set = ARTCDataset(path=path_to_data, indices=val_indices, transform_data=transform_data, transform_labels=transform_labels)

    dataloader_train = DataLoader(train_set, batch_size=parser_args.batch_size, shuffle=True, num_workers=12)
    dataloader_validation = DataLoader(validation_set, batch_size=parser_args.batch_size, shuffle=False, num_workers=12)
    return dataloader_train, dataloader_validation


def main(parser_args):
    """Main function to create trainer engine, add handlers to train and validation engines.
    Then runs train engine to perform training and validation.

    Args:
        parser_args (dict): parsed arguments
    """

    device = torch.device(parser_args.device)

    dataloader_train, dataloader_validation = get_dataloaders(parser_args)
    criterion = nn.CrossEntropyLoss()

    unet = SphericalUNet(Icosahedron(), 10242, 6, "combinatorial")
    unet = unet.to(device)
    # unet = nn.DataParallel(unet)
    lr = parser_args.learning_rate
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    def trainer(engine, batch):
        """Train Function to define train engine.
        Called for every batch of the train engine, for each epoch.

        Args:
            engine (ignite.engine): train engine
            batch (:obj:`torch.utils.data.dataloader`): batch from train dataloader

        Returns:
            :obj:`torch.tensor` : train loss for that batch and epoch
        """
        unet.train()
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        output = unet(data)

        B, V, C = output.shape
        B_labels, V_labels, C_labels = labels.shape
        output = output.view(B * V, C)
        labels = labels.view(B_labels * V_labels, C_labels).max(1)[1]

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    writer = SummaryWriter(parser_args.tensorboard_path)

    engine_train = Engine(trainer)

    engine_validate = create_supervised_evaluator(
        model=unet, metrics={"AP": EpochMetric(average_precision_compute_fn)}, device=device, output_transform=validate_output_transform
    )

    engine_train.add_event_handler(Events.EPOCH_STARTED, lambda x: print("Starting Epoch: {}".format(x.state.epoch)))
    engine_train.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @engine_train.on(Events.EPOCH_COMPLETED)
    def epoch_validation(engine):
        """Handler to run the validation engine at the end of the train engine's epoch.

        Args:
            engine (ignite.engine): train engine
        """
        print("beginning validation epoch")
        engine_validate.run(dataloader_validation)

    reduce_lr_plateau = ReduceLROnPlateau(
        optimizer,
        mode=parser_args.reducelronplateau_mode,
        factor=parser_args.reducelronplateau_factor,
        patience=parser_args.reducelronplateau_patience,
    )

    @engine_validate.on(Events.EPOCH_COMPLETED)
    def update_reduce_on_plateau(engine):
        """Handler to reduce the learning rate on plateau at the end of the validation engine's epoch

        Args:
            engine (ignite.engine): validation engine
        """
        ap = engine.state.metrics["AP"]
        mAP = np.mean(ap[1 : len(ap)])
        reduce_lr_plateau.step(mAP)

    @engine_validate.on(Events.EPOCH_COMPLETED)
    def save_epoch_results(engine):
        """Handler to save the metrics at the end of the validation engine's epoch

        Args:
            engine (ignite.engine): validation engine
        """
        ap = engine.state.metrics["AP"]
        mAP = np.mean(ap[1 : len(ap)])
        print("Average precisions:", ap)
        print("mAP:", mAP)
        writer.add_scalars(
            "metrics", {"mean average precision (AR+TC)": mAP, "AR average precision": ap[1], "TC average precision": ap[2]}, engine_train.state.epoch
        )
        writer.close()

    step_scheduler = StepLR(optimizer, step_size=parser_args.steplr_step_size, gamma=parser_args.steplr_gamma)
    scheduler = create_lr_scheduler_with_warmup(
        step_scheduler,
        warmup_start_value=parser_args.warmuplr_warmup_start_value,
        warmup_end_value=parser_args.warmuplr_warmup_end_value,
        warmup_duration=parser_args.warmuplr_warmup_duration,
    )
    engine_validate.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    earlystopper = EarlyStopping(
        patience=parser_args.earlystopping_patience, score_function=lambda x: -x.state.metrics["AP"][1], trainer=engine_train
    )
    engine_validate.add_event_handler(Events.EPOCH_COMPLETED, earlystopper)

    add_tensorboard(engine_train, optimizer, unet, log_dir=parser_args.tensorboard_path)

    engine_train.run(dataloader_train, max_epochs=parser_args.n_epochs)

    torch.save(unet.state_dict(), parser_args.model_save_path + "unet_state.pt")


if __name__ == "__main__":
    # run with (for example):
    # python run_ar_tc_ignite.py --config-file config.example.yml --device cuda:2 --download False --path_to_data /data/climate/data_5_all
    parser_args = parse_config(create_parser())

    main(parser_args)
