"""Initializing device
"""


import torch
from torch import nn
from torchvision import transforms

from deepsphere.data.datasets.dataset import ARTCTemporaldataset
from deepsphere.data.transforms.transforms import Stack
from deepsphere.models.spherical_unet.unet_model import SphericalUNetTemporalConv, SphericalUNetTemporalLSTM


def init_device(device, unet):
    """Initialize device based on cpu/gpu and number of gpu

    Args:
        device (str): cpu or gpu
        ids (list of int or str): list of gpus that should be used
        unet (torch.Module): the model to place on the device(s)

    Raises:
        Exception: There is an error in configuring the cpu or gpu

    Returns:
        torch.Module, torch.device: the model placed on device, the device
    """
    if device is None:
        device = torch.device("cpu")
        unet = unet.to(device)
    elif len(device) == 0:
        device = torch.device("cuda")
        unet = unet.to(device)
        unet = nn.DataParallel(unet)
    elif len(device) == 1:
        device = torch.device("cuda:{}".format(device[0]))
        unet = unet.to(device)
    elif len(device) > 1:
        ids = device
        device = torch.device("cuda:{}".format(ids[0]))
        unet = unet.to(device)
        unet = nn.DataParallel(unet, device_ids=[int(i) for i in ids])
    else:
        raise Exception("Device set up impossible.")

    return unet, device


def init_unet_temp(parser):
    """Initialize UNet

    Args:
        parser (dict): parser arguments

    Returns:
        unet: the model
    """
    pooling_class = parser.pooling_class
    n_pixels = parser.n_pixels
    depth = parser.depth
    laplacian_type = parser.laplacian_type
    sequence_length = parser.sequence_length
    kernel_size = parser.kernel_size
    if parser.type == "LSTM":
        unet = SphericalUNetTemporalLSTM(pooling_class, n_pixels, depth, laplacian_type, sequence_length, kernel_size)
    elif parser.type == "conv":
        unet = SphericalUNetTemporalConv(pooling_class, n_pixels, depth, laplacian_type, sequence_length, kernel_size)
    else:
        raise Exception("The first element after --temp must be either 'LSTM' or 'conv' to specify the type.")
    return unet


def init_dataset_temp(parser, indices, transform_image, transform_labels):
    """Initialize the dataset

    Args:
        parser (dict): parser arguments
        indices (list): The list of indices we want included in the dataset
        transform_image (list): The list of torchvision transforms we want to apply to the images
        transform_labels (list): The list of torchvision transforms we want to apply to the labels

    Returns:
        dataset: the dataset
    """
    path_to_data = parser.path_to_data
    download = parser.download
    if parser.type == "LSTM":
        transform_sample = transforms.Compose([Stack()])
    elif parser.type == "conv":
        transform_sample = transforms.Compose([transforms.Lambda(lambda item: torch.stack(item, dim=1).reshape(item[0].size(0), -1))])
    else:
        raise Exception("Invalid temporality type.")
    dataset = ARTCTemporaldataset(
        path=path_to_data,
        download=download,
        sequence_length=parser.sequence_length,
        prediction_shift=parser.prediction_shift,
        indices=indices,
        transform_image=transform_image,
        transform_labels=transform_labels,
        transform_sample=transform_sample,
    )
    return dataset
