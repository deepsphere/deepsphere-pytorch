"""Initializing device
"""

import torch
from torch import nn


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
