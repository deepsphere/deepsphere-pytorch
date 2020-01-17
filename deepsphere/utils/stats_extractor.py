"""Get Means and Standard deviations for all features of a dataset.
"""
import numpy as np
import torch


def stats_extractor(dataset):
    """Iterates over a dataset object
    It is iterated over so as to calculate the mean and standard deviation.

    Args:
        dataset (:obj:`torch.utils.data.dataloader`): dataset object to iterate over

    Returns:
        :obj:numpy.array, :obj:numpy.array : computed means and standard deviation
    """

    F, V = torch.Tensor(dataset[0][0]).shape
    summing = torch.zeros(F)
    square_summing = torch.zeros(F)
    total = 0

    for item in dataset:
        item = torch.Tensor(item[0])
        summing += torch.sum(item, dim=1)
        total += V

    means = torch.unsqueeze(summing / total, dim=1)

    for item in dataset:
        item = torch.Tensor(item[0])
        square_summing += torch.sum((item - means) ** 2, dim=1)

    stds = np.sqrt(square_summing / (total - 1))

    return torch.squeeze(means, dim=1).numpy(), stds.numpy()
