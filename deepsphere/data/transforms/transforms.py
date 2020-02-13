"""Transformations for samples of atmospheric rivers and tropical cyclones dataset.
"""
import torch


class ToTensor:
    """Convert raw data and labels to PyTorch tensor.
    """

    def __call__(self, item):
        """Function call operator to change type.

        Args:
            item (:obj:`numpy.array`): Numpy array that needs to be transformed.
        Returns:
            :obj:`torch.Tensor`: Sample of size (vertices, features).
        """
        return torch.Tensor(item)


class Permute:
    """Permute first and second dimension.
    """

    def __call__(self, item):
        """Permute first and second dimension.

        Args:
            item (:obj:`torch.Tensor`): Torch tensor that needs to be transformed.

        Returns:
            :obj:`torch.Tensor`: Permuted input tensor.
        """
        return item.permute(1, 0)


class Normalize:
    """Normalize using mean and std.
    """

    def __init__(self, mean, std):
        """Initialization

        Args:
            mean (:obj:`numpy.array`): means of each feature
            std (:obj:`numpy.array`): standard deviations of each feature
        """
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __call__(self, item):
        """
        Args:
            item (:obj:`torch.Tensor`): Sample of size (vertices, features) to be normalized on its features.

        Returns:
            :obj:`torch.Tensor`: Normalized input tensor.
        """
        return (item - self.mean) / self.std


class Stack:
    """Stack images in torch tensor.
    """

    def __init__(self, dimension=0):
        """Initialization

        Args:
            dimension int: The dimension to be used for stacking.
        """
        self.dimension = dimension

    def __call__(self, item):
        """Stack images in torch tensor.

        Args:
            item (:obj:`torch.Tensor`): Torch tensor that needs to be transformed.

        Returns:
            :obj:`torch.Tensor`: Stacked input tensor.
        """
        return torch.stack(item, dim=self.dimension)
