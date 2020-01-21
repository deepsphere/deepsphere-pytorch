"""
EquiAngular Sampling's Pooling and Unpooling.
The pooling goes down two bandwidths at a time.
This represents (in the term of classic pooling kernel sizes) a division (pooling) or multiplication (unpooling) of the number of pixels by 4.
The kernel size for all modules is henced fixed.

Equiangular sampling theory from:
*FFTs for the 2-Sphere:Improvements and Variations* by Healy (doi=10.1.1.51.5335)

Bandwidth : int or list or tuple. Hence we have a symetric or asymetric sampling. It corresponds to the resolution of the sampling scheme.
:math:`pixels = (2*bw)^{2}`
Allowed number of pixels:

- (bw=1) 4 pixels,
- (bw=2) 16 pixels,
- (bw=3) 36 pixels,
- (bw=4) 64 pixels,
- (bw=5) 100 pixels.

If latitude bandwidth is different from longitude bandwidth then we have:
:math:`pixels = ((2*bw_{latitude})**2)*((2*bw_{longitude})**2)`
"""

# pylint: disable=W0221

import torch.nn as nn
import torch.nn.functional as F

from deepsphere.utils.samplings import equiangular_calculator


def reformat(x):
    """Reformat the input from a 4D tensor to a 3D tensor

    Args:
        x (:obj:`torch.tensor`): a 4D tensor
    Returns:
        :obj:`torch.tensor`: a 3D tensor
    """
    x = x.permute(0, 2, 3, 1)
    N, D1, D2, Feat = x.size()
    x = x.view(N, D1 * D2, Feat)
    return x


class EquiangularMaxPool(nn.MaxPool1d):
    """EquiAngular Maxpooling module using MaxPool 1d from torch
    """

    def __init__(self, ratio, return_indices=False):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """calls Maxpool1d and if desired, keeps indices of the pixels pooled to unpool them

        Args:
            input (:obj:`torch.tensor`): batch x pixels x features

        Returns:
            tuple(:obj:`torch.tensor`, list(int)): batch x pooled pixels x features and the indices of the pixels pooled
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)

        if self.return_indices:
            x, indices = F.max_pool2d(x, self.kernel_size, return_indices=self.return_indices)
        else:
            x = F.max_pool2d(x, self.kernel_size)
        x = reformat(x)

        if self.return_indices:
            output = x, indices
        else:
            output = x

        return output


class EquiangularAvgPool(nn.AvgPool1d):
    """EquiAngular Average Pooling using Average Pooling 1d from pytorch
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4)

    def forward(self, x):
        """calls Avgpool1d

        Args:
            x (:obj:`torch.tensor`): batch x pixels x features

        Returns:
            :obj:`torch.tensor` -- batch x pooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.kernel_size)
        x = reformat(x)

        return x


class EquiangularMaxUnpool(nn.MaxUnpool1d):
    """Equiangular Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by EquiAngMaxPool

        Args:
            x (:obj:`torch.tensor`): batch x pixels x features
            indices (int): indices of pixels equiangular maxpooled previously

        Returns:
            :obj:`torch.tensor`: batch x unpooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.max_unpool2d(x, indices, kernel_size=(4, 4))
        x = reformat(x)
        return x


class EquiangularAvgUnpool(nn.Module):
    """EquiAngular Average Unpooling version 1 using the interpolate function when unpooling
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        self.kernel_size = 4
        super().__init__()

    def forward(self, x):
        """calls pytorch's interpolate function to create the values while unpooling based on the nearby values
        Args:
            x (:obj:`torch.tensor`): batch x pixels x features
        Returns:
            :obj:`torch.tensor`: batch x unpooled pixels x features
        """

        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.kernel_size), mode="nearest")
        x = reformat(x)
        return x


class Equiangular:
    """Equiangular class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, ratio=1, mode="average"):
        """Initialize equiangular pooling and unpooling objects.

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = EquiangularMaxPool(ratio)
            self.__unpooling = EquiangularMaxUnpool(ratio)
        else:
            self.__pooling = EquiangularAvgPool(ratio)
            self.__unpooling = EquiangularAvgUnpool(ratio)

    @property
    def pooling(self):
        """Getter for the pooling class
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Getter for the unpooling class
        """
        return self.__unpooling
