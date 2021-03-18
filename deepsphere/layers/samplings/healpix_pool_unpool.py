"""Healpix Sampling's Pooling and Unpooling
The pooling divides the number of nsides by 2 each time.
This represents (in the term of classic pooling kernel sizes) a division (pooling) or multiplication (unpooling) of the number of pixels by 4.
The kernel size for all modules is hence fixed.

Sampling theory from:
*HEALPix — a Framework for High Resolution Discretization, and Fast Analysis of Data Distributed on the Sphere* by Gorski (doi: 10.1086/427976)

Figure 1 for relation number of sides and number of pixels and for unpooling using tile.
The area of the pixels are the same hence latitude and longitude of the resolution are the same.

The lowest resolution possible with the HEALPix base partitioning of the sphere surface into 12 equal sized pixels
See: https://healpix.jpl.nasa.gov/

:math:`N_{pixels} = 12 * N_{sides}^2`
Nsides is the number of divisions from the baseline of 12 equal sized pixels

"""
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=W0221


class HealpixMaxPool(nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self, return_indices=False):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch

        Args:
            x (:obj:`torch.tensor`):[batch x pixels x features]

        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [batch x pooled pixels x features] and indices of pooled pixels
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size)
        else:
            x = F.max_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output


class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch

        Arguments:
            x (:obj:`torch.tensor`): [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """Healpix Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by HealpixMaxPool

        Args:
            tuple(x (:obj:`torch.tensor`) : [batch x pixels x features]
            indices (int)): indices of pixels equiangular maxpooled previously

        Returns:
            [:obj:`torch.tensor`] -- [batch x unpooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.max_unpool1d(x, indices, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x


class HealpixAvgUnpool(nn.Module):
    """Healpix Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        self.kernel_size = 4
        super().__init__()

    def forward(self, x):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor

        Arguments:
            x (:obj:`torch.tensor`): [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x unpooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, scale_factor=self.kernel_size, mode="nearest")
        x = x.permute(0, 2, 1)
        return x


class Healpix:
    """Healpix class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average"):
        """Initialize healpix pooling and unpooling objects.

        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = HealpixMaxPool()
            self.__unpooling = HealpixMaxUnpool()
        else:
            self.__pooling = HealpixAvgPool()
            self.__unpooling = HealpixAvgUnpool()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling
