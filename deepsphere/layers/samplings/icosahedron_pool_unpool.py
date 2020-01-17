"""Icosahedron Sampling's Pooling and Unpooling.
Each pooling takes down an order in the icosahedron.
Each unpooling adds the number of pixels corresponding to the next order.

Icosahedron is a polyhedron with 12 vertices and, 20 faces, where a regular icosahedron is a Platonic solid.
All faces are regular (equilateral) triangles.
This default Icosahedron can be considered at level 0, meaning that no further subdivision has occurred from the platonic solid.
See: https://github.com/maxjiang93/ugscnn/blob/master/meshcnn/mesh.py from Max Jiang
"""
# pylint: disable=W0221

import math

import torch.nn as nn
import torch.nn.functional as F


class IcosahedronPool(nn.Module):
    """Isocahedron Pooling, consists in keeping only a subset of the original pixels (considering the ordering of an isocahedron sampling method).
    """

    def forward(self, x):
        """Forward function calculates the subset of pixels to keep based on input size and the kernel_size.

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pixels pooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        pool_order = order - 1
        subset_pixels_keep = int(10 * math.pow(4, pool_order) + 2)
        return x[:, :subset_pixels_keep, :]


class IcosahedronUnpool(nn.Module):
    """Isocahedron Unpooling, consists in adding 1 values to match the desired un pooling size
    """

    def forward(self, x):
        """Forward calculates the subset of pixels that will result from the unpooling kernel_size and then adds 1 valued pixels to match this size

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x pixels unpooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        unpool_order = order + 1
        additional_pixels = int((10 * math.pow(4, unpool_order)) + 2)
        subset_pixels_add = additional_pixels - M
        return F.pad(x, (0, 0, 0, subset_pixels_add, 0, 0), "constant", value=1)


class Icosahedron:
    """Icosahedron class, which simply groups together the corresponding pooling and unpooling.
    """

    def __init__(self):
        """Initialize icosahedron pooling and unpooling objects.
        """
        self.__pooling = IcosahedronPool()
        self.__unpooling = IcosahedronUnpool()

    @property
    def pooling(self):
        """Get pooling.
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling.
        """
        return self.__unpooling
