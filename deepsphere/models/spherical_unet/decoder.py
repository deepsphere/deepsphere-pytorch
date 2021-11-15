"""Decoder for Spherical UNet.
"""
# pylint: disable=W0221

import torch
from torch import nn

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBNPoolCheb(nn.Module):
    """Building Block calling a SphericalChebBNPool block then a SphericalCheb.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, middle_channels, lap, pooling, kernel_size)
        self.spherical_cheb = SphericalChebConv(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        x = self.spherical_cheb(x)
        return x


class SphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb_bn_pool = SphericalChebBNPool(in_channels, out_channels, lap, pooling, kernel_size)
        self.spherical_cheb_bn = SphericalChebBN(in_channels + out_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_pool(x)
        # pylint: disable=E1101
        x = torch.cat((x, concat_data), dim=2)
        # pylint: enable=E1101
        x = self.spherical_cheb_bn(x)
        return x


class Decoder(nn.Module):
    """The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.dec_l1 = SphericalChebBNPoolConcat(512, 512, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = SphericalChebBNPoolConcat(512, 256, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = SphericalChebBNPoolConcat(256, 128, laps[3], self.unpooling, self.kernel_size)
        self.dec_l4 = SphericalChebBNPoolConcat(128, 64, laps[4], self.unpooling, self.kernel_size)
        self.dec_l5 = SphericalChebBNPoolCheb(64, 32, 3, laps[5], self.unpooling, self.kernel_size)
        # Switch from Logits to Probabilities if evaluating model
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3, x_enc4):
        """Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        x = self.dec_l4(x, x_enc4)
        x = self.dec_l5(x)
        if not self.training:
            x = self.softmax(x)
        return x
