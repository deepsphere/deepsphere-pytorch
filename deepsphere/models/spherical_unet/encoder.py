"""Encoder for Spherical UNet.
"""
# pylint: disable=W0221
from torch import nn

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.models.spherical_unet.utils import SphericalChebBN, SphericalChebBNPool


class SphericalChebBN2(nn.Module):
    """Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size=3):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """

        super().__init__()

        self.spherical_cheb_bn_1 = SphericalChebBN(in_channels, middle_channels, lap, kernel_size)
        self.spherical_cheb_bn_2 = SphericalChebBN(middle_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x


class SphericalChebPool(nn.Module):
    """Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size=3):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        return x


class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.pooling = pooling
        self.enc_l5 = SphericalChebBN2(16, 32, 64, laps[5])
        self.enc_l4 = SphericalChebBNPool(64, 128, laps[4], self.pooling)
        self.enc_l3 = SphericalChebBNPool(128, 256, laps[3], self.pooling)
        self.enc_l2 = SphericalChebBNPool(256, 512, laps[2], self.pooling)
        self.enc_l1 = SphericalChebBNPool(512, 512, laps[1], self.pooling)
        self.enc_l0 = SphericalChebPool(512, 512, laps[0], self.pooling)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x vertices x channels/features]
        """
        x_enc5 = self.enc_l5(x)
        x_enc4 = self.enc_l4(x_enc5)
        x_enc3 = self.enc_l3(x_enc4)
        x_enc2 = self.enc_l2(x_enc3)
        x_enc1 = self.enc_l1(x_enc2)
        x_enc0 = self.enc_l0(x_enc1)

        return x_enc0, x_enc1, x_enc2, x_enc3, x_enc4
