"""Spherical Graph Convolutional Neural Network with UNet autoencoder architecture.
"""

# pylint: disable=W0221


from torch import nn

from deepsphere.layers.samplings.equiangular_pool_unpool import Equiangular
from deepsphere.layers.samplings.healpix_pool_unpool import Healpix
from deepsphere.layers.samplings.icosahedron_pool_unpool import Icosahedron
from deepsphere.models.spherical_unet.decoder import Decoder
from deepsphere.models.spherical_unet.encoder import Encoder
from deepsphere.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians


class SphericalUNet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__()
        self.pooling_class = pooling_class
        self.ratio = ratio

        if isinstance(self.pooling_class, Icosahedron):
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif isinstance(self.pooling_class, Healpix):
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif isinstance(self.pooling_class, Equiangular):
            self.laps = get_equiangular_laplacians(N, depth, self.ratio, laplacian_type)
        else:
            raise ValueError("Error: sampling method unknown. Please use icosahedron, healpix or equiangular.")

        self.encoder = Encoder(self.pooling_class.pooling, self.laps)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output
