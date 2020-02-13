"""Spherical Graph Convolutional Neural Network with UNet autoencoder architecture.
"""

# pylint: disable=W0221

import torch
from torch import nn

from deepsphere.layers.samplings.equiangular_pool_unpool import Equiangular
from deepsphere.layers.samplings.healpix_pool_unpool import Healpix
from deepsphere.layers.samplings.icosahedron_pool_unpool import Icosahedron
from deepsphere.models.spherical_unet.decoder import Decoder
from deepsphere.models.spherical_unet.encoder import Encoder, EncoderTemporalConv
from deepsphere.utils.laplacian_funcs import get_equiangular_laplacians, get_healpix_laplacians, get_icosahedron_laplacians


class SphericalUNet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        if pooling_class == "icosahedron":
            self.pooling_class = Icosahedron()
            self.laps = get_icosahedron_laplacians(N, depth, laplacian_type)
        elif pooling_class == "healpix":
            self.pooling_class = Healpix()
            self.laps = get_healpix_laplacians(N, depth, laplacian_type)
        elif pooling_class == "equiangular":
            self.pooling_class = Equiangular()
            self.laps = get_equiangular_laplacians(N, depth, self.ratio, laplacian_type)
        else:
            raise ValueError("Error: sampling method unknown. Please use icosahedron, healpix or equiangular.")

        self.encoder = Encoder(self.pooling_class.pooling, self.laps, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size)

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


class SphericalUNetTemporalLSTM(SphericalUNet):
    """Sphericall GCNN Autoencoder with LSTM.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, sequence_length, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            sequence_length (int): The number of images used per sample
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__(pooling_class, N, depth, laplacian_type, kernel_size, ratio)
        self.sequence_length = sequence_length
        n_pixels = self.laps[0].size(0)
        n_features = self.encoder.enc_l0.spherical_cheb.chebconv.in_channels
        self.lstm_l0 = nn.LSTM(input_size=n_pixels * n_features, hidden_size=n_pixels * n_features, batch_first=True)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        device = x.device
        encoders_l0 = []
        for idx in range(self.sequence_length):
            encoding = self.encoder(x[:, idx, :, :].squeeze(dim=1))
            encoders_l0.append(encoding[0].reshape(encoding[0].size(0), 1, -1))

        encoders_l0 = torch.cat(encoders_l0, axis=1).to(device)
        lstm_output_l0, _ = self.lstm_l0(encoders_l0)
        lstm_output_l0 = lstm_output_l0[:, -1, :].reshape(-1, encoding[0].size(1), encoding[0].size(2))

        output = self.decoder(lstm_output_l0, encoding[1], encoding[2], encoding[3], encoding[4])
        return output


class SphericalUNetTemporalConv(SphericalUNet):
    """Spherical GCNN Autoencoder with temporality by means of convolution over time.
    """

    def __init__(self, pooling_class, N, depth, laplacian_type, sequence_length, kernel_size, ratio=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            sequence_length (int): The number of images used per sample
            kernel_size (int): chebychev polynomial degree
            ratio (float): Parameter for equiangular sampling
        """
        super().__init__(pooling_class, N, depth, laplacian_type, kernel_size, ratio)
        self.sequence_length = sequence_length
        self.encoder = EncoderTemporalConv(self.pooling_class.pooling, self.laps, self.sequence_length, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.unpooling, self.laps, self.kernel_size)

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
