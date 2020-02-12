"""Chebyshev convolution layer. For the moment taking as-is from Michaël Defferrard's implementation. For v0.15 we will rewrite parts of this layer.
"""
# pylint: disable=W0221

import math

import torch
from torch import nn


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V, Fin = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials

    # transform to Chebyshev basis
    x0 = inputs.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    inputs = x0.unsqueeze(0)  # 1 x V x Fin*B

    if K > 0:
        x1 = torch.sparse.mm(laplacian, x0)  # V x Fin*B
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for _ in range(1, K - 1):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

    inputs = inputs.view([K, V, Fin, B])  # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    inputs = inputs.view([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)
    inputs = inputs.matmul(weight)  # B*V x Fout
    inputs = inputs.view([B, V, Fout])  # B x V x Fout

    return inputs


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, conv=cheb_conv):
        """Initialize the Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.

        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs


class SphericalChebConv(nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_size (int): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size)

    def state_dict(self, *args, **kwargs):
        """! WARNING !

        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith("laplacian"):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def forward(self, x):
        """Forward pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.chebconv(self.laplacian, x)
        return x
