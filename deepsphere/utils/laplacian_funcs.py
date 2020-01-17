"""Functions related to getting the laplacian and the right number of pixels after pooling/unpooling.
"""

import numpy as np
import torch
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from pygsp.graphs.nngraphs.sphereicosahedron import SphereIcosahedron
from pygsp.graphs.sphereequiangular import SphereEquiangular
from scipy import sparse
from scipy.sparse import coo_matrix

from deepsphere.utils.samplings import (
    equiangular_bandwidth,
    equiangular_dimension_unpack,
    healpix_resolution_calculator,
    icosahedron_nodes_calculator,
    icosahedron_order_calculator,
)


def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.

    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.

    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    """

    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def get_icosahedron_laplacians(nodes, depth, laplacian_type):
    """Get the icosahedron laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    order = icosahedron_order_calculator(nodes)
    for _ in range(depth):
        nodes = icosahedron_nodes_calculator(order)
        order_initial = icosahedron_order_calculator(nodes)
        G = SphereIcosahedron(level=int(order_initial))
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
        order -= 1
    return laps[::-1]


def get_healpix_laplacians(nodes, depth, laplacian_type):
    """Get the healpix laplacian list for a certain depth.

    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    pixel_num = nodes
    for i in range(depth):
        pixel_num = int(pixel_num / (4 ** i))
        resolution = healpix_resolution_calculator(pixel_num)
        G = SphereHealpix(Nside=resolution)
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]


def get_equiangular_laplacians(nodes, depth, ratio, laplacian_type):
    """Get the equiangular laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.

    Returns:
        laps (list): increasing list of laplacians
    """
    laps = []
    pixel_num = nodes
    for _ in range(depth):
        dim1, dim2 = equiangular_dimension_unpack(pixel_num, ratio)
        bw1 = equiangular_bandwidth(dim1)
        bw2 = equiangular_bandwidth(dim2)
        bw = [bw1, bw2]
        G = SphereEquiangular(bandwidth=bw, sampling="SOFT")
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]
