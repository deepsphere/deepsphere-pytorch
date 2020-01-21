"""Different samplings require various calculations.
The calculations present here are for equiangular, healpix, icosahedron samplings.
"""
import math


def equiangular_bandwidth(nodes):
    """Calculate the equiangular bandwidth based on input nodes

    Args:
        nodes (int): the number of nodes should be a power of 4

    Returns:
        int: the corresponding bandwidth
    """
    bw = math.sqrt(nodes) / 2
    return bw


def equiangular_dimension_unpack(nodes, ratio):
    """Calculate the two underlying dimensions
    from the total number of nodes

    Args:
        nodes (int): combined dimensions
        ratio (float): ratio between the two dimensions

    Returns:
        int, int: separated dimensions
    """
    dim1 = int((nodes / ratio) ** 0.5)
    dim2 = int((nodes * ratio) ** 0.5)
    return dim1, dim2


def equiangular_calculator(tensor, ratio):
    """From a 3D input tensor and a known ratio between the latitude
    dimension and longitude dimension of the data, reformat the 3D input
    into a 4D output while also obtaining the bandwidth.

    Args:
        tensor (:obj:`torch.tensor`): 3D input tensor
        ratio (float): the ratio between the latitude and longitude dimension of the data

    Returns:
        :obj:`torch.tensor`, int, int: 4D tensor, the bandwidths for lat. and long.
    """
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    bw_dim1 = equiangular_bandwidth(dim1)
    bw_dim2 = equiangular_bandwidth(dim2)
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor, [bw_dim1, bw_dim2]


def healpix_resolution_calculator(nodes):
    """Calculate the resolution of a healpix graph
    for a given number of nodes.

    Args:
        nodes (int): number of nodes in healpix sampling

    Returns:
        int: resolution for the matching healpix graph
    """
    resolution = int(math.sqrt(nodes / 12))
    return resolution


def icosahedron_order_calculator(nodes):
    """Calculate the order of a icosahedron graph
    for a given number of nodes.

    Args:
        nodes (int): number of nodes in icosahedron sampling

    Returns:
        int: order for the matching icosahedron graph
    """
    order = math.log((nodes - 2) / 10) / math.log(4)
    return order


def icosahedron_nodes_calculator(order):
    """Calculate the number of nodes
    corresponding to the order of an icosahedron graph

    Args:
        order (int): order of an icosahedron graph

    Returns:
        int: number of nodes in icosahedron sampling for that order
    """
    nodes = 10 * (4 ** order) + 2
    return nodes
