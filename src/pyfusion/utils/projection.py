import numpy as np

from pyfusion.utils import structs


def calculate_eigen(eigen_decom: structs.EigenDecomp) -> structs.TensorField:
    """
    Calculate the tensor field from the eigen decomposition.

    Args:
        eigen_decom (structs.EigenDecomposition): The eigen decomposition.

    Returns:
        structs.TensorField: The resulting tensor field.
    """
    eigen_1 = eigen_decom.eigen_1
    eigen_2 = eigen_decom.eigen_2

    V = np.stack([eigen_1.eigvec, eigen_2.eigvec], axis=-1)

    Lambda = np.stack([eigen_1.eigval, eigen_2.eigval], axis=-1)

    V_scaled = V * Lambda[..., np.newaxis, :]

    tensor_field = np.matmul(V_scaled, np.swapaxes(V, -1, -2))

    return structs.TensorField(data=tensor_field)


def calculate_eigen3d(eigen_decom: structs.EigenDecomp3D) -> structs.TensorField:
    """
    Calculate the tensor field from the eigen decomposition. Projecting the 3D eigenvectors to x-y-plane.

    Args:
        eigen_decom (structs.EigenDecomp3D): The eigen decomposition.

    Returns:
        structs.TensorField: The resulting tensor field.
    """
    eigen_1 = eigen_decom.eigen_1
    eigen_2 = eigen_decom.eigen_2
    eigen_3 = eigen_decom.eigen_3

    V = np.stack([eigen_1.eigvec[..., :-1], eigen_2.eigvec[..., :-1], eigen_3.eigvec[..., :-1]], axis=-1)

    Lambda = np.stack([eigen_1.eigval, eigen_2.eigval, eigen_3.eigval], axis=-1)

    V_scaled = V * Lambda[..., np.newaxis, :]

    tensor_field = np.matmul(V_scaled, np.swapaxes(V, -1, -2))

    return structs.TensorField(data=tensor_field)
