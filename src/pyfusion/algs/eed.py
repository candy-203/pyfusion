from typing import Callable

import numpy as np

from pyfusion.structs import math
from pyfusion.utils import operations, projection


def step(image: math.Image, tensor_field: math.TensorField, step_size: float) -> math.Image:
    """
    Perform a single step of diffusion using a tensor field.

    Args:
        image (structs.Image): The input image.
        dmri (structs.TensorField): The diffusion tensor field.
        step_size (float): The step size for the diffusion.

    Returns:
        structs.Image: The resulting image after diffusion.
    """
    # Calculate the gradient of the image
    grad = operations.nabla(image)

    # Apply the diffusion tensor to the gradient
    grad_tensor = operations.apply_tensor_field(tensor_field, grad)

    # Calculate the divergence of the resulting vector field
    div = operations.divergence(grad_tensor)
    div.data *= step_size
    div.data += image.data
    return math.Image(data=div.data)


def apply_heat_conduction(
    tensor_field: math.EigenDecomp, heat_conduction: Callable[[math.FloatArr], math.FloatArr]
) -> math.TensorField:
    """
    Apply heat conduction to the eigenvalues of a tensor field.

    Args:
        tensor_field (structs.EigenDecomp): The tensor field.
        heat_conduction (Callable[[structs.FloatArr], structs.FloatArr]): The heat conduction function to apply.

    Returns:
        structs.TensorField: The resulting tensor field after applying heat conduction, as follows:
            \\lambda_1 = \\lambda_1 \\cdot \\text{heat_conduction}(\\mu_1)
            \\lambda_2 = 1
        The eigenvalues are modified according to the heat conduction function, while the eigenvectors remain unchanged.
    """
    ones = np.ones(tensor_field.eigen_2.eigval.shape)
    eigval_1 = heat_conduction(tensor_field.eigen_1.eigval)

    eigen_1 = math.Eigen(
        eigvec=tensor_field.eigen_1.eigvec,
        eigval=eigval_1,
    )
    eigen_2 = math.Eigen(
        eigvec=tensor_field.eigen_2.eigvec,
        eigval=ones,
    )
    eigen_decomp = math.EigenDecomp(eigen_1=eigen_1, eigen_2=eigen_2)

    return projection.calculate_eigen(eigen_decomp)
