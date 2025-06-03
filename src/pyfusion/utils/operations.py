from typing import Callable
import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore

from pyfusion.structs import math as math
from pyfusion.utils.projection import calculate_eigen


def nabla(image: math.Image) -> math.VectorField:
    """
    Calculate the gradient of a 2D image using central differences.

    Args:
        bound_cond (structs.BoundCond): The boundary condition to use.
        image (structs.Image): The input image.

    Returns:
        structs.VectorField: The gradient of the image.
    """
    grad = np.gradient(image.data, edge_order=1)
    grad = np.stack(grad, axis=-1)
    return math.VectorField(data=grad)


def apply_tensor(tensor: math.Tensor, vector_field: math.VectorField) -> math.VectorField:
    """
    Apply a 2x2 tensor to a 2D vector field.

    Args:
        tensor (structs.Tensor): The tensor to apply.
        vector_field (structs.VectorField): The vector field to apply the tensor to.

    Returns:
        structs.VectorField: The resulting vector field.
    """
    res: math.FloatArr = np.tensordot(vector_field.data, tensor.data, axes=(1))  # type: ignore
    return math.VectorField(data=res)


def apply_tensor_field(tensor_field: math.TensorField, vector_field: math.VectorField) -> math.VectorField:
    """
    Apply a 2x2 tensor field to a 2D vector field.

    Args:
        tensor_field (structs.TensorField): The tensor field to apply.
        vector_field (structs.VectorField): The vector field to apply the tensor field to.

    Returns:
        structs.VectorField: The resulting vector field.
    """
    # ensure the tensor field and vector field have the same shape
    if tensor_field.data.shape[:2] != vector_field.data.shape[:2]:
        raise ValueError(
            f"Tensor field and vector field must have the same spatial dimensions. I recieived: {tensor_field.data.shape[:2]} and {vector_field.data.shape[:2]}"
        )
    res: math.FloatArr = np.matmul(tensor_field.data, vector_field.data[..., None])[..., 0]
    return math.VectorField(data=res)


def divergence(vector_field: math.VectorField) -> math.Image:
    """
    Calculate the divergence of a 2D vector field using central differences.

    Args:
        vector_field (structs.VectorField): The input vector field.

    Returns:
        structs.Image: The divergence of the vector field.
    """
    x_grad = np.gradient(vector_field.data[..., 0], axis=1)
    y_grad = np.gradient(vector_field.data[..., 1], axis=0)
    return math.Image(data=x_grad + y_grad)


def rotate(vector_field: math.VectorField) -> math.VectorField:
    """
    Rotate a 2D vector field by 90 degrees.

    Args:
        vector_field (structs.VectorField): The input vector field.

    Returns:
        structs.VectorField: The rotated vector field.
    """
    rotated_data = np.empty_like(vector_field.data)
    rotated_data[..., 0] = -vector_field.data[..., 1]
    rotated_data[..., 1] = vector_field.data[..., 0]
    return math.VectorField(data=rotated_data)


def structure_tensor(image: math.Image, heat_conduction: Callable[[math.FloatArr], math.FloatArr]) -> math.TensorField:
    """
    Calculate the structure tensor of a 2D image, with the integration scale of 0.
    **NOTE**: Presmoothing has to be done before.

    Args:
        image (structs.Image): The input image.

    Returns:
        structs.MatrixField: The structure tensor of the image.
    """
    grad = nabla(image)
    magnitudes = grad.data[..., 0] ** 2 + grad.data[..., 1] ** 2
    rotated_grad = rotate(grad)
    zeros = np.zeros_like(magnitudes)

    eigen_1 = math.Eigen(eigvec=grad.data, eigval=magnitudes)
    eigen_2 = math.Eigen(eigvec=rotated_grad.data, eigval=zeros)
    eigen_decomp = math.EigenDecomp(eigen_1=eigen_1, eigen_2=eigen_2)

    final_decomp = apply_heat_conduction(eigen_decomp, heat_conduction)

    return calculate_eigen(final_decomp)


def smooth(image: math.Image, sigma: float) -> math.Image:
    """
    Smooth a 2D image using a Gaussian filter.

    Args:
        image (structs.Image): The input image.
        sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        structs.Image: The smoothed image.
    """
    smoothed_data = gaussian_filter(image.data, sigma=sigma)  # type: ignore
    return math.Image(data=smoothed_data)  # type: ignore


def apply_heat_conduction(
    eigen_decomp: math.EigenDecomp, heat_conduction: Callable[[math.FloatArr], math.FloatArr]
) -> math.EigenDecomp:
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
    ones = np.ones(eigen_decomp.eigen_2.eigval.shape)
    eigval_1 = heat_conduction(eigen_decomp.eigen_1.eigval)

    eigen_1 = math.Eigen(
        eigvec=eigen_decomp.eigen_1.eigvec,
        eigval=eigval_1,
    )
    eigen_2 = math.Eigen(
        eigvec=eigen_decomp.eigen_2.eigvec,
        eigval=ones,
    )
    return math.EigenDecomp(eigen_1=eigen_1, eigen_2=eigen_2)

