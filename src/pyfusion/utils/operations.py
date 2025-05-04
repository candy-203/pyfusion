import numpy as np

from pyfusion.structs import math as math


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


def structure_tensor(image: math.Image) -> math.TensorField:
    """
    Calculate the structure tensor of a 2D image.

    Args:
        image (structs.Image): The input image.

    Returns:
        structs.MatrixField: The structure tensor of the image.
    """
    grad = nabla(image)
    tensor = grad.data[..., :, None] @ grad.data[..., None, :]  # v @ v^T of each vector
    return math.TensorField(data=tensor)
