from pyfusion.structs import math
from pyfusion.utils import operations


def step(image: math.Image, tensor_field: math.TensorField, step_size: float) -> math.Image:
    """
    Perform a single step of diffusion using a tensor field.

    Args:
        image (structs.Image): The input image.
        dmri (structs.TensorField): The diffusion tensor field.
        step_size (float): The step size for the diffusion.
        sigma (float): The standard deviation for Gaussian smoothing.

    Returns:
        structs.Image: The resulting image after diffusion.
    """
    # Calculate the gradient of the smoothed image
    grad = operations.nabla(image)

    # Apply the diffusion tensor to the gradient
    grad_tensor = operations.apply_tensor_field(tensor_field, grad)

    # Calculate the divergence of the resulting vector field
    div = operations.divergence(grad_tensor)
    div.data *= step_size
    div.data += image.data
    return math.Image(data=div.data)
