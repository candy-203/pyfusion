from pyfusion.utils import structs, operations


def step(image: structs.Image, dmri: structs.TensorField, step_size: float) -> structs.Image:
    """
    Perform a single step of diffusion using the diffusion tensor.

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
    grad_tensor = operations.apply_tensor_field(dmri, grad)

    # Calculate the divergence of the resulting vector field
    div = operations.divergence(grad_tensor)
    div.data *= step_size
    div.data += image.data
    return structs.Image(data=div.data)
