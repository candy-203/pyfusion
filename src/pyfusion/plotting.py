# type: ignore
import matplotlib.pyplot as plt
import numpy as np


def plot(data, figsize: tuple[int, int] = (10, 8), title: str | None = None):
    """Plot the given data.

    :param tuple[int, int] figsize: The figure size in inches, defaults to (10, 8)
    :param data: The data has to be a 2D array with gray values
    :param str | None title: The title to display, empty when None, defaults to None
    """
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.show()


def plot_side_by_side(
    data_a,
    data_b,
    figsize_a: tuple[int, int] = (10, 8),
    figsize_b: tuple[int, int] = (10, 8),
    title_a: str | None = None,
    title_b: str | None = None,
):
    """Plot two images side by side.

    :param tuple[int, int] figsize_a: The figure size for the first image, defaults to (10, 8)
    :param tuple[int, int] figsize_b: The figure size for the second image, defaults to (10, 8)
    :param data_a: The first image data has to be a 2D array with gray values
    :param data_b: The second image data has to be a 2D array with gray values
    :param str | None title_a: The title to display for the first image, empty when None, defaults to None
    :param str | None title_b: The title to display for the second image, empty when None, defaults to None
    """
    _, axs = plt.subplots(1, 2, figsize=(figsize_a[0] + figsize_b[0], max(figsize_a[1], figsize_b[1])))
    im_a = axs[0].imshow(data_a, cmap="gray")
    im_b = axs[1].imshow(data_b, cmap="gray")
    if title_a:
        axs[0].set_title(title_a)
    if title_b:
        axs[1].set_title(title_b)
    axs[0].axis("off")
    axs[1].axis("off")
    plt.colorbar(im_a, ax=axs[0])
    plt.colorbar(im_b, ax=axs[1])
    plt.show()


def plot_with_vector_field(img, vector_field, figsize: tuple[int, int] = (10, 8), title: str | None = None):
    """Plot an image with a vectorfield overlaying it

    :param _type_ img: The image data to plot
    :param _type_ vector_field: The vector field data to overlay
    :param tuple[int, int] figsize: The figure size in inches, defaults to (10, 8)
    :param str | None title: The title to display, empty when None, defaults to None
    """    
    n, m = img.shape
    Y, X = np.mgrid[0:n, 0:m]

    U, V = vector_field[..., 0], vector_field[..., 1]

    # Compute direction (angle) and map to color
    angles = np.arctan2(V, U)
    angles_normalized = (angles + np.pi) / (2 * np.pi)
    color_map = plt.cm.hsv(angles_normalized)  # shape: (H, W, 4)
    
    # Flatten everything to 1D
    Xf = X.flatten()
    Yf = Y.flatten()
    Uf = U.flatten()
    Vf = V.flatten()
    Cf = color_map.reshape(-1, 4)  # RGBA

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None, origin='upper')
    plt.quiver(Xf, Yf, Uf, Vf, color=Cf, angles='xy', scale_units='xy', scale=1.0)
    plt.axis('off')
    plt.show()
