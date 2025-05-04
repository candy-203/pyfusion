# type: ignore

import os

import nibabel as nib

from pyfusion.structs.formats import FSL


def load_fsl(dir: str, slice: int) -> FSL:
    """
    Load FSL data from a directory and extract a specific slice. See https://fsl.fmrib.ox.ac.uk/fsl/docs/#/diffusion/dtifit for more information.
    
    Args:
        dir (str): Directory containing the FSL data files. 3D data files should be named "FA.nii.gz", "L1.nii.gz", "L2.nii.gz", "L3.nii.gz", "V1.nii.gz", "V2.nii.gz", and "V3.nii.gz".
        slice (int): Slice index to extract from the 3D data.
    
    Returns:
        FSL: An instance of the FSL class containing the loaded data.
    
    Raises:
        ValueError: If the slice index is out of bounds or if the directory does not contain the required files.
    """

    # Check if the directory exists
    if not os.path.isdir(dir):
        raise ValueError(f"Directory {dir} does not exist")
    if slice < 0:
        raise ValueError("Slice index must be non-negative")

    FA = nib.load(os.path.join(dir, "FA.nii.gz")).get_fdata()
    L1 = nib.load(os.path.join(dir, "L1.nii.gz")).get_fdata()
    L2 = nib.load(os.path.join(dir, "L2.nii.gz")).get_fdata()
    L3 = nib.load(os.path.join(dir, "L3.nii.gz")).get_fdata()
    V1 = nib.load(os.path.join(dir, "V1.nii.gz")).get_fdata()
    V2 = nib.load(os.path.join(dir, "V2.nii.gz")).get_fdata()
    V3 = nib.load(os.path.join(dir, "V3.nii.gz")).get_fdata()

    # Check if the slice index is within the bounds of the data
    if slice >= FA.shape[2]:
        raise ValueError(f"Slice index {slice} is out of bounds for the data with shape {FA.shape}")
    # Extract the specified slice from each 3D array
    # Note: The slice index is assumed to be in the third dimension (z-axis)
    # and the data is assumed to be in the format (x, y, z)
    # Adjust the slicing to match the data format

    FA = FA[:, :, slice]
    L1 = L1[:, :, slice]
    L2 = L2[:, :, slice]
    L3 = L3[:, :, slice]
    V1 = V1[:, :, slice, :]
    V2 = V2[:, :, slice, :]
    V3 = V3[:, :, slice, :]

    return FSL(
        FA=FA,
        L1=L1,
        L2=L2,
        L3=L3,
        V1=V1,
        V2=V2,
        V3=V3,
    )
