import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

# define type Image as a 2D numpy array with float64 data type
type FloatArr = npt.NDArray[np.float64]


### Custom BaseClass


class NPBaseModel(BaseModel):
    """
    A custom base model that allows for arbitrary types in its configuration.
    This is useful for numpy arrays and other types that are not natively supported by Pydantic.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


### Helper functions


def validate_float_array(dim_list: list[None | int], arr: FloatArr) -> FloatArr:
    if arr.ndim != len(dim_list):
        raise ValueError(f"Array of {arr.ndim}D must be {len(dim_list)}D")
    for i, dim in enumerate(dim_list):
        if dim is not None and arr.shape[i] != dim:
            raise ValueError(f"Array dimension {i} must be {dim}, but got {arr.shape[i]}")
    return arr


def check_dimensions(max_dim: int | None, arr_1: FloatArr, arr_2: FloatArr) -> None:
    if arr_1.shape[:max_dim] != arr_2.shape[:max_dim]:
        raise ValueError(
            f"Arrays must have the same spatial dimensions. I received: {arr_1.shape[:2]} and {arr_2.shape[:2]}"
        )
