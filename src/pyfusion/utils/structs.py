from functools import partial
from typing import Annotated, Self

import numpy as np
import numpy.typing as npt
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

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


### Main classes


class Image(NPBaseModel):
    """
    A class representing an image with a 2D numpy array.
    """

    data: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]


class VectorField(NPBaseModel):
    """
    A class representing a vector field with a 3D numpy array.
    """

    data: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 2]))]


class Eigen(NPBaseModel):
    """
    A class representing the i-th eigenvalues and eigenvectors of a 2x2 matrix field.
    """

    eigvec: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 2]))]
    eigval: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        check_dimensions(2, self.eigvec, self.eigval)
        return self


class EigenDecomp(NPBaseModel):
    """
    A class representing the eigen decomposition of a 2x2 matrix field.
    """
    eigen_1: Eigen
    eigen_2: Eigen

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        check_dimensions(None, self.eigen_1.eigvec, self.eigen_2.eigvec)
        return self


class Eigen3D(NPBaseModel):
    """
    A class representing the i-th eigenvalues and eigenvectors of a 3x3 matrix field.
    """

    eigvec: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 3]))]
    eigval: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        check_dimensions(2, self.eigvec, self.eigval)

        return self


class EigenDecomp3D(NPBaseModel):
    """
    A class representing the eigen decomposition of a 3x3 matrix field.
    """
    eigen_1: Eigen3D
    eigen_2: Eigen3D
    eigen_3: Eigen3D

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        check_dimensions(None, self.eigen_1.eigvec, self.eigen_2.eigvec)
        check_dimensions(None, self.eigen_1.eigvec, self.eigen_3.eigvec)

        return self


class TensorField(NPBaseModel):
    """
    A class representing a matrix field with a 4D numpy array.
    """

    data: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 2, 2]))]


class Tensor(NPBaseModel):
    """
    A class representing a tensor, in this context a 2x2 matrix.
    """

    data: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [2, 2]))]
