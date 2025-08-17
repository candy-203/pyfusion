from functools import partial
from typing import Annotated, Self

from pydantic import AfterValidator, model_validator
from pyfusion.structs.validation_utils import (
    FloatArr,
    NPBaseModel,
    check_dimensions,
    validate_float_array,
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
