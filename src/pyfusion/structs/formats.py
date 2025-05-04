from functools import partial
from typing import Annotated, Self
from pydantic import AfterValidator, model_validator

from pyfusion.structs.validation_utils import FloatArr, NPBaseModel, check_dimensions, validate_float_array


class FSL(NPBaseModel):
    L1: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]
    L2: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]
    L3: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]
    V1: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 3]))]
    V2: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 3]))]
    V3: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None, 3]))]
    FA: Annotated[FloatArr, AfterValidator(partial(validate_float_array, [None, None]))]

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        # Check that the dimensions of L1, L2, L3, V1, V2, V3, and FA are consistent
        for attr in ["L1", "L2", "L3", "V1", "V2", "V3"]:
            check_dimensions(2, self.FA, getattr(self, attr))
        return self
