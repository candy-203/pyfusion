import numpy as np
import pytest

from pyfusion.structs import math
from pyfusion.utils import projection


@pytest.fixture
def eigen_decomp() -> math.EigenDecomp:
    eigvec_1 = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    eigval_1 = np.array([[1.0, 2.0]])
    eigvec_2 = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    eigval_2 = np.array([[3.0, 4.0]])

    return math.EigenDecomp(
        eigen_1=math.Eigen(eigvec=eigvec_1, eigval=eigval_1),
        eigen_2=math.Eigen(eigvec=eigvec_2, eigval=eigval_2),
    )


@pytest.fixture
def eigen_decomp_3d() -> math.EigenDecomp3D:
    eigvec_1 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    eigval_1 = np.array([[1.0, 2.0]])
    eigvec_2 = np.array([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    eigval_2 = np.array([[3.0, 4.0]])
    eigvec_3 = np.array([[[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])
    eigval_3 = np.array([[1.0, 0.0]])

    return math.EigenDecomp3D(
        eigen_1=math.Eigen3D(eigvec=eigvec_1, eigval=eigval_1),
        eigen_2=math.Eigen3D(eigvec=eigvec_2, eigval=eigval_2),
        eigen_3=math.Eigen3D(eigvec=eigvec_3, eigval=eigval_3),
    )


def test_projection_calculate_eigen(eigen_decomp: math.EigenDecomp):
    tensor_field = projection.calculate_eigen(eigen_decomp)

    # Check the shape of the tensor field
    assert tensor_field.data.shape == (1, 2, 2, 2)

    # Check the values of the tensor field
    expected_tensor_field = np.array([[[1.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 2.0]]])
    np.testing.assert_array_almost_equal(tensor_field.data[0], expected_tensor_field)


def test_projection_calculate_eigen3d(eigen_decomp_3d: math.EigenDecomp3D):
    tensor_field = projection.calculate_eigen3d(eigen_decomp_3d)

    # Check the shape of the tensor field
    assert tensor_field.data.shape == (1, 2, 2, 2)

    # Check the values of the tensor field
    expected_tensor_field = np.array([[[2.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 2.0]]])
    np.testing.assert_array_almost_equal(tensor_field.data[0], expected_tensor_field)
