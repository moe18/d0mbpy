import pytest
from d0mbpy.d0mbpy import LinAlg


# Helper function to compare matrices with a given tolerance
def assert_matrices_equal(mat1, mat2, tol=1e-6):
    for i in range(len(mat1.data)):
        for j in range(len(mat1.data[0])):
            assert abs(mat1.data[i][j] - mat2.data[i][j]) < tol

# Test case for matrix addition
def test_matrix_addition():
    matrix1 = LinAlg([[1, 2], [3, 4]])
    matrix2 = LinAlg([[5, 6], [7, 8]])
    result = matrix1 + matrix2

    expected_result = LinAlg([[6, 8], [10, 12]])
    assert_matrices_equal(result, expected_result)

# Test case for matrix multiplication
def test_matrix_multiplication():
    matrix1 = LinAlg([[1, 2], [3, 4]])
    matrix2 = LinAlg([[5, 6], [7, 8]])
    result = matrix1 * matrix2

    expected_result = LinAlg([[19, 22], [43, 50]])
    assert_matrices_equal(result, expected_result)

# Test case for shape method
def test_shape_method():
    matrix = LinAlg([[1, 2, 3], [4, 5, 6]])
    result = matrix.shape()

    assert result == (2, 3)

# Test case for transpose method
def test_transpose_method():
    matrix = LinAlg([[1, 2, 3], [4, 5, 6]])
    result = matrix.transpose()

    expected_result = LinAlg([[1, 4], [2, 5], [3, 6]])
    assert_matrices_equal(result, expected_result)

# Test case for matrix multiplication with a vector
def test_matrix_vector_multiplication():
    matrix = LinAlg([[1, 2, 3], [4, 5, 6]])
    vector = LinAlg([[2], [3], [4]])
    result = matrix * vector

    expected_result = LinAlg([[20], [47]])
    assert_matrices_equal(result, expected_result)
