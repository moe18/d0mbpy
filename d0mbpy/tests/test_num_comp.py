import pytest
from d0mbpy.d0mbpy import NumComp, LinAlg


# Helper function to compare matrices with a given tolerance
def assert_matrices_equal(mat1, mat2, tol=1e-6):
    for i in range(len(mat1.data)):
        for j in range(len(mat1.data[0])):
            assert abs(mat1.data[i][j] - mat2.data[i][j]) < tol


def assert_equation(val, val1, tol=1e-4):
    assert abs(val - val1)< tol


def test_softmax():
    a = NumComp()
    val = a.softmax(2,LinAlg([[1,2,3]]))

    expected_result = .244728
    print(val)
    assert_equation(val, expected_result)

def test_dir():
    def f(x):
        return x**2
    
    a = NumComp(f)

    assert_equation(a.dirivative(), 6)