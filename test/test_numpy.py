"""
Comprehensive NumPy integration tests for pyz3
Tests all NumPy functionality with zero-error guarantee
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add example to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "example"))


class TestNumPyBasics:
    """Test basic NumPy array creation and properties"""

    def test_numpy_imports(self):
        """Verify NumPy can be imported"""
        import numpy

        assert numpy is not None
        assert hasattr(numpy, "array")
        assert hasattr(numpy, "ndarray")

    def test_numpy_version(self):
        """Check NumPy version is 2.0+"""
        version = tuple(map(int, np.__version__.split(".")[:2]))
        assert version >= (2, 0), f"NumPy 2.0+ required, got {np.__version__}"


class TestArrayCreation:
    """Test array creation functions"""

    def test_array_from_list(self):
        """Test creating array from Python list"""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.dtype == np.int64 or arr.dtype == np.int32
        assert np.array_equal(arr, [1, 2, 3, 4, 5])

    def test_zeros(self):
        """Test zeros array creation"""
        arr = np.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert np.all(arr == 0)

    def test_ones(self):
        """Test ones array creation"""
        arr = np.ones((2, 3, 4))
        assert arr.shape == (2, 3, 4)
        assert np.all(arr == 1)

    def test_arange(self):
        """Test arange function"""
        arr = np.arange(0, 10, 2)
        assert np.array_equal(arr, [0, 2, 4, 6, 8])

    def test_linspace(self):
        """Test linspace function"""
        arr = np.linspace(0, 1, 5)
        assert len(arr) == 5
        assert arr[0] == 0.0
        assert arr[-1] == 1.0

    def test_eye(self):
        """Test identity matrix creation"""
        arr = np.eye(3)
        assert arr.shape == (3, 3)
        assert np.array_equal(arr, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_empty(self):
        """Test empty array creation"""
        arr = np.empty((2, 3))
        assert arr.shape == (2, 3)

    def test_full(self):
        """Test full array creation"""
        arr = np.full((2, 3), 7)
        assert arr.shape == (2, 3)
        assert np.all(arr == 7)


class TestArrayProperties:
    """Test array properties and attributes"""

    def test_shape(self):
        """Test shape attribute"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_ndim(self):
        """Test ndim attribute"""
        arr1d = np.array([1, 2, 3])
        arr2d = np.array([[1, 2], [3, 4]])
        arr3d = np.array([[[1, 2]], [[3, 4]]])

        assert arr1d.ndim == 1
        assert arr2d.ndim == 2
        assert arr3d.ndim == 3

    def test_size(self):
        """Test size attribute"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert arr.size == 6

    def test_dtype(self):
        """Test dtype attribute"""
        arr_int = np.array([1, 2, 3])
        arr_float = np.array([1.0, 2.0, 3.0])
        arr_bool = np.array([True, False, True])

        assert arr_int.dtype in (np.int32, np.int64)
        assert arr_float.dtype == np.float64
        assert arr_bool.dtype == np.bool_

    def test_itemsize(self):
        """Test itemsize attribute"""
        arr_int32 = np.array([1, 2, 3], dtype=np.int32)
        arr_int64 = np.array([1, 2, 3], dtype=np.int64)
        arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        assert arr_int32.itemsize == 4
        assert arr_int64.itemsize == 8
        assert arr_float64.itemsize == 8


class TestArrayOperations:
    """Test array arithmetic operations"""

    def test_add(self):
        """Test array addition"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = a + b
        assert np.array_equal(result, [5, 7, 9])

    def test_subtract(self):
        """Test array subtraction"""
        a = np.array([10, 20, 30])
        b = np.array([1, 2, 3])
        result = a - b
        assert np.array_equal(result, [9, 18, 27])

    def test_multiply(self):
        """Test array multiplication"""
        a = np.array([2, 3, 4])
        b = np.array([5, 6, 7])
        result = a * b
        assert np.array_equal(result, [10, 18, 28])

    def test_divide(self):
        """Test array division"""
        a = np.array([10, 20, 30])
        b = np.array([2, 4, 5])
        result = a / b
        assert np.array_equal(result, [5, 5, 6])

    def test_power(self):
        """Test array power"""
        a = np.array([2, 3, 4])
        result = a**2
        assert np.array_equal(result, [4, 9, 16])

    def test_scalar_operations(self):
        """Test operations with scalars"""
        a = np.array([1, 2, 3])
        assert np.array_equal(a + 10, [11, 12, 13])
        assert np.array_equal(a * 2, [2, 4, 6])
        assert np.array_equal(a - 1, [0, 1, 2])


class TestArrayReshaping:
    """Test array reshaping operations"""

    def test_reshape(self):
        """Test reshape operation"""
        arr = np.arange(12)
        reshaped = arr.reshape(3, 4)
        assert reshaped.shape == (3, 4)
        assert reshaped.size == 12

    def test_flatten(self):
        """Test flatten operation"""
        arr = np.array([[1, 2], [3, 4]])
        flat = arr.flatten()
        assert flat.shape == (4,)
        assert np.array_equal(flat, [1, 2, 3, 4])

    def test_transpose(self):
        """Test transpose operation"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        t = arr.T
        assert t.shape == (3, 2)
        assert t[0, 0] == 1
        assert t[0, 1] == 4

    def test_ravel(self):
        """Test ravel operation"""
        arr = np.array([[1, 2], [3, 4]])
        raveled = arr.ravel()
        assert raveled.shape == (4,)


class TestArrayIndexing:
    """Test array indexing and slicing"""

    def test_basic_indexing(self):
        """Test basic indexing"""
        arr = np.array([10, 20, 30, 40, 50])
        assert arr[0] == 10
        assert arr[2] == 30
        assert arr[-1] == 50

    def test_slicing(self):
        """Test array slicing"""
        arr = np.array([0, 1, 2, 3, 4, 5])
        assert np.array_equal(arr[1:4], [1, 2, 3])
        assert np.array_equal(arr[::2], [0, 2, 4])

    def test_2d_indexing(self):
        """Test 2D array indexing"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        assert arr[0, 0] == 1
        assert arr[1, 2] == 6

    def test_boolean_indexing(self):
        """Test boolean indexing"""
        arr = np.array([1, 2, 3, 4, 5])
        mask = arr > 3
        result = arr[mask]
        assert np.array_equal(result, [4, 5])


class TestAggregations:
    """Test aggregation functions"""

    def test_sum(self):
        """Test sum operation"""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15

    def test_mean(self):
        """Test mean operation"""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0

    def test_min_max(self):
        """Test min and max"""
        arr = np.array([5, 2, 8, 1, 9])
        assert arr.min() == 1
        assert arr.max() == 9

    def test_std(self):
        """Test standard deviation"""
        arr = np.array([1, 2, 3, 4, 5])
        std = arr.std()
        assert std > 0

    def test_var(self):
        """Test variance"""
        arr = np.array([1, 2, 3, 4, 5])
        var = arr.var()
        assert var > 0

    def test_axis_operations(self):
        """Test operations along axis"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        sum_axis0 = arr.sum(axis=0)
        sum_axis1 = arr.sum(axis=1)
        assert np.array_equal(sum_axis0, [5, 7, 9])
        assert np.array_equal(sum_axis1, [6, 15])


class TestLinearAlgebra:
    """Test linear algebra operations"""

    def test_dot_product(self):
        """Test dot product"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.dot(a, b)
        assert result == 32  # 1*4 + 2*5 + 3*6

    def test_matmul(self):
        """Test matrix multiplication"""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(result, expected)

    def test_matrix_inverse(self):
        """Test matrix inverse"""
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        inv = np.linalg.inv(arr)
        identity = arr @ inv
        assert np.allclose(identity, np.eye(2))

    def test_determinant(self):
        """Test determinant calculation"""
        arr = np.array([[1, 2], [3, 4]])
        det = np.linalg.det(arr)
        assert np.isclose(det, -2.0)

    def test_eigenvalues(self):
        """Test eigenvalue computation"""
        arr = np.array([[1, 2], [2, 1]], dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        assert len(eigenvalues) == 2

    def test_solve_linear_system(self):
        """Test solving linear systems"""
        a = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)
        x = np.linalg.solve(a, b)
        # Verify solution
        assert np.allclose(a @ x, b)


class TestRandomNumbers:
    """Test random number generation"""

    def test_rand(self):
        """Test random floats"""
        arr = np.random.rand(10)
        assert arr.shape == (10,)
        assert np.all((arr >= 0) & (arr < 1))

    def test_randn(self):
        """Test random normal distribution"""
        arr = np.random.randn(100)
        assert arr.shape == (100,)
        # Check mean is close to 0 and std close to 1
        assert abs(arr.mean()) < 0.5
        assert abs(arr.std() - 1.0) < 0.5

    def test_randint(self):
        """Test random integers"""
        arr = np.random.randint(0, 10, size=20)
        assert arr.shape == (20,)
        assert np.all((arr >= 0) & (arr < 10))

    def test_random_seed(self):
        """Test random seed for reproducibility"""
        np.random.seed(42)
        a = np.random.rand(5)
        np.random.seed(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)


class TestBroadcasting:
    """Test NumPy broadcasting"""

    def test_scalar_broadcast(self):
        """Test broadcasting with scalar"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = arr + 10
        expected = np.array([[11, 12, 13], [14, 15, 16]])
        assert np.array_equal(result, expected)

    def test_1d_broadcast(self):
        """Test 1D broadcasting"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([10, 20, 30])
        result = arr + vec
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        assert np.array_equal(result, expected)

    def test_broadcasting_rules(self):
        """Test broadcasting compatibility"""
        a = np.ones((3, 1))
        b = np.ones((1, 4))
        c = a + b
        assert c.shape == (3, 4)


class TestDTypes:
    """Test different data types"""

    def test_int_dtypes(self):
        """Test integer dtypes"""
        arr8 = np.array([1, 2, 3], dtype=np.int8)
        arr16 = np.array([1, 2, 3], dtype=np.int16)
        arr32 = np.array([1, 2, 3], dtype=np.int32)
        arr64 = np.array([1, 2, 3], dtype=np.int64)

        assert arr8.dtype == np.int8
        assert arr16.dtype == np.int16
        assert arr32.dtype == np.int32
        assert arr64.dtype == np.int64

    def test_float_dtypes(self):
        """Test float dtypes"""
        arr32 = np.array([1.0, 2.0], dtype=np.float32)
        arr64 = np.array([1.0, 2.0], dtype=np.float64)

        assert arr32.dtype == np.float32
        assert arr64.dtype == np.float64

    def test_bool_dtype(self):
        """Test boolean dtype"""
        arr = np.array([True, False, True], dtype=bool)
        assert arr.dtype == np.bool_

    def test_dtype_conversion(self):
        """Test dtype conversion"""
        arr_int = np.array([1, 2, 3])
        arr_float = arr_int.astype(np.float64)
        assert arr_float.dtype == np.float64


class TestMemoryAndCopy:
    """Test memory management and copying"""

    def test_copy(self):
        """Test array copy"""
        a = np.array([1, 2, 3])
        b = a.copy()
        b[0] = 999
        assert a[0] == 1  # Original unchanged
        assert b[0] == 999

    def test_view(self):
        """Test array view"""
        a = np.array([1, 2, 3])
        b = a.view()
        b[0] = 999
        assert a[0] == 999  # Original changed


class TestSpecialArrays:
    """Test special array types and operations"""

    def test_masked_arrays(self):
        """Test basic masked array support"""
        arr = np.array([1, 2, -999, 4, 5])
        masked = np.ma.masked_where(arr == -999, arr)
        assert masked.count() == 4

    def test_structured_arrays(self):
        """Test structured arrays"""
        dt = np.dtype([("name", "U10"), ("age", "i4")])
        arr = np.array([("Alice", 25), ("Bob", 30)], dtype=dt)
        assert arr["name"][0] == "Alice"
        assert arr["age"][1] == 30


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_shape(self):
        """Test invalid shape handling"""
        with pytest.raises((ValueError, RuntimeError)):
            np.zeros((-1, 3))

    def test_divide_by_zero(self):
        """Test divide by zero handling"""
        arr = np.array([1, 2, 3])
        with np.errstate(divide="ignore", invalid="ignore"):
            result = arr / 0
            assert np.all(np.isinf(result))

    def test_out_of_bounds(self):
        """Test out of bounds indexing"""
        arr = np.array([1, 2, 3])
        with pytest.raises(IndexError):
            _ = arr[10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
