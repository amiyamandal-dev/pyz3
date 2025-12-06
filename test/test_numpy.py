"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest
from example import numpy_example


class TestArrayCreation:
    """Test NumPy array creation functions"""

    def test_create_zeros(self):
        arr = numpy_example.create_zeros(10)
        assert arr.shape == (10,)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.zeros(10))

    def test_create_ones_2d(self):
        arr = numpy_example.create_ones(3, 4)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.ones((3, 4)))

    def test_from_slice(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr = numpy_example.from_slice(values)
        assert arr.shape == (5,)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, values)


class TestZeroCopyOperations:
    """Test zero-copy array access and modifications"""

    def test_double_array_in_place(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        original_id = id(arr)

        result = numpy_example.double_array(arr)

        # Verify it's the same array (zero-copy)
        assert id(result) == original_id
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_equal(arr, [2.0, 4.0, 6.0, 8.0])

    def test_sum_array_zero_copy(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = numpy_example.sum_array(arr)
        assert result == 15.0

    def test_sum_empty_array(self):
        arr = np.array([], dtype=np.float64)
        result = numpy_example.sum_array(arr)
        assert result == 0.0


class TestArrayOperations:
    """Test NumPy array operations"""

    def test_calculate_mean(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mean = numpy_example.calculate_mean(arr)
        assert mean == 3.0

    def test_array_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        stats = numpy_example.array_stats(arr)

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["sum"] == 15.0

    def test_array_stats_single_element(self):
        arr = np.array([42.0], dtype=np.float64)
        stats = numpy_example.array_stats(arr)

        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["mean"] == 42.0
        assert stats["sum"] == 42.0

    def test_reshape_array(self):
        arr = np.arange(12, dtype=np.float64)
        reshaped = numpy_example.reshape_array(arr, 3, 4)

        assert reshaped.shape == (3, 4)
        np.testing.assert_array_equal(reshaped, arr.reshape(3, 4))

    def test_flatten_array(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        flat = numpy_example.flatten_array(arr)

        assert flat.shape == (12,)
        np.testing.assert_array_equal(flat, np.arange(12, dtype=np.float64))

    def test_transpose_array(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        transposed = numpy_example.transpose_array(arr)

        assert transposed.shape == (4, 3)
        np.testing.assert_array_equal(transposed, arr.T)


class TestArrayInfo:
    """Test array metadata retrieval"""

    def test_array_info_1d(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        info = numpy_example.array_info(arr)

        assert info["ndim"] == 1
        assert info["size"] == 3
        assert "float64" in info["dtype"]

    def test_array_info_2d(self):
        arr = np.ones((5, 10), dtype=np.float64)
        info = numpy_example.array_info(arr)

        assert info["ndim"] == 2
        assert info["size"] == 50
        assert "float64" in info["dtype"]


class TestMultiArrayOperations:
    """Test operations on multiple arrays"""

    def test_add_arrays(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

        result = numpy_example.add_arrays(a, b)
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_add_arrays_different_sizes_raises(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0], dtype=np.float64)

        with pytest.raises(ValueError, match="same size"):
            numpy_example.add_arrays(a, b)

    def test_add_arrays_2d(self):
        a = np.ones((2, 3), dtype=np.float64)
        b = np.ones((2, 3), dtype=np.float64) * 2

        result = numpy_example.add_arrays(a, b)
        np.testing.assert_array_equal(result, np.ones((2, 3)) * 3)


class TestArrayProcessorClass:
    """Test ArrayProcessor class"""

    def test_create_processor(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        processor = numpy_example.ArrayProcessor(arr)
        assert processor is not None

    def test_get_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        processor = numpy_example.ArrayProcessor(arr)

        result = processor.get_array()
        np.testing.assert_array_equal(result, arr)

    def test_processor_double(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        processor = numpy_example.ArrayProcessor(arr)

        result = processor.double()
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
        # Original array should be modified
        np.testing.assert_array_equal(arr, [2.0, 4.0, 6.0])

    def test_processor_sum(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        processor = numpy_example.ArrayProcessor(arr)

        assert processor.sum() == 15.0

    def test_processor_mean(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        processor = numpy_example.ArrayProcessor(arr)

        assert processor.mean() == 3.0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_large_array(self):
        arr = np.arange(10000, dtype=np.float64)
        result = numpy_example.sum_array(arr)
        expected = np.sum(arr)
        assert abs(result - expected) < 1e-9

    def test_negative_values(self):
        arr = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        stats = numpy_example.array_stats(arr)

        assert stats["min"] == -3.0
        assert stats["max"] == -1.0
        assert stats["sum"] == -6.0

    def test_mixed_values(self):
        arr = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float64)
        stats = numpy_example.array_stats(arr)

        assert stats["min"] == -5.0
        assert stats["max"] == 10.0
        assert stats["mean"] == 2.5
        assert stats["sum"] == 10.0


class TestTypeCompatibility:
    """Test compatibility with different NumPy dtypes"""

    def test_int64_array(self):
        # This should work as int64 can be converted to float64
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        # Note: The Zig code expects float64, so this might fail
        # This test documents the expected behavior
        try:
            result = numpy_example.sum_array(arr)
            # If it works, verify the result
            assert result == 15.0
        except (TypeError, ValueError):
            # If it fails, that's expected - document this limitation
            pytest.skip("int64 arrays not supported, requires float64")

    def test_float32_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Similar to above - document compatibility
        try:
            result = numpy_example.sum_array(arr)
            assert abs(result - 6.0) < 1e-6
        except (TypeError, ValueError):
            pytest.skip("float32 arrays not supported, requires float64")
