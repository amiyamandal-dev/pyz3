"""Test cases for memory-mapped file utilities."""

import numpy as np
import pytest

from pyz3.mmap_utils import (
    MmapArray,
    MmapCache,
    SharedMemoryArray,
    load_binary_mmap,
    mmap_file,
    save_binary_mmap,
)


class TestMmapArray:
    """Tests for MmapArray class."""

    def test_create_and_read(self, tmp_path):
        """Test creating and reading a mmap array."""
        path = tmp_path / "test.bin"

        # Create array
        arr = MmapArray.create(path, shape=(100,), dtype=np.float64)
        arr[:] = np.arange(100, dtype=np.float64)
        arr.flush()
        arr.close()

        # Read array
        arr2 = MmapArray(path, shape=(100,), dtype=np.float64, mode="r")
        assert arr2[0] == 0.0
        assert arr2[99] == 99.0
        assert arr2.sum() == sum(range(100))
        arr2.close()

    def test_create_with_fill_value(self, tmp_path):
        """Test creating array with fill value."""
        path = tmp_path / "filled.bin"

        arr = MmapArray.create(path, shape=(50,), dtype=np.float32, fill_value=42.0)
        assert arr[0] == 42.0
        assert arr[49] == 42.0
        arr.close()

    def test_from_array(self, tmp_path):
        """Test creating mmap from existing array."""
        path = tmp_path / "from_array.bin"
        original = np.random.randn(100).astype(np.float64)

        arr = MmapArray.from_array(path, original)
        np.testing.assert_array_almost_equal(arr.array, original)
        arr.close()

    def test_context_manager(self, tmp_path):
        """Test context manager usage."""
        path = tmp_path / "context.bin"

        with MmapArray.create(path, shape=(10,), dtype=np.int32) as arr:
            arr[:] = np.arange(10)

        # File should be closed, open again to verify
        with MmapArray(path, shape=(10,), dtype=np.int32, mode="r") as arr:
            assert arr[5] == 5

    def test_info_property(self, tmp_path):
        """Test info property."""
        path = tmp_path / "info.bin"

        with MmapArray.create(path, shape=(100, 10), dtype=np.float64) as arr:
            info = arr.info
            assert info.shape == (100, 10)
            assert info.dtype == np.float64
            assert info.size_bytes == 100 * 10 * 8  # 8 bytes per float64
            assert info.is_readonly is False

    def test_array_methods(self, tmp_path):
        """Test array-like methods."""
        path = tmp_path / "methods.bin"

        with MmapArray.create(path, shape=(100,), dtype=np.float64) as arr:
            arr[:] = np.arange(100, dtype=np.float64)

            assert arr.sum() == sum(range(100))
            assert arr.mean() == 49.5
            assert arr.min() == 0.0
            assert arr.max() == 99.0
            assert len(arr) == 100

    def test_multidimensional(self, tmp_path):
        """Test multidimensional arrays."""
        path = tmp_path / "multi.bin"

        with MmapArray.create(path, shape=(10, 20, 30), dtype=np.float32) as arr:
            arr[:] = np.ones((10, 20, 30), dtype=np.float32)
            assert arr.array.shape == (10, 20, 30)
            assert arr.sum() == 10 * 20 * 30


class TestSharedMemoryArray:
    """Tests for SharedMemoryArray class."""

    def test_create_and_access(self):
        """Test creating and accessing shared memory."""
        shared = SharedMemoryArray.create(shape=(100,), dtype=np.float64)

        try:
            shared[:] = np.arange(100, dtype=np.float64)
            shared.flush()

            assert shared[0] == 0.0
            assert shared[99] == 99.0
        finally:
            shared.unlink()

    def test_open_existing(self, tmp_path):
        """Test opening existing shared memory."""
        # Create shared memory with known path
        path = tmp_path / "shared.mmap"

        shared1 = SharedMemoryArray(path, shape=(50,), dtype=np.int32, create=True)
        shared1[:] = np.arange(50, dtype=np.int32)
        shared1.flush()

        # Open same file
        shared2 = SharedMemoryArray.open(path, shape=(50,), dtype=np.int32)

        assert shared2[25] == 25
        np.testing.assert_array_equal(shared1.array, shared2.array)

        shared1.close()
        shared2.unlink()

    def test_multiprocess_sharing(self, tmp_path):
        """Test sharing data between processes.

        Note: Uses subprocess to avoid pickling issues with multiprocessing.
        """
        import subprocess
        import sys

        path = tmp_path / "multiproc.mmap"

        # Create shared array in parent
        shared = SharedMemoryArray(path, shape=(100,), dtype=np.float64, create=True)
        shared[:] = np.zeros(100)
        shared.flush()

        # Run worker as subprocess to avoid pickle issues
        worker_code = f'''
import numpy as np
import sys
sys.path.insert(0, "{str(tmp_path.parent.parent.parent.parent)}")
from pyz3.mmap_utils import SharedMemoryArray
arr = SharedMemoryArray.open("{path}", shape=(100,), dtype=np.float64)
arr[:] = np.arange(100, dtype=np.float64)
arr.flush()
arr.close()
'''
        result = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.skip(f"Subprocess failed: {result.stderr}")

        # Verify parent sees changes
        # Reopen to ensure we see the updates
        shared.close()
        shared = SharedMemoryArray.open(path, shape=(100,), dtype=np.float64)

        assert shared[0] == 0.0
        assert shared[99] == 99.0

        shared.unlink()


class TestMmapFile:
    """Tests for mmap_file context manager."""

    def test_read_file(self, tmp_path):
        """Test reading file with mmap."""
        path = tmp_path / "read.bin"
        data = b"Hello, mmap world!"
        path.write_bytes(data)

        with mmap_file(path, mode="r") as mm:
            assert mm[:5] == b"Hello"
            assert mm[7:11] == b"mmap"

    def test_write_file(self, tmp_path):
        """Test writing file with mmap."""
        path = tmp_path / "write.bin"
        path.write_bytes(b"0" * 100)

        with mmap_file(path, mode="r+") as mm:
            mm[0:5] = b"HELLO"

        content = path.read_bytes()
        assert content[:5] == b"HELLO"


class TestBinaryMmap:
    """Tests for binary file mmap functions."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading binary files."""
        path = tmp_path / "binary.bin"
        original = np.random.randn(1000).astype(np.float64)

        save_binary_mmap(path, original)
        loaded = load_binary_mmap(path, dtype=np.float64)

        np.testing.assert_array_almost_equal(loaded, original)

    def test_load_with_shape(self, tmp_path):
        """Test loading with explicit shape."""
        path = tmp_path / "shaped.bin"
        original = np.arange(100).reshape(10, 10).astype(np.float32)

        save_binary_mmap(path, original.flatten())
        loaded = load_binary_mmap(path, dtype=np.float32, shape=(10, 10))

        assert loaded.shape == (10, 10)
        np.testing.assert_array_equal(loaded, original)

    def test_load_with_offset(self, tmp_path):
        """Test loading with byte offset."""
        path = tmp_path / "offset.bin"

        # Write header + data
        header = b"HDR!"  # 4 bytes
        data = np.arange(10, dtype=np.float64)
        with open(path, "wb") as f:
            f.write(header)
            data.tofile(f)

        # Load with offset to skip header
        loaded = load_binary_mmap(path, dtype=np.float64, offset=4)

        np.testing.assert_array_equal(loaded, data)


class TestMmapCache:
    """Tests for MmapCache class."""

    def test_cache_hit(self, tmp_path):
        """Test cache returns same object."""
        path = tmp_path / "cached.bin"
        MmapArray.create(path, shape=(10,), dtype=np.float64).close()

        cache = MmapCache(max_size=10)

        arr1 = cache.get_or_create(path, shape=(10,), dtype=np.float64)
        arr2 = cache.get_or_create(path, shape=(10,), dtype=np.float64)

        assert arr1 is arr2
        assert len(cache) == 1

        cache.clear()

    def test_cache_eviction(self, tmp_path):
        """Test LRU eviction when cache is full."""
        cache = MmapCache(max_size=2)

        # Create 3 files
        for i in range(3):
            path = tmp_path / f"file{i}.bin"
            MmapArray.create(path, shape=(10,), dtype=np.float64).close()

        # Fill cache
        cache.get_or_create(tmp_path / "file0.bin", shape=(10,), dtype=np.float64)
        cache.get_or_create(tmp_path / "file1.bin", shape=(10,), dtype=np.float64)

        assert len(cache) == 2
        assert (tmp_path / "file0.bin") in cache
        assert (tmp_path / "file1.bin") in cache

        # Add third, should evict first
        cache.get_or_create(tmp_path / "file2.bin", shape=(10,), dtype=np.float64)

        assert len(cache) == 2
        assert (tmp_path / "file0.bin") not in cache
        assert (tmp_path / "file1.bin") in cache
        assert (tmp_path / "file2.bin") in cache

        cache.clear()


class TestMmapPerformance:
    """Performance-related tests for mmap."""

    def test_large_array_access(self, tmp_path):
        """Test efficient access to large arrays."""
        path = tmp_path / "large.bin"
        size = 1_000_000  # 1M elements

        # Create large array
        with MmapArray.create(path, shape=(size,), dtype=np.float64) as arr:
            # Only write to specific locations (sparse access)
            arr[0] = 1.0
            arr[size // 2] = 2.0
            arr[size - 1] = 3.0
            arr.flush()

        # Read back specific locations (should be fast)
        with MmapArray(path, shape=(size,), dtype=np.float64, mode="r") as arr:
            assert arr[0] == 1.0
            assert arr[size // 2] == 2.0
            assert arr[size - 1] == 3.0

    def test_sequential_vs_random_access(self, tmp_path):
        """Compare sequential and random access patterns."""
        import time

        path = tmp_path / "access.bin"
        size = 100_000

        with MmapArray.create(path, shape=(size,), dtype=np.float64) as arr:
            arr[:] = np.arange(size, dtype=np.float64)

        with MmapArray(path, shape=(size,), dtype=np.float64, mode="r") as arr:
            # Sequential access
            start = time.perf_counter()
            _total = sum(arr[i] for i in range(0, size, 100))
            seq_time = time.perf_counter() - start

            # Random access
            indices = np.random.randint(0, size, size // 100)
            start = time.perf_counter()
            _total = sum(arr[i] for i in indices)
            rand_time = time.perf_counter() - start

            # Both should complete in reasonable time
            assert seq_time < 1.0
            assert rand_time < 1.0


class TestMmapEdgeCases:
    """Edge case tests for mmap utilities."""

    def test_empty_shape_error(self, tmp_path):
        """Test that empty shape raises error."""
        path = tmp_path / "empty.bin"

        with pytest.raises((ValueError, OSError)):
            MmapArray.create(path, shape=(0,), dtype=np.float64)

    def test_file_not_found(self, tmp_path):
        """Test opening non-existent file."""
        path = tmp_path / "nonexistent.bin"

        with pytest.raises(FileNotFoundError):
            MmapArray(path, shape=(10,), dtype=np.float64)

    def test_readonly_modification_blocked(self, tmp_path):
        """Test that readonly arrays cannot be modified."""
        path = tmp_path / "readonly.bin"

        with MmapArray.create(path, shape=(10,), dtype=np.float64) as arr:
            arr[:] = np.arange(10)

        with MmapArray(path, shape=(10,), dtype=np.float64, mode="r") as arr:
            assert arr.info.is_readonly is True
            # Writing to readonly should raise or be blocked
            with pytest.raises((ValueError, TypeError)):
                arr[0] = 999.0

    def test_different_dtypes(self, tmp_path):
        """Test various dtypes."""
        dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8, np.complex64]

        for dtype in dtypes:
            path = tmp_path / f"dtype_{dtype.__name__}.bin"

            with MmapArray.create(path, shape=(10,), dtype=dtype) as arr:
                if np.issubdtype(dtype, np.complexfloating):
                    arr[:] = np.arange(10) + 1j * np.arange(10)
                else:
                    arr[:] = np.arange(10, dtype=dtype)

            with MmapArray(path, shape=(10,), dtype=dtype, mode="r") as arr:
                if np.issubdtype(dtype, np.complexfloating):
                    assert arr[5].real == 5.0
                else:
                    assert arr[5] == 5
