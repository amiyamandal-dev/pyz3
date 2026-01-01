"""Memory-mapped file utilities for pyz3.

This module provides utilities for memory-mapped I/O, enabling zero-copy
data access and shared memory between processes.

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

import mmap
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray

from pyz3.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MmapArrayInfo:
    """Information about a memory-mapped array."""

    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    size_bytes: int
    is_readonly: bool


class MmapArray:
    """A memory-mapped NumPy array with zero-copy access.

    This class wraps a NumPy array backed by a memory-mapped file,
    allowing efficient access to large arrays without loading them
    entirely into RAM.

    Example:
        >>> # Create a new mmap array
        >>> arr = MmapArray.create("data.bin", shape=(1000000,), dtype=np.float64)
        >>> arr[:] = np.random.randn(1000000)
        >>> arr.flush()
        >>>
        >>> # Later, open existing array (zero-copy)
        >>> arr = MmapArray.open("data.bin", shape=(1000000,), dtype=np.float64)
        >>> print(arr.mean())
    """

    def __init__(
        self,
        path: str | Path,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        mode: str = "r+",
    ):
        """Open an existing memory-mapped array.

        Args:
            path: Path to the memory-mapped file
            shape: Shape of the array
            dtype: Data type of the array
            mode: File mode ('r' for read-only, 'r+' for read-write)
        """
        self.path = Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mmap: mmap.mmap | None = None
        self._array: NDArray | None = None
        self._file = None

        self._open()

    def _open(self) -> None:
        """Open the memory-mapped file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Mmap file not found: {self.path}")

        expected_size = int(np.prod(self.shape)) * self.dtype.itemsize

        # Open file
        file_mode = "rb" if self.mode == "r" else "r+b"
        self._file = open(self.path, file_mode)

        # Create mmap
        access = mmap.ACCESS_READ if self.mode == "r" else mmap.ACCESS_WRITE
        self._mmap = mmap.mmap(self._file.fileno(), expected_size, access=access)

        # Create numpy array view
        self._array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self._mmap,
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        fill_value: float | None = None,
    ) -> "MmapArray":
        """Create a new memory-mapped array.

        Args:
            path: Path where the file will be created
            shape: Shape of the array
            dtype: Data type of the array
            fill_value: Optional value to fill the array with

        Returns:
            New MmapArray instance
        """
        path = Path(path)
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize

        # Create file with correct size
        with open(path, "wb") as f:
            f.write(b"\x00" * size)

        # Open and optionally fill
        arr = cls(path, shape, dtype, mode="r+")

        if fill_value is not None:
            arr._array.fill(fill_value)
            arr.flush()

        return arr

    @classmethod
    def from_array(
        cls,
        path: str | Path,
        array: NDArray,
    ) -> "MmapArray":
        """Create a memory-mapped array from an existing NumPy array.

        Args:
            path: Path where the file will be created
            array: NumPy array to save

        Returns:
            New MmapArray instance with data copied from input array
        """
        mmap_arr = cls.create(path, array.shape, array.dtype)
        mmap_arr._array[:] = array
        mmap_arr.flush()
        return mmap_arr

    @property
    def array(self) -> NDArray:
        """Get the underlying NumPy array."""
        if self._array is None:
            raise RuntimeError("MmapArray is closed")
        return self._array

    @property
    def info(self) -> MmapArrayInfo:
        """Get information about this mmap array."""
        return MmapArrayInfo(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            size_bytes=int(np.prod(self.shape)) * self.dtype.itemsize,
            is_readonly=self.mode == "r",
        )

    def flush(self) -> None:
        """Flush changes to disk."""
        if self._mmap is not None:
            self._mmap.flush()

    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._array = None

    def __enter__(self) -> "MmapArray":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # Array-like interface
    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self) -> int:
        return len(self.array)

    def __array__(self, dtype=None) -> NDArray:
        if dtype is None:
            return self.array
        return self.array.astype(dtype)

    # Common array methods
    def sum(self, *args, **kwargs):
        return self.array.sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.array.mean(*args, **kwargs)

    def std(self, *args, **kwargs):
        return self.array.std(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self.array.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self.array.max(*args, **kwargs)


class SharedMemoryArray:
    """Shared memory array for inter-process communication.

    This class creates a memory-mapped array that can be accessed
    by multiple processes without copying data.

    Example:
        >>> # In parent process
        >>> shared = SharedMemoryArray.create(shape=(1000,), dtype=np.float64)
        >>> shared[:] = np.arange(1000)
        >>>
        >>> # Pass shared.path to child process
        >>> # In child process
        >>> shared = SharedMemoryArray.open(path, shape=(1000,), dtype=np.float64)
        >>> print(shared.sum())  # Reads parent's data directly
    """

    def __init__(
        self,
        path: str | Path,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        create: bool = False,
    ):
        """Initialize shared memory array.

        Args:
            path: Path to the shared memory file
            shape: Shape of the array
            dtype: Data type
            create: If True, create new file; if False, open existing
        """
        self.path = Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._mmap_array: MmapArray | None = None

        if create:
            self._mmap_array = MmapArray.create(self.path, shape, dtype, fill_value=0)
        else:
            self._mmap_array = MmapArray(self.path, shape, dtype, mode="r+")

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        name: str | None = None,
    ) -> "SharedMemoryArray":
        """Create a new shared memory array.

        Args:
            shape: Shape of the array
            dtype: Data type
            name: Optional name for the shared memory file

        Returns:
            New SharedMemoryArray instance
        """
        if name is None:
            name = f"pyz3_shared_{os.getpid()}_{id(shape)}"

        # Create in temp directory for cross-process access
        path = Path(tempfile.gettempdir()) / f"{name}.mmap"

        return cls(path, shape, dtype, create=True)

    @classmethod
    def open(
        cls,
        path: str | Path,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
    ) -> "SharedMemoryArray":
        """Open an existing shared memory array.

        Args:
            path: Path to the shared memory file
            shape: Shape of the array
            dtype: Data type

        Returns:
            SharedMemoryArray instance
        """
        return cls(path, shape, dtype, create=False)

    @property
    def array(self) -> NDArray:
        """Get the underlying NumPy array."""
        if self._mmap_array is None:
            raise RuntimeError("SharedMemoryArray is closed")
        return self._mmap_array.array

    def flush(self) -> None:
        """Flush changes to disk."""
        if self._mmap_array is not None:
            self._mmap_array.flush()

    def close(self) -> None:
        """Close the shared memory."""
        if self._mmap_array is not None:
            self._mmap_array.close()
            self._mmap_array = None

    def unlink(self) -> None:
        """Remove the shared memory file."""
        self.close()
        if self.path.exists():
            self.path.unlink()

    def __enter__(self) -> "SharedMemoryArray":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self) -> int:
        return len(self.array)


@contextmanager
def mmap_file(
    path: str | Path,
    mode: str = "r",
) -> Generator[mmap.mmap, None, None]:
    """Context manager for memory-mapping a file.

    Args:
        path: Path to the file
        mode: 'r' for read-only, 'r+' for read-write

    Yields:
        mmap object

    Example:
        >>> with mmap_file("large_file.bin") as mm:
        ...     data = mm[0:1000]  # Read first 1000 bytes
    """
    path = Path(path)
    file_mode = "rb" if mode == "r" else "r+b"
    access = mmap.ACCESS_READ if mode == "r" else mmap.ACCESS_WRITE

    with open(path, file_mode) as f:
        with mmap.mmap(f.fileno(), 0, access=access) as mm:
            yield mm


def load_binary_mmap(
    path: str | Path,
    dtype: DTypeLike = np.float64,
    shape: tuple[int, ...] | None = None,
    offset: int = 0,
) -> NDArray:
    """Load a binary file as a memory-mapped NumPy array.

    This is a convenience function for loading binary data files
    with zero-copy access.

    Args:
        path: Path to the binary file
        dtype: Data type of the array
        shape: Shape of the array (inferred from file size if None)
        offset: Byte offset into the file

    Returns:
        Memory-mapped NumPy array

    Example:
        >>> # Load a binary file of float64 values
        >>> arr = load_binary_mmap("weights.bin", dtype=np.float64)
        >>> print(arr.shape, arr.mean())
    """
    path = Path(path)
    dtype = np.dtype(dtype)

    file_size = path.stat().st_size - offset

    if shape is None:
        # Infer shape as 1D array
        n_elements = file_size // dtype.itemsize
        shape = (n_elements,)

    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def save_binary_mmap(
    path: str | Path,
    array: NDArray,
) -> None:
    """Save a NumPy array to a binary file suitable for mmap loading.

    Args:
        path: Path where the file will be saved
        array: NumPy array to save

    Example:
        >>> arr = np.random.randn(1000000)
        >>> save_binary_mmap("data.bin", arr)
        >>>
        >>> # Later load with zero-copy
        >>> loaded = load_binary_mmap("data.bin", dtype=np.float64)
    """
    array.tofile(path)


class MmapCache:
    """Cache for memory-mapped arrays.

    Provides automatic caching and lifecycle management for
    frequently accessed memory-mapped files.

    Example:
        >>> cache = MmapCache(max_size=10)
        >>> arr = cache.get_or_create("data.bin", shape=(1000,), dtype=np.float64)
        >>> # Array is cached for fast subsequent access
        >>> arr2 = cache.get_or_create("data.bin", shape=(1000,), dtype=np.float64)
        >>> assert arr is arr2  # Same object
    """

    def __init__(self, max_size: int = 100):
        """Initialize the cache.

        Args:
            max_size: Maximum number of cached arrays
        """
        self.max_size = max_size
        self._cache: dict[Path, MmapArray] = {}
        self._access_order: list[Path] = []

    def get_or_create(
        self,
        path: str | Path,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        mode: str = "r",
    ) -> MmapArray:
        """Get a cached mmap array or create a new one.

        Args:
            path: Path to the mmap file
            shape: Shape of the array
            dtype: Data type
            mode: File mode

        Returns:
            MmapArray instance (possibly cached)
        """
        path = Path(path).resolve()

        if path in self._cache:
            # Move to end of access order
            self._access_order.remove(path)
            self._access_order.append(path)
            return self._cache[path]

        # Create new mmap array
        arr = MmapArray(path, shape, dtype, mode)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            old_arr = self._cache.pop(oldest)
            old_arr.close()

        self._cache[path] = arr
        self._access_order.append(path)

        return arr

    def clear(self) -> None:
        """Clear all cached arrays."""
        for arr in self._cache.values():
            arr.close()
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, path: str | Path) -> bool:
        return Path(path).resolve() in self._cache
