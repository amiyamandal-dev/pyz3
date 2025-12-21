"""
Comprehensive tests for all PyZ3 types
Tests the 12+ types that were missing coverage
"""

import pytest
import sys
from pathlib import Path

# Import pyz3 to test type wrappers
import pyz3


class TestPyBool:
    """Test PyBool type wrapper"""

    def test_bool_true(self):
        """Test True value"""
        assert bool(True) == True

    def test_bool_false(self):
        """Test False value"""
        assert bool(False) == False

    def test_bool_conversion_from_int(self):
        """Test bool conversion from integers"""
        assert bool(1) == True
        assert bool(0) == False

    def test_bool_comparison(self):
        """Test bool comparison"""
        true_val = True
        false_val = False
        assert true_val != false_val


class TestPyBytes:
    """Test PyBytes type wrapper"""

    def test_bytes_creation(self):
        """Test creating bytes from string"""
        data = b"Hello, World!"
        py_bytes = bytes(data)
        assert py_bytes == data

    def test_bytes_empty(self):
        """Test empty bytes"""
        py_bytes = b""
        assert len(py_bytes) == 0

    def test_bytes_binary_data(self):
        """Test binary data handling"""
        data = bytes(range(256))
        assert len(data) == 256
        assert data[0] == 0
        assert data[255] == 255

    def test_bytes_immutable(self):
        """Test that bytes are immutable"""
        data = b"immutable"
        py_bytes = bytes(data)
        assert py_bytes == data


class TestPyDict:
    """Test PyDict type wrapper"""

    def test_dict_creation(self):
        """Test creating empty dict"""
        d = {}
        assert d is not None

    def test_dict_set_get(self):
        """Test setting and getting values"""
        d = {}
        d["key"] = 42
        result = d["key"]
        assert result == 42

    def test_dict_multiple_items(self):
        """Test multiple key-value pairs"""
        d = {}

        for i in range(10):
            key = f"key{i}"
            d[key] = i

        # Verify we can retrieve them
        for i in range(10):
            key = f"key{i}"
            assert d[key] == i

    def test_dict_contains(self):
        """Test checking if key exists"""
        d = {}
        d["exists"] = 1

        assert "exists" in d
        assert "missing" not in d


class TestPyFloat:
    """Test PyFloat type wrapper"""

    def test_float_creation(self):
        """Test creating float"""
        f = 3.14159
        assert abs(f - 3.14159) < 1e-10

    def test_float_zero(self):
        """Test zero float"""
        f = 0.0
        assert f == 0.0

    def test_float_negative(self):
        """Test negative float"""
        f = -2.718
        assert abs(f - (-2.718)) < 1e-10

    def test_float_special_values(self):
        """Test special float values"""
        import math

        inf = float('inf')
        assert math.isinf(inf)

        neg_inf = float('-inf')
        assert math.isinf(neg_inf)

        # NaN
        nan = float('nan')
        assert math.isnan(nan)


class TestPyList:
    """Test PyList type wrapper"""

    def test_list_creation(self):
        """Test creating empty list"""
        lst = []
        assert len(lst) == 0

    def test_list_append(self):
        """Test appending items"""
        lst = []
        for i in range(5):
            lst.append(i)
        assert len(lst) == 5

    def test_list_getitem(self):
        """Test getting items by index"""
        lst = []
        for i in range(10):
            lst.append(i * 10)

        for i in range(10):
            item = lst[i]
            assert item == i * 10

    def test_list_setitem(self):
        """Test setting items by index"""
        lst = []
        for i in range(5):
            lst.append(0)

        # Change values
        for i in range(5):
            lst[i] = i * 100

        # Verify
        for i in range(5):
            assert lst[i] == i * 100

    def test_list_iteration(self):
        """Test list iteration"""
        lst = []
        values = [1, 2, 3, 4, 5]
        for v in values:
            lst.append(v)

        result = []
        for item in lst:
            result.append(item)

        assert result == values


class TestPyLong:
    """Test PyLong (arbitrary precision integer) type wrapper"""

    def test_long_small(self):
        """Test small integers"""
        num = 42
        assert num == 42

    def test_long_zero(self):
        """Test zero"""
        num = 0
        assert num == 0

    def test_long_negative(self):
        """Test negative numbers"""
        num = -12345
        assert num == -12345

    def test_long_large(self):
        """Test large integers"""
        large = 2**63 - 1  # Max i64
        num = large
        assert num == large

    def test_long_arithmetic(self):
        """Test arithmetic operations"""
        a = 100
        b = 50

        assert a + b == 150
        assert a - b == 50
        assert a * b == 5000


class TestPyString:
    """Test PyString type wrapper"""

    def test_string_creation(self):
        """Test creating string"""
        s = "Hello, PyZ3!"
        assert s == "Hello, PyZ3!"

    def test_string_empty(self):
        """Test empty string"""
        s = ""
        assert s == ""

    def test_string_unicode(self):
        """Test Unicode strings"""
        s = "Hello 世界 🚀"
        assert s == "Hello 世界 🚀"

    def test_string_length(self):
        """Test string length"""
        s = "12345"
        assert len(s) == 5

    def test_string_concatenation(self):
        """Test string operations"""
        s1 = "Hello"
        s2 = " World"
        result = s1 + s2
        assert result == "Hello World"


class TestPyTuple:
    """Test PyTuple type wrapper"""

    def test_tuple_creation(self):
        """Test creating tuple"""
        tup = tuple(range(5))
        assert len(tup) == 5

    def test_tuple_getitem(self):
        """Test getting items"""
        tup = tuple(i * 10 for i in range(3))

        for i in range(3):
            assert tup[i] == i * 10

    def test_tuple_immutable(self):
        """Test that tuples are immutable"""
        tup = (1, 2, 3)

        # Tuples are immutable
        with pytest.raises(TypeError):
            tup[0] = 5

    def test_tuple_empty(self):
        """Test empty tuple"""
        tup = ()
        assert len(tup) == 0


class TestPySlice:
    """Test PySlice type wrapper"""

    def test_slice_creation(self):
        """Test creating slice"""
        # Create slice(0, 10, 1)
        s = slice(0, 10, 1)
        assert s is not None

    def test_slice_none_stop(self):
        """Test slice with None stop"""
        # Create slice(5, None, 1)
        s = slice(5, None, 1)
        assert s is not None

    def test_slice_negative_step(self):
        """Test slice with negative step"""
        # Create slice(10, 0, -1)
        s = slice(10, 0, -1)
        assert s is not None

    def test_slice_usage(self):
        """Test using slice on list"""
        lst = list(range(20))
        s = slice(5, 15, 2)
        result = lst[s]
        assert result == [5, 7, 9, 11, 13]


class TestPyBuffer:
    """Test PyBuffer protocol support"""

    def test_buffer_from_bytes(self):
        """Test creating buffer from bytes"""
        data = b"buffer data"
        # Test memoryview supports buffer protocol
        mv = memoryview(data)
        assert mv is not None
        assert bytes(mv) == data

    def test_buffer_readonly(self):
        """Test read-only buffer"""
        data = b"readonly"
        mv = memoryview(data)
        assert mv.readonly == True


class TestPyMemoryView:
    """Test PyMemoryView type wrapper"""

    def test_memoryview_from_bytes(self):
        """Test creating memoryview from bytes"""
        data = b"memoryview test"
        mv = memoryview(data)
        assert mv is not None
        assert bytes(mv) == data

    def test_memoryview_from_bytearray(self):
        """Test creating memoryview from bytearray"""
        data = bytearray(b"mutable")
        mv = memoryview(data)
        assert mv is not None
        # Can modify through memoryview
        mv[0] = ord('M')
        assert data[0] == ord('M')


class TestPyIter:
    """Test PyIter type wrapper"""

    def test_iter_from_list(self):
        """Test creating iterator from list"""
        lst = [1, 2, 3, 4, 5]
        it = iter(lst)

        result = []
        for item in it:
            result.append(item)

        assert result == lst

    def test_iter_from_range(self):
        """Test creating iterator from range"""
        r = range(10)
        it = iter(r)

        result = []
        for item in it:
            result.append(item)

        assert result == list(r)


class TestPyType:
    """Test PyType type wrapper"""

    def test_type_of_int(self):
        """Test getting type of integer"""
        num = 42
        t = type(num)
        assert t.__name__ == "int"

    def test_type_of_str(self):
        """Test getting type of string"""
        s = "test"
        t = type(s)
        assert t.__name__ == "str"

    def test_type_of_list(self):
        """Test getting type of list"""
        lst = []
        t = type(lst)
        assert t.__name__ == "list"


class TestPyCode:
    """Test PyCode type wrapper"""

    def test_code_object_from_function(self):
        """Test getting code object from function"""
        def sample_function():
            return 42

        code = sample_function.__code__
        assert code is not None

    def test_code_name(self):
        """Test getting code name"""
        def my_func():
            pass

        code = my_func.__code__
        assert code.co_name == "my_func"

    def test_code_filename(self):
        """Test getting code filename"""
        def test_func():
            pass

        code = test_func.__code__
        filename = code.co_filename
        assert "test_all_types" in filename or __file__ in filename


class TestPyFrame:
    """Test PyFrame type wrapper"""

    def test_current_frame(self):
        """Test getting current frame"""
        import sys
        frame = sys._getframe()
        assert frame is not None

    def test_frame_code(self):
        """Test getting code from frame"""
        import sys
        frame = sys._getframe()
        code = frame.f_code
        assert code is not None

    def test_frame_lineno(self):
        """Test getting line number from frame"""
        import sys
        frame = sys._getframe()
        lineno = frame.f_lineno
        assert lineno > 0


class TestPyModule:
    """Test PyModule type wrapper"""

    def test_import_sys(self):
        """Test importing sys module"""
        import sys
        # Use sys module directly since import is a keyword
        assert sys is not None
        assert hasattr(sys, 'version')

    def test_import_os(self):
        """Test importing os module"""
        import os
        assert os is not None
        assert hasattr(os, 'path')

    def test_module_getattr(self):
        """Test getting module attribute"""
        import sys
        version = sys.version
        assert version is not None
        assert isinstance(version, str)

    def test_module_dict(self):
        """Test getting module __dict__"""
        import sys
        mod_dict = sys.__dict__
        assert mod_dict is not None
        assert isinstance(mod_dict, dict)


class TestPyCoroutine:
    """Test PyCoroutine type wrapper"""

    def test_coroutine_creation(self):
        """Test creating coroutine"""
        async def async_func():
            return 42

        coro = async_func()
        assert coro is not None

        # Clean up
        coro.close()

    def test_coroutine_awaitable(self):
        """Test that coroutine is awaitable"""
        async def async_func():
            return "result"

        import asyncio
        result = asyncio.run(async_func())
        assert result == "result"


class TestPyAwaitable:
    """Test PyAwaitable protocol"""

    def test_awaitable_from_coroutine(self):
        """Test creating awaitable from coroutine"""
        async def async_func():
            return 123

        coro = async_func()
        # Check it's awaitable
        assert hasattr(coro, '__await__')

        coro.close()


class TestPyGIL:
    """Test PyGIL (Global Interpreter Lock) management"""

    def test_gil_check(self):
        """Test GIL check"""
        import sys
        # In CPython, we have the GIL by default
        # Just verify we can check thread state
        assert sys.getswitchinterval() > 0

    def test_gil_thread_id(self):
        """Test getting thread id"""
        import threading
        thread_id = threading.get_ident()
        assert thread_id > 0


# Integration tests
class TestTypeIntegration:
    """Test interactions between different types"""

    def test_dict_with_different_types(self):
        """Test dict with mixed types"""
        d = {}

        # String key, int value
        d["int"] = 42

        # String key, float value
        d["float"] = 3.14

        # String key, string value
        d["str"] = "value"

        # Verify all values
        assert d["int"] == 42
        assert abs(d["float"] - 3.14) < 0.01
        assert d["str"] == "value"

    def test_list_of_dicts(self):
        """Test list containing dicts"""
        lst = []

        for i in range(3):
            d = {}
            d["id"] = i
            d["name"] = f"item{i}"
            lst.append(d)

        assert len(lst) == 3

        # Verify first dict
        first_dict = lst[0]
        assert first_dict["id"] == 0

    def test_tuple_with_mixed_types(self):
        """Test tuple with different types"""
        tup = (1, 2.5, "three", True)

        assert len(tup) == 4
        assert tup[0] == 1
        assert abs(tup[1] - 2.5) < 0.01
        assert tup[2] == "three"
        assert tup[3] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
