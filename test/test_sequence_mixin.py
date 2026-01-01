"""
Tests for PySequence mixin functionality.

This test file verifies that sequence operations work correctly
when accessed from Python through Zig extensions using PySequenceMixin.
"""

import pytest


def test_sequence_basic_operations():
    """Test basic sequence operations like len, getitem, contains."""
    # These will work once we have a Zig module that uses PySequenceMixin
    # For now, we test with Python's built-in sequences

    test_list = [1, 2, 3, 4, 5]

    # Length
    assert len(test_list) == 5

    # Indexing (positive and negative)
    assert test_list[0] == 1
    assert test_list[-1] == 5

    # Membership
    assert 3 in test_list
    assert 99 not in test_list


def test_sequence_search_operations():
    """Test search and membership operations."""
    test_list = [10, 20, 10, 30, 10]

    # Index
    assert test_list.index(20) == 1
    assert test_list.index(10) == 0  # First occurrence

    # Count
    assert test_list.count(10) == 3
    assert test_list.count(20) == 1
    assert test_list.count(99) == 0

    # Contains
    assert 20 in test_list
    assert 99 not in test_list


def test_sequence_slicing():
    """Test slice operations."""
    test_list = [0, 1, 2, 3, 4]

    # Get slice
    assert test_list[1:4] == [1, 2, 3]
    assert test_list[:2] == [0, 1]
    assert test_list[3:] == [3, 4]
    assert test_list[::2] == [0, 2, 4]

    # Negative indices in slices
    assert test_list[-2:] == [3, 4]
    assert test_list[:-2] == [0, 1, 2]


def test_sequence_concatenation():
    """Test concatenation and repetition."""
    list1 = [1, 2]
    list2 = [3, 4]

    # Concatenation
    assert list1 + list2 == [1, 2, 3, 4]

    # Repetition
    assert list1 * 3 == [1, 2, 1, 2, 1, 2]

    # In-place concatenation
    list3 = [1, 2]
    list3 += [3, 4]
    assert list3 == [1, 2, 3, 4]

    # In-place repetition
    list4 = [1, 2]
    list4 *= 2
    assert list4 == [1, 2, 1, 2]


def test_sequence_iteration():
    """Test iteration over sequences."""
    test_list = [10, 20, 30]

    # Basic iteration
    result = []
    for item in test_list:
        result.append(item)
    assert result == [10, 20, 30]

    # Enumerate
    indexed = list(enumerate(test_list))
    assert indexed == [(0, 10), (1, 20), (2, 30)]

    # List comprehension
    doubled = [x * 2 for x in test_list]
    assert doubled == [20, 40, 60]


def test_sequence_mutable_operations():
    """Test mutable sequence operations."""
    test_list = [1, 2, 3, 4, 5]

    # Set item
    test_list[0] = 10
    assert test_list[0] == 10

    # Delete item
    del test_list[1]
    assert test_list == [10, 3, 4, 5]

    # Set slice
    test_list[1:3] = [30, 40]
    assert test_list == [10, 30, 40, 5]

    # Delete slice
    del test_list[1:3]
    assert test_list == [10, 5]


def test_sequence_conversion():
    """Test conversion between sequence types."""
    # List to tuple
    test_list = [1, 2, 3]
    test_tuple = tuple(test_list)
    assert test_tuple == (1, 2, 3)
    assert isinstance(test_tuple, tuple)

    # Tuple to list
    test_list2 = list(test_tuple)
    assert test_list2 == [1, 2, 3]
    assert isinstance(test_list2, list)

    # Range to list
    test_range = range(5)
    range_list = list(test_range)
    assert range_list == [0, 1, 2, 3, 4]


def test_sequence_edge_cases():
    """Test edge cases and error conditions."""
    test_list = [1, 2, 3]

    # Empty sequence
    empty_list = []
    assert len(empty_list) == 0
    assert list(empty_list) == []

    # Index out of range
    with pytest.raises(IndexError):
        _ = test_list[10]

    with pytest.raises(IndexError):
        _ = test_list[-10]

    # Value not found
    with pytest.raises(ValueError):
        test_list.index(999)

    # Invalid slice indices (should work, just return empty or partial)
    assert test_list[10:20] == []
    assert test_list[-100:100] == [1, 2, 3]


def test_sequence_with_different_types():
    """Test sequences with various data types."""
    # Mixed types
    mixed = [1, "hello", 3.14, True, None]
    assert len(mixed) == 5
    assert mixed[0] == 1
    assert mixed[1] == "hello"
    assert mixed[2] == 3.14
    assert mixed[3] is True
    assert mixed[4] is None

    # Nested sequences
    nested = [[1, 2], [3, 4], [5, 6]]
    assert len(nested) == 3
    assert nested[0] == [1, 2]
    assert nested[1][1] == 4


def test_sequence_functional_operations():
    """Test functional-style operations on sequences."""
    test_list = [1, 2, 3, 4, 5]

    # Map
    doubled = list(map(lambda x: x * 2, test_list))
    assert doubled == [2, 4, 6, 8, 10]

    # Filter
    evens = list(filter(lambda x: x % 2 == 0, test_list))
    assert evens == [2, 4]

    # Reduce (sum)
    from functools import reduce

    total = reduce(lambda acc, x: acc + x, test_list, 0)
    assert total == 15

    # Any
    assert any(x > 3 for x in test_list)
    assert not any(x > 10 for x in test_list)

    # All
    assert all(x > 0 for x in test_list)
    assert not all(x > 2 for x in test_list)


def test_sequence_immutable_tuple():
    """Test that tuples are immutable sequences."""
    test_tuple = (1, 2, 3, 4, 5)

    # All read operations should work
    assert len(test_tuple) == 5
    assert test_tuple[0] == 1
    assert test_tuple[-1] == 5
    assert 3 in test_tuple
    assert test_tuple.index(3) == 2
    assert test_tuple.count(1) == 1

    # Mutation should fail
    with pytest.raises(TypeError):
        test_tuple[0] = 10  # type: ignore

    with pytest.raises(TypeError):
        del test_tuple[0]  # type: ignore

    # But concatenation and repetition should work (create new tuples)
    assert test_tuple + (6, 7) == (1, 2, 3, 4, 5, 6, 7)
    assert test_tuple * 2 == (1, 2, 3, 4, 5, 1, 2, 3, 4, 5)


def test_sequence_performance_characteristics():
    """Test that sequence operations have expected behavior."""
    # Large list
    large_list = list(range(10000))

    # Length should be constant time
    assert len(large_list) == 10000

    # Index access should be constant time
    assert large_list[5000] == 5000
    assert large_list[-1] == 9999

    # Slicing creates new list
    slice_result = large_list[1000:2000]
    assert len(slice_result) == 1000
    assert slice_result[0] == 1000

    # Count and index are O(n)
    assert large_list.count(5000) == 1
    assert large_list.index(5000) == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
