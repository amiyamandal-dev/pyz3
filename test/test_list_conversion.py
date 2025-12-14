"""Tests for PyList <=> Zig array/slice conversion"""
import pytest


def test_sum_list(example):
    """Test converting Python list to Zig slice"""
    result = example.list_conversion_example.sum_list([1, 2, 3, 4, 5])
    assert result == 15


def test_create_range(example):
    """Test creating Python list from Zig array"""
    result = example.list_conversion_example.create_range(0, 5)
    assert result == [0, 1, 2, 3, 4]


def test_scale_vector(example):
    """Test list transformation"""
    result = example.list_conversion_example.scale_vector(
        [1.0, 2.0, 3.0],
        2.5
    )
    assert result == [2.5, 5.0, 7.5]


def test_dot_product(example):
    """Test dot product of two vectors"""
    result = example.list_conversion_example.dot_product(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    )
    assert result == 32.0  # 1*4 + 2*5 + 3*6


def test_minmax(example):
    """Test finding min and max"""
    result = example.list_conversion_example.minmax(
        [3.5, 1.2, 5.8, 2.1, 4.6]
    )
    assert result["min"] == 1.2
    assert result["max"] == 5.8


def test_minmax_empty_list(example):
    """Test that empty list raises error"""
    with pytest.raises(ValueError, match="List cannot be empty"):
        example.list_conversion_example.minmax([])


def test_automatic_create(example):
    """Test automatic py.create() conversion"""
    result = example.list_conversion_example.automatic_create()
    assert result == [10, 20, 30, 40, 50]


def test_automatic_extract(example):
    """Test automatic py.as() conversion"""
    result = example.list_conversion_example.automatic_extract(
        [5, 10, 15, 20]
    )
    assert result == 50


def test_sum_matrix(example):
    """Test nested list conversion"""
    result = example.list_conversion_example.sum_matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    )
    assert result == 45


def test_create_matrix(example):
    """Test creating nested lists"""
    result = example.list_conversion_example.create_matrix(2, 3)
    assert result == [
        [1, 2, 3],
        [4, 5, 6]
    ]


def test_filter_above(example):
    """Test filtering values"""
    result = example.list_conversion_example.filter_above(
        [1.5, 3.2, 2.1, 4.8, 1.9, 5.1],
        3.0
    )
    assert result == [3.2, 4.8, 5.1]


def test_dot_product_length_mismatch(example):
    """Test that mismatched lengths raise error"""
    with pytest.raises(ValueError, match="Vectors must have same length"):
        example.list_conversion_example.dot_product(
            [1.0, 2.0, 3.0],
            [4.0, 5.0]
        )


def test_empty_list(example):
    """Test empty list conversion"""
    result = example.list_conversion_example.sum_list([])
    assert result == 0


def test_single_element(example):
    """Test single element list"""
    result = example.list_conversion_example.sum_list([42])
    assert result == 42


def test_floats(example):
    """Test that passing floats to int function raises TypeError"""
    with pytest.raises(TypeError, match="expected int"):
        example.list_conversion_example.sum_list([1.5, 2.5, 3.0])
