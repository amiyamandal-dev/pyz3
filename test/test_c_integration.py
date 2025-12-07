"""
Tests for C/C++ integration example.
Demonstrates calling C functions from Python via Zig wrappers.
"""

import pytest


def test_c_add():
    """Test C add function."""
    import example.c_integration

    assert example.c_integration.add(2, 3) == 5
    assert example.c_integration.add(-1, 1) == 0
    assert example.c_integration.add(100, 200) == 300


def test_c_multiply():
    """Test C multiply function."""
    import example.c_integration

    assert example.c_integration.multiply(3, 4) == 12
    assert example.c_integration.multiply(-2, 5) == -10
    assert example.c_integration.multiply(0, 100) == 0


def test_c_divide():
    """Test C divide function."""
    import example.c_integration

    assert example.c_integration.divide(10.0, 2.0) == 5.0
    assert example.c_integration.divide(7.0, 2.0) == 3.5
    assert abs(example.c_integration.divide(1.0, 3.0) - 0.333333) < 0.001


def test_c_divide_by_zero():
    """Test C divide by zero raises exception."""
    import example.c_integration

    with pytest.raises(Exception):  # ZeroDivisionError
        example.c_integration.divide(10.0, 0.0)


def test_c_factorial():
    """Test C factorial function."""
    import example.c_integration

    assert example.c_integration.factorial(0) == 1
    assert example.c_integration.factorial(1) == 1
    assert example.c_integration.factorial(5) == 120
    assert example.c_integration.factorial(10) == 3628800


def test_c_factorial_negative():
    """Test C factorial with negative number raises exception."""
    import example.c_integration

    with pytest.raises(Exception):  # ValueError
        example.c_integration.factorial(-1)


def test_c_factorial_too_large():
    """Test C factorial with too large number raises exception."""
    import example.c_integration

    with pytest.raises(Exception):  # ValueError
        example.c_integration.factorial(21)


def test_c_fibonacci():
    """Test C fibonacci function."""
    import example.c_integration

    assert example.c_integration.fibonacci(0) == 0
    assert example.c_integration.fibonacci(1) == 1
    assert example.c_integration.fibonacci(2) == 1
    assert example.c_integration.fibonacci(3) == 2
    assert example.c_integration.fibonacci(5) == 5
    assert example.c_integration.fibonacci(10) == 55


def test_c_fibonacci_negative():
    """Test C fibonacci with negative number raises exception."""
    import example.c_integration

    with pytest.raises(Exception):  # ValueError
        example.c_integration.fibonacci(-1)


def test_c_fibonacci_too_large():
    """Test C fibonacci with too large number raises exception."""
    import example.c_integration

    with pytest.raises(Exception):  # ValueError
        example.c_integration.fibonacci(31)
