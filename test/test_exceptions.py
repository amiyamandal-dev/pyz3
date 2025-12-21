import pytest

from example import exceptions


def test_exceptions():
    with pytest.raises(ValueError) as exc:
        exceptions.raise_value_error("hello!")
    assert str(exc.value) == "hello!"


def test_custom_error():
    with pytest.raises(RuntimeError) as exc:
        exceptions.raise_custom_error()
    assert str(exc.value) == "Oops"
