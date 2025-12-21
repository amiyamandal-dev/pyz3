from pathlib import Path

from example import code


def test_line_no():
    assert code.line_number() == 7
    assert code.first_line_number() == 6


def test_function_name():
    assert code.function_name() == "test_function_name"


def test_file_name():
    assert Path(code.file_name()).name == "test_code.py"
