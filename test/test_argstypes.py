from example import args_types


def test_zigstruct():
    arg = {"foo": 1234, "bar": True}
    assert args_types.zigstruct(arg)
