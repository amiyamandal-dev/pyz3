from example import modules
from example.modules import submod


def test_module_docstring():
    assert modules.__doc__.startswith("Zig multi-line strings make it easy to define a docstring...")


def test_modules_function():
    assert modules.hello("Nick") == "Hello, Nick. It's Ziggy"


def test_modules_state():
    assert modules.whoami() == "Ziggy"


def test_modules_mutable_state():
    assert modules.count() == 0
    modules.increment()
    modules.increment()
    assert modules.count() == 2


def test_submodules():
    assert submod.world() == "Hello, World!"
