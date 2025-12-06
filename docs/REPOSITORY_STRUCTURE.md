# Repository Structure

Clean, organized structure for the pyz3 framework.

## Root Directory

```
ziggy-pydust/
├── README.md                    # Main project documentation
├── LICENSE                      # Apache 2.0 license
├── pyproject.toml              # Python package configuration
├── poetry.lock                  # Locked dependencies
├── build.zig                    # Main build configuration
├── pytest.build.zig             # Pytest-specific build config
├── pydust.build.zig            # Pydust build helper
├── pyconf.dummy.zig            # Dummy config for testing
├── renovate.json               # Dependency update config
├── mkdocs.yml                  # Documentation site config
└── run_all_tests.sh            # Comprehensive test runner
```

## Source Code (`pydust/`)

Core framework implementation:

```
pydust/
├── __init__.py                 # Python package entry point
├── __main__.py                 # CLI entry point
├── init.py                     # Project initialization (cookiecutter)
├── deploy.py                   # PyPI deployment utilities
├── develop.py                  # Development mode utilities
├── watch.py                    # File watching and hot reload
├── wheel.py                    # Wheel building for distribution
├── deps.py                     # C/C++ dependency management
├── buildzig.py                 # Zig build integration
├── config.py                   # Configuration management
├── pytest_plugin.py            # Pytest integration
└── src/                        # Zig source code
    ├── pydust.zig              # Main Zig module
    ├── conversions.zig         # Type conversions
    ├── functions.zig           # Function wrapping
    ├── trampoline.zig          # Call trampolines
    ├── mem.zig                 # Memory management
    ├── gil.zig                 # GIL handling
    ├── builtins.zig            # Python builtins
    ├── errors.zig              # Error handling
    └── types/                  # Python type implementations
        ├── obj.zig             # PyObject base
        ├── str.zig             # PyString
        ├── int.zig             # PyLong
        ├── float.zig           # PyFloat
        ├── list.zig            # PyList
        ├── dict.zig            # PyDict
        ├── tuple.zig           # PyTuple
        └── ...                 # Other types
```

## Template (`ziggy-pydust-template/`)

Cookiecutter template for new projects:

```
ziggy-pydust-template/
├── cookiecutter.json           # Template variables
├── hooks/
│   └── post_gen_project.py    # Post-generation setup
└── {{cookiecutter.project_slug}}/
    ├── .github/workflows/      # CI/CD templates
    ├── .vscode/                # VSCode config templates
    ├── src/                    # Zig source templates
    ├── test/                   # Test templates
    ├── pyproject.toml          # Project config template
    ├── build.py                # Build script template
    └── README.md               # Project README template
```

## Tests (`test/`)

Comprehensive test suite:

```
test/
├── test_hello.py               # Basic functionality tests
├── test_classes.py             # Class tests
├── test_functions.py           # Function tests
├── test_memory.py              # Memory management tests
├── test_gil.py                 # GIL tests
├── test_exceptions.py          # Error handling tests
├── test_new_types.py           # New type tests
├── test_new_features.py        # Feature integration tests
├── test_init_deploy.py         # CLI and deployment tests
└── test_debugging.py           # Debugging tools tests
```

## Examples (`example/`)

Reference implementations:

```
example/
├── hello.zig                   # Basic hello world
├── functions.zig               # Function examples
├── classes.zig                 # Class examples
├── exceptions.zig              # Error handling
├── memory.zig                  # Memory management
├── gil.zig                     # GIL usage
├── buffers.zig                 # Buffer protocol
├── iterators.zig               # Iterator protocol
├── operators.zig               # Operator overloading
└── new_container_types.zig     # Advanced containers
```

## Documentation (`docs/`)

User-facing documentation:

```
docs/
├── index.md                    # Documentation home
├── getting_started.md          # Getting started guide
├── CLI.md                      # CLI reference
├── DEPENDENCY_MANAGEMENT.md    # C/C++ integration guide
├── distribution.md             # Distribution guide
├── DISTRIBUTION_QUICKSTART.md  # Quick distribution guide
├── ROADMAP.md                  # Future plans
├── guide/                      # Detailed guides
│   ├── intro.md
│   ├── functions.md
│   ├── classes.md
│   └── ...
└── development/                # Developer documentation
    ├── README.md               # Development docs overview
    ├── FIXES_APPLIED.md        # Recent fixes
    ├── SECURITY_*.md           # Security documentation
    ├── *_IMPLEMENTATION.md     # Implementation notes
    └── *_SUMMARY.md            # Feature summaries
```

## Configuration Files

### Python Configuration

- **pyproject.toml** - Package metadata, dependencies, build config
- **poetry.lock** - Locked dependency versions
- **.pypirc.template** - PyPI configuration template

### Zig Configuration

- **build.zig** - Main Zig build script
- **pydust.build.zig** - Pydust-specific build helpers
- **pytest.build.zig** - Test build configuration

### Development Tools

- **.vscode/** - VSCode workspace settings and launch configs
- **.gdbinit** - GDB debugger initialization
- **.lldbinit** - LLDB debugger initialization
- **.gitignore** - Git ignore patterns

### CI/CD

- **.github/workflows/** - GitHub Actions workflows
  - `ci.yml` - Continuous integration
  - `publish.yml` - PyPI publishing

## Build Artifacts (Ignored)

These directories are created during build but not committed:

```
.zig-cache/                     # Zig build cache
.venv/                          # Python virtual environment
.pytest_cache/                  # Pytest cache
zig-out/                        # Build output
dist/                           # Distribution packages
__pycache__/                    # Python bytecode
*.so, *.dylib, *.dll           # Compiled extensions
```

## Key Design Principles

1. **Clear Separation**
   - User docs in `docs/`
   - Developer docs in `docs/development/`
   - Examples separate from tests

2. **Single Source of Truth**
   - Template in `ziggy-pydust-template/`
   - No duplicate template systems
   - Cookiecutter for all project generation

3. **Modular Organization**
   - CLI commands in separate files
   - Zig types in `pydust/src/types/`
   - Clear module boundaries

4. **Developer Experience**
   - Comprehensive examples
   - Detailed documentation
   - Debugger configurations included
   - Watch mode for rapid iteration

## File Naming Conventions

- **Python files**: `lowercase_with_underscores.py`
- **Zig files**: `lowercase.zig`
- **Documentation**: `UPPERCASE_FOR_MAJOR.md`, `lowercase_for_guides.md`
- **Tests**: `test_*.py` for pytest discovery
- **Scripts**: `*.sh` for shell scripts

## Adding New Features

When adding features:

1. **Implementation**
   - Add code to appropriate module in `pydust/` or `pydust/src/`
   - Create tests in `test/`
   - Add example in `example/`

2. **Documentation**
   - User guide in `docs/`
   - Implementation notes in `docs/development/`
   - Update README.md if it affects quick start

3. **Template**
   - Update `ziggy-pydust-template/` if affects new projects
   - Test template generation

4. **Testing**
   - Add to `run_all_tests.sh` if needed
   - Ensure pytest discovers new tests

## Maintenance

- **Keep docs/ clean** - User-facing only
- **Archive in development/** - Implementation details
- **Update README** - Reflect current capabilities
- **Version updates** - Keep dependencies current via Renovate

---

**Last Updated:** 2025-12-05
**Status:** ✅ Clean and organized
