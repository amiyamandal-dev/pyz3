# Cookiecutter-Only Template System

## Overview

The pyz3 project initialization system now **exclusively uses cookiecutter templates**. The legacy template system has been removed for a cleaner, more maintainable codebase.

## Changes Made

### 1. Removed Legacy Templates

**Deleted from `pyz3/init.py`:**
- All `TEMPLATE_*` constants (TEMPLATE_PYPROJECT, TEMPLATE_BUILD_ZIG, etc.)
- Old `init_project()` implementation with inline templates
- `_check_cookiecutter_available()` function (cookiecutter now required)

**Simplified to:**
- Single `init_project_cookiecutter()` function
- Alias: `init_project = init_project_cookiecutter` for backward compatibility

### 2. Updated CLI

**Removed flags:**
- `--legacy` flag (no longer needed)
- `-f, --force` flag (cookiecutter handles this)

**Retained flags:**
- `-n, --name`: Package name
- `-a, --author`: Author name
- `--email`: Author email
- `--description`: Project description
- `--no-interactive`: Non-interactive mode

### 3. Simplified Code Flow

```python
# Before (dual system):
if use_cookiecutter:
    init_project_cookiecutter(...)
else:
    init_project(...)  # legacy

# After (cookiecutter only):
init_project_cookiecutter(...)
```

## Usage

### Install cookiecutter

Cookiecutter is now a **required dependency**:

```bash
pip install cookiecutter
# or
uv pip install cookiecutter
```

### Initialize New Project (Interactive)

```bash
pyz3 init
```

You'll be prompted for:
- Project name
- Author name
- Author email
- Description
- Python version

### Initialize New Project (Non-Interactive)

```bash
pyz3 init -n myproject \
  --description "My awesome Zig extension" \
  --email "me@example.com" \
  --no-interactive
```

### Create New Project in Directory

```bash
pyz3 new myproject
# Creates: ./myproject/
```

With custom parent directory:

```bash
pyz3 new myproject -p /path/to/parent
# Creates: /path/to/parent/myproject/
```

## Generated Project Structure

```
myproject/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Automated testing
│       └── publish.yml      # PyPI publishing
├── .vscode/
│   ├── extensions.json      # Recommended extensions
│   └── launch.json          # Debug configuration
├── src/
│   └── myproject.zig        # Zig source code
├── myproject/
│   ├── __init__.py          # Python package
│   └── _lib.pyi             # Type stubs
├── test/
│   ├── __init__.py
│   └── test_myproject.py    # Tests
├── .gitignore
├── build.py                 # Build script
├── LICENSE
├── pyproject.toml           # Project configuration
├── README.md                # Documentation
└── renovate.json            # Dependency updates
```

## Benefits of Cookiecutter-Only

### 1. **Consistency**
- Single source of truth for project templates
- All projects use the same modern structure
- Easier to maintain and update templates

### 2. **Rich Features**
- CI/CD pipelines out of the box
- VSCode integration
- Type stubs for IDE support
- Comprehensive examples (Fibonacci implementation)

### 3. **Simpler Codebase**
- Removed ~400 lines of template constants
- Eliminated dual-system complexity
- Clearer code flow

### 4. **Better Maintenance**
- Templates in separate files (easier to edit)
- Post-generation hooks for automation
- Version control for template changes

### 5. **Modern Workflow**
- Automatic git initialization
- Smart tool detection (uv, Poetry, pip)
- Context-aware next steps

## Template Location

```
pyZ3/
├── pyz3/
│   ├── __main__.py         # CLI
│   └── init.py             # Template integration
│
└── pyZ3-template/  # Cookiecutter template
    ├── cookiecutter.json
    ├── hooks/
    │   └── post_gen_project.py
    └── {{cookiecutter.project_slug}}/
        └── ... template files ...
```

## Development Workflow

### 1. Initialize Project

```bash
cd ~/projects
pyz3 init -n myextension --no-interactive
cd myextension
```

### 2. Set Up Environment (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Develop

```bash
# Edit src/myextension.zig
pyz3 develop

# Run tests
pytest

# Watch mode
pyz3 watch --test
```

### 4. Build & Deploy

```bash
# Build wheels
pyz3 build-wheel --all-platforms

# Validate
pyz3 check --strict

# Deploy to PyPI
pyz3 deploy --username __token__ --password $PYPI_TOKEN
```

## Template Variables

Defined in `pyZ3-template/cookiecutter.json`:

| Variable | Example | Used For |
|----------|---------|----------|
| `project_name` | "My Project" | Human-readable name |
| `project_slug` | "my-project" | PyPI package name |
| `package_name` | "my_project" | Python import name |
| `zig_file_name` | "my_project" | Zig source file |
| `module_name` | "_lib" | Compiled module |
| `author_name` | "John Doe" | Package author |
| `author_email` | "john@example.com" | Contact email |
| `description` | "..." | Project description |
| `version` | "0.1.0" | Initial version |
| `python_version` | "3.11" | Min Python version |

## Post-Generation Hook

After project creation, `hooks/post_gen_project.py` automatically:

1. **Initializes Git**
   ```bash
   git init
   git add .
   git commit -m "Initial commit from pyZ3-template"
   ```

2. **Detects Tools**
   - Checks for `uv`, `Poetry`, `pip`
   - Provides appropriate next steps

3. **Prints Instructions**
   ```
   Next steps:
     uv venv
     source .venv/bin/activate
     uv pip install -e .
     pytest
   ```

## Error Handling

### Cookiecutter Not Installed

```bash
$ pyz3 init
❌ Error: cookiecutter is required to initialize projects.

To install cookiecutter:
  pip install cookiecutter
  # or
  uv pip install cookiecutter
```

### Template Not Found

```bash
❌ Error: Template directory not found at /path/to/pyZ3-template

Please ensure pyZ3-template is in the repository root.
```

### Generation Failed

```bash
❌ Error: Failed to initialize project: <error details>
<traceback>
```

## Migration Guide

### For Users

If you were using the legacy templates:

**Before:**
```bash
pyz3 init --legacy -n myproject -a "Name <email>" -f
```

**After:**
```bash
# Install cookiecutter first
uv pip install cookiecutter

# Use regular init (no --legacy flag)
pyz3 init -n myproject --email "email" --no-interactive
```

### For Developers

If you were calling `init.init_project()` directly:

**Before:**
```python
from pyz3.init import init_project
init_project(path, package_name, author, force=True)
```

**After:**
```python
from pyz3.init import init_project_cookiecutter

# More parameters available
init_project_cookiecutter(
    path=path,
    package_name=package_name,
    author_name="Name",
    author_email="email@example.com",
    description="My project",
    use_interactive=False,
)

# Or use alias
from pyz3.init import init_project
init_project(...)  # Same as init_project_cookiecutter
```

## Testing

Tests in `test/test_init_deploy.py` now:

- Skip tests if cookiecutter not installed
- Verify template directory exists
- Test non-interactive project generation
- Check generated project structure

Run tests:

```bash
# Install cookiecutter for full test coverage
uv pip install cookiecutter

# Run tests
pytest test/test_init_deploy.py -v
```

## Summary

✅ **Removed:** 400+ lines of legacy template code
✅ **Simplified:** Single template system
✅ **Required:** cookiecutter dependency
✅ **Enhanced:** Modern project structure with CI/CD
✅ **Maintained:** Backward compatibility via alias

The cookiecutter-only approach provides a cleaner, more maintainable system while delivering better project templates to users.
