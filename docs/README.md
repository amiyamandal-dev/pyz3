# pyZ3 Documentation

Complete documentation for the pyZ3 framework - a high-performance Python extension framework in Zig.

## 📚 Documentation Structure

### 🚀 Getting Started
- **[Main README](../README.md)** - Project overview and quick start
- **[Setup Checklist](setup/SETUP_CHECKLIST.md)** - Complete setup instructions
- **[Final Setup Summary](setup/FINAL_SETUP_SUMMARY.md)** - Everything you need to know

### 📖 User Guides
- **[User Guide](guide/index.md)** - Complete framework guide
- **[NumPy Integration](guide/numpy.md)** - NumPy arrays and operations
- **[Functions](guide/functions.md)** - Writing Python functions in Zig
- **[Classes](guide/classes.md)** - Creating Python classes
- **[Exceptions](guide/exceptions.md)** - Error handling
- **[Memory Management](guide/_5_memory.md)** - Memory safety
- **[Buffer Protocol](guide/_6_buffers.md)** - Working with buffers
- **[GIL Management](guide/gil.md)** - Global Interpreter Lock
- **[Testing](guide/_4_testing.md)** - Testing your extensions

### 🔧 Development
- **[Development Guide](development/README.md)** - All implementation docs
- **[CLI Usage](development/CLI_USAGE_EXAMPLES.md)** - Command-line interface
- **[Test Guide](development/TEST_ALL_GUIDE.md)** - Running tests

### 📦 Distribution
- **[Distribution Guide](DISTRIBUTION_QUICKSTART.md)** - Package and publish
- **[Quick Release Guide](setup/QUICK_RELEASE_GUIDE.md)** - Fast release reference
- **[Automated Releases](setup/AUTOMATED_RELEASES.md)** - Auto-release system
- **[PyPI Token Setup](setup/PYPI_TOKEN_SETUP.md)** - PyPI configuration

### 🔍 Reference
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet
- **[Repository Structure](REPOSITORY_STRUCTURE.md)** - Project layout
- **[Roadmap](ROADMAP.md)** - Future plans and features
- **[CLI Reference](CLI.md)** - Complete CLI documentation

### 🔧 Fixes & Troubleshooting
- **[Build Fixes Summary](fixes/BUILD_FIXES_SUMMARY.md)** - All build fixes
- **[Python Version Fix](fixes/PYTHON_VERSION_FIX.md)** - Python 3.11+ requirement
- **[Wheel Build Fix](fixes/WHEEL_BUILD_FIX.md)** - Platform tags and auditwheel
- **[Version Attribute Fix](fixes/VERSION_ATTRIBUTE_FIX.md)** - Module version

### 📜 Project Information
- **[Fork Notice](FORK_NOTICE.md)** - Attribution and differences
- **[Rename Summary](RENAME_SUMMARY.md)** - Package rename details
- **[Cleanup Summary](CLEANUP_SUMMARY.md)** - Repository organization
- **[License](../LICENSE)** - Apache 2.0

## 🎯 Quick Links

### New to pyZ3?
1. Start with [README](../README.md)
2. Follow [Setup Checklist](setup/SETUP_CHECKLIST.md)
3. Read [User Guide](guide/index.md)
4. Try [Quick Reference](QUICK_REFERENCE.md)

### Want to Release?
1. Read [Quick Release Guide](setup/QUICK_RELEASE_GUIDE.md)
2. Use [Automated Releases](setup/AUTOMATED_RELEASES.md)
3. Configure [PyPI Setup](setup/PYPI_TOKEN_SETUP.md)

### Having Build Issues?
1. Check [Build Fixes Summary](fixes/BUILD_FIXES_SUMMARY.md)
2. Review specific fix docs in `fixes/`

### Want NumPy Integration?
1. Read [NumPy Guide](guide/numpy.md)
2. Check [Examples](../example/numpy_example.zig)
3. Review [Tests](../test/test_numpy.py)

## 📂 Directory Structure

```
docs/
├── README.md                    # This file
├── guide/                       # User guides
│   ├── index.md
│   ├── numpy.md                # NumPy integration
│   ├── functions.md
│   ├── classes.md
│   └── ...
├── setup/                       # Setup documentation
│   ├── SETUP_CHECKLIST.md
│   ├── AUTOMATED_RELEASES.md
│   ├── QUICK_RELEASE_GUIDE.md
│   ├── PYPI_TOKEN_SETUP.md
│   └── FINAL_SETUP_SUMMARY.md
├── fixes/                       # Build fix documentation
│   ├── BUILD_FIXES_SUMMARY.md
│   ├── PYTHON_VERSION_FIX.md
│   ├── WHEEL_BUILD_FIX.md
│   └── VERSION_ATTRIBUTE_FIX.md
├── development/                 # Development docs
│   ├── README.md
│   ├── CLI_USAGE_EXAMPLES.md
│   └── ...
├── FORK_NOTICE.md              # Fork attribution
├── RENAME_SUMMARY.md           # Rename documentation
├── QUICK_REFERENCE.md          # Command reference
├── REPOSITORY_STRUCTURE.md     # Project structure
├── ROADMAP.md                  # Future plans
├── CLI.md                      # CLI documentation
└── DISTRIBUTION_QUICKSTART.md  # Distribution guide
```

## 🆘 Need Help?

- **General Questions**: Check [User Guide](guide/index.md)
- **Setup Issues**: See [Setup Checklist](setup/SETUP_CHECKLIST.md)
- **Build Errors**: Review [Build Fixes](fixes/BUILD_FIXES_SUMMARY.md)
- **Release Problems**: Read [Release Guide](setup/QUICK_RELEASE_GUIDE.md)
- **NumPy Questions**: See [NumPy Guide](guide/numpy.md)

## 🔗 External Links

- **GitHub Repository**: https://github.com/yourusername/pyZ3
- **PyPI Package**: https://pypi.org/project/pyZ3/
- **Original Project**: https://github.com/fulcrum-so/ziggy-pydust
- **Zig Language**: https://ziglang.org
- **NumPy**: https://numpy.org

---

**Last Updated**: 2025-12-06
**Version**: 0.1.0
**Status**: Production Ready
