# Welcome to pyZ3

**pyZ3** is a high-performance framework for building Python extension modules in Zig. It provides seamless Python-Zig interoperability, first-class NumPy integration, and a powerful CLI to streamline your development workflow.

This documentation will guide you through everything from getting started to advanced performance optimization and distribution.

---

## 🚀 Getting Started

New to pyZ3? Start here.

- **[Introduction & Setup](getting_started.md)**: A guide to installing prerequisites and setting up your first project.
- **[Quick Start Guide](guides/quick_start.md)**: A 5-minute guide to creating and running your first extension.
- **[CLI Reference](CLI.md)**: A complete reference for all command-line interface commands.

---

## 📖 User Guides

Dive deeper into specific features and topics.

### Core Concepts
- **[Modules](guides/modules.md)**: Learn how to structure your Zig code as Python modules.
- **[Functions](guides/functions.md)**: Define Python-callable functions with automatic type conversion.
- **[Classes](guides/classes.md)**: Build Python classes in Zig with methods and properties.
- **[Exceptions](guides/exceptions.md)**: Handle errors and raise Python exceptions from Zig.
- **[Testing](guides/testing.md)**: Write and run tests for your extensions using Pytest.

### Advanced Topics
- **[NumPy Integration](guides/numpy.md)**: Leverage zero-copy array access for high-performance data science.
- **[Memory Management](guides/memory.md)**: Understand reference counting and pyZ3's memory model.
- **[Buffer Protocol](guides/buffers.md)**: Work with Python's buffer protocol for efficient data exchange.
- **[GIL Management](guides/gil.md)**: Release the GIL for true parallel processing in CPU-bound tasks.
- **[C/C++ Integration](guides/c_cpp_integration.md)**: Use existing C/C++ libraries in your pyZ3 project.
- **[Dependency Management](guides/dependency_management.md)**: Manage external C/C++ dependencies automatically.

### `zigimport` - The Magic Import Hook
- **[Intro to zigimport](guides/zigimport_readme.md)**: Import `.zig` files directly from Python, no manual compilation needed.
- **[Advanced zigimport](guides/zigimport_advanced.md)**: Explore dependency tracking, watch mode, and remote caching.

### Distribution
- **[Distribution Quick Start](guides/distribution_quickstart.md)**: A fast track to building and publishing your package.
- **[Full Distribution Guide](guides/distribution.md)**: In-depth guide to cross-compilation and wheel building.

---

## 👨‍💻 For Contributors

Interested in contributing to pyZ3?

- **[Development Setup](../DEVELOPMENT.md)**: A complete guide to setting up a development environment for pyz3 itself.
- **[Repository Structure](REPOSITORY_STRUCTURE.md)**: An overview of the project's layout.
- **[Roadmap](ROADMAP.md)**: See what's next for pyZ3.
- **[Implementation Notes](development/README.md)**: Read the technical summaries behind pyZ3's features.

---

## 🆘 Resources

- **[Quick Reference](QUICK_REFERENCE.md)**: A cheat sheet for common commands and code patterns.
- **[Examples](examples/)**: A directory full of working examples.
- **[GitHub Repository](https://github.com/amiyamandal-dev/pyz3)**: Report issues, ask questions, and contribute.