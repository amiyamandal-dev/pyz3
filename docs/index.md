# pyz3 Documentation Index

Welcome to the pyz3 documentation! This index helps you find what you need quickly.

## 📚 Getting Started

- **[Main README](../README.md)** - Project overview, installation, and quick start
- **[Quick Start Guide](guides/QUICK_START.md)** - Get up and running in 5 minutes
- **[Getting Started](getting_started.md)** - Detailed setup and configuration

## 🔧 User Guides

### Core Features
- **[Modules](guide/modules.md)** - Creating and using Python modules in Zig
- **[Functions](guide/functions.md)** - Defining Python-callable functions
- **[Classes](guide/classes.md)** - Building Python classes in Zig
- **[Exceptions](guide/exceptions.md)** - Error handling and exceptions
- **[GIL Management](guide/gil.md)** - Working with Python's Global Interpreter Lock
- **[Memory Management](guide/_5_memory.md)** - Memory safety and lifecycle
- **[Buffer Protocol](guide/_6_buffers.md)** - Zero-copy data exchange
- **[NumPy Integration](guide/numpy.md)** - Working with NumPy arrays

### Advanced Features
- **[zigimport - Auto Import](guides/ZIGIMPORT_README.md)** - Import .zig files like Python modules
- **[zigimport Advanced Features](guides/ZIGIMPORT_ADVANCED.md)** - Dependency tracking, watch mode, caching
- **[zigimport Implementation](guides/ZIGIMPORT_COMPLETE.md)** - Implementation details and architecture

## 👨‍💻 Development

- **[DEVELOPMENT.md](../DEVELOPMENT.md)** - Development setup and workflow
- **[Implementation Summary](development/IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[Reorganization Summary](development/REORGANIZATION_SUMMARY.md)** - Project reorganization notes

## 🆘 Troubleshooting

Common issues and solutions:
1. **Build errors** → See [DEVELOPMENT.md](../DEVELOPMENT.md#troubleshooting)
2. **Import errors** → Check [zigimport guide](guides/ZIGIMPORT_README.md)
3. **Memory leaks** → Read [Memory Management](guide/_5_memory.md)

## 🔗 External Links

- [GitHub Repository](https://github.com/amiyamandal-dev/pyz3)
- [Zig Documentation](https://ziglang.org/documentation/master/)
- [Python C API Reference](https://docs.python.org/3/c-api/)
