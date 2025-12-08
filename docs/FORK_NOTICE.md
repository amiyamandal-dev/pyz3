# Fork Notice

This repository (pyZ3) is a hard fork of [ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust).

## Purpose of Fork

pyZ3 was created to:
1. Add built-in NumPy integration with zero-copy array access
2. Enhance cross-compilation and distribution workflows
3. Create an independent project maintained separately from the original
4. Focus on data science and scientific computing use cases

## Major Differences from Original

### New Features
- ✅ **DType System** - Type-safe dtype mapping at compile time
- ✅ **Enhanced Cross-Compilation** - Improved multi-platform wheel building
- ✅ **Updated CLI** - Renamed commands from `pydust` to `pyz3`
- ✅ **Comprehensive Documentation** - Full NumPy integration guide

### API Changes
- Package renamed from `pydust` to `pyz3`
- Module import: `@import("pyz3")` instead of `@import("pydust")`
- CLI command: `pyz3` instead of `pydust`
- Environment variable: `PYZ3_OPTIMIZE` instead of `PYDUST_OPTIMIZE`

### File Structure Changes
```
pydust/          → pyz3/
pydust.build.zig → pyz3.build.zig
ziggy-pydust-template/ → pyZ3-template/
```

## Attribution

This project is based on ziggy-pydust and maintains the same Apache 2.0 license.

**Original Authors**: Nicholas Gates and the ziggy-pydust contributors
**Original Repository**: https://github.com/fulcrum-so/ziggy-pydust

We are deeply grateful to the original authors for creating such an excellent foundation!

## Relationship to Original

This is an **independent fork** maintained separately from ziggy-pydust. While we may occasionally sync improvements from the upstream project, pyZ3 follows its own development roadmap with a focus on:

- NumPy and scientific computing integration
- Data science workflows
- Enhanced distribution tooling
- Independent feature development

## License

Apache License 2.0 (same as original)

## Contributing

Contributions to pyZ3 are welcome! Please note that:
- This is a separate project from ziggy-pydust
- Pull requests should be directed to this repository, not the original
- We maintain compatibility with Zig 0.15.x and Python 3.11+

## Links

- **Original Project**: [ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust)
- **This Fork (pyZ3)**: https://github.com/amiyamandal-dev/pyz3
- **PyPI Package**: https://pypi.org/project/pyZ3/

---

**Last Updated**: 2025-12-06
**Fork Version**: 0.1.0
