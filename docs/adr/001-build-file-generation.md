# ADR 001: Build File Generation Strategy

**Status:** Accepted
**Date:** 2025-12-22
**Decision Makers:** pyz3 Core Team

## Context

pyz3 needs to integrate seamlessly with both Zig's build system and Python's packaging ecosystem. Users should be able to build Python extension modules written in Zig without understanding the complexities of either build system.

### Problems

1. **Zig Build System**: Requires a `build.zig` file at the project root
2. **Python Packaging**: Expects standard Python packaging tools (pip, setuptools, etc.)
3. **Version Management**: Framework updates should not break user projects
4. **IDE Support**: ZLS (Zig Language Server) needs access to build configuration

### Constraints

- Users should not need to manually maintain build configuration
- Framework updates should propagate automatically
- Must support multiple extension modules in one project
- IDE tooling should work out of the box

## Decision

We use a **two-tier build file system**:

### Source Files (Framework)
- `pyz3/src/pyz3.build.zig` - Canonical build logic
  - Contains all PyZ3Step definitions
  - Handles Python FFI, memory management, trampolines
  - Updated only when the framework is updated

### Generated Files (Project)
- `pyz3.build.zig` - Generated at build time (root of user project)
  - Copied from `pyz3/src/pyz3.build.zig`
  - Provides Zig build system integration
  - Added to `.gitignore` - not tracked in version control

- `build.zig` - Generated per-project configuration
  - Created from `pyproject.toml` configuration
  - Imports `pyz3.build.zig` for framework functions
  - Self-managed mode allows manual customization

### Build Process

```
1. User runs: pip install -e . (or similar)
2. pyz3/buildzig.py:zig_build() is invoked
3. Check if pyz3.build.zig needs update (hash comparison)
4. If changed: copy pyz3/src/pyz3.build.zig → pyz3.build.zig
5. Generate build.zig from pyproject.toml (if not self-managed)
6. Run: zig build
```

### Caching Optimization

To avoid unnecessary file operations:
- Hash-based change detection before copying
- Only regenerate when source content changes
- Cache miss triggers full copy + rebuild

## Consequences

### Positive ✅

- **Separation of Concerns**: Framework code separate from user code
- **Automatic Updates**: Users get framework updates without manual changes
- **IDE Support**: ZLS can read generated `pyz3.build.zig` for completion
- **Performance**: Hash-based caching avoids redundant file operations
- **Flexibility**: Self-managed mode for advanced users

### Negative ❌

- **Confusion**: New contributors see two `pyz3.build.zig` files
- **Documentation Burden**: Must explain the two-tier system
- **Git Confusion**: Users might accidentally track generated files

### Mitigations

1. **Clear `.gitignore` entries**: Auto-generated files are ignored
2. **Logging**: Build system shows when files are copied vs cached
3. **Documentation**: This ADR explains the rationale
4. **Comments**: Generated files have headers explaining their origin

## Alternatives Considered

### 1. Single Build File (Rejected)
**Approach**: One `pyz3.build.zig` tracked in git

**Pros:**
- Simpler mental model
- No file generation needed

**Cons:**
- Framework updates break user projects
- No way to update build logic without breaking changes
- Users must manually sync with framework

### 2. Build Script in Python Only (Rejected)
**Approach**: Generate everything at runtime in Python

**Pros:**
- No Zig files to manage
- Pure Python workflow

**Cons:**
- ZLS cannot provide IDE support
- Cannot leverage Zig's build caching
- Slower builds (no incremental compilation)

### 3. Zig Package Manager (Future Consideration)
**Approach**: Use Zig's native package manager when stable

**Pros:**
- Idiomatic Zig workflow
- Better dependency management

**Cons:**
- Not yet stable in Zig 0.14
- Harder Python integration
- May revisit in Zig 1.0

## References

- [Zig Build System Documentation](https://ziglang.org/learn/build-system/)
- [Python Extension Modules](https://docs.python.org/3/extending/extending.html)
- [ZLS (Zig Language Server)](https://github.com/zigtools/zls)

## Revision History

- 2025-12-22: Initial decision
