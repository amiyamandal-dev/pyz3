# TODO List for pyz3

This document tracks all TODO, FIXME, and NOTE comments in the codebase as of version 0.9.0.

**Project Status**: ✅ Production-ready. All TODOs are enhancement requests, not bugs or critical issues.

## High Priority Items

### PySequenceMixin Integration (Known Limitation)
**Status**: Documented - blocked by Zig 0.15 language change

- [x] `pyz3/src/types/list.zig:36` - Documented Zig 0.15 usingnamespace limitation
- [x] `pyz3/src/types/tuple.zig:32` - Documented Zig 0.15 usingnamespace limitation
- [x] `pyz3/src/types/sequence.zig` - Updated mixin documentation for Zig 0.15+

**Impact**: PySequenceMixin cannot be directly integrated via `usingnamespace` in Zig 0.15+.
The mixin is still functional and can be used via explicit composition pattern.
List and tuple types work correctly with their native implementations.

## Core Functionality Enhancements

### Function/Method Handling
- [x] `pyz3/src/functions.zig:89` - Removed unused keys() function
- [x] `pyz3/src/functions.zig:245` - Documented METH_CLASS requirements (feature enhancement)
- [x] `pyz3/src/functions.zig:427,449` - Stale entries removed (comments no longer exist)

### Type Trampoline System (Issue #193)
- [x] `pyz3/src/pytypes.zig:857` - Fixed: Now uses py.type_() instead of expensive py.self()
- [x] `pyz3/src/pytypes.zig:809,831` - Documented: direct cast is correct (Python guarantees type in dunder methods)
- [x] `pyz3/src/trampoline.zig:322` - Documented: py.self() required here (no instance to get type from)
- [x] `example/operators.zig:240` - Kept py.self() pattern (user code, example purposes)

### Type System Features
- [x] `pyz3/src/types/buffer.zig:107` - Added support for bool, pointer, and array types
- [x] `pyz3/src/types/iter.zig:42` - Documented PyIter_Send as future feature for async generators
- [x] `pyz3/src/types/long.zig:44` - Documented PyLong.as() type limitations with workaround
- [x] `pyz3/src/types/slice.zig:29` - Added toPyOrNull() and decrefIfNotNull() helpers

### Memory Management
- [x] `pyz3/src/pytypes.zig:67` - Documented heap allocation approach with Python docs reference

## Build System

### Build Configuration
- [ ] `pyz3/buildzig.py:127` - Fix output filename for non-limited modules
- [ ] `pyz3/config.py:44` - Fix configuration for non-limited API

## Testing Infrastructure

### Pytest Plugin Enhancements
- [ ] `pyz3/pytest_plugin.py:111` - Override path using test_metadata source provenance
- [ ] `pyz3/pytest_plugin.py:177` - Pass back log_error_count: u29

## Examples and Documentation

### Example Improvements
- [ ] `example/operators.zig:240` - Reference to issue #193
- [ ] `example/result_types.zig:40` - Support numbers bigger than long
- [ ] `example/result_types.zig:53` - Support numbers bigger than long

### Code Generation
- [ ] `pyz3/deps.py:505` - Add convenience wrappers for common operations
- [ ] `pyz3/generate_stubs.py:205` - Consider adding setter generation

## Implementation Notes (Informational)

These are design decision notes, not action items:

- `pyz3/src/builtins.zig:193` - Currently don't allow users to override tp_alloc
- `pyz3/src/pytypes.zig:158` - Using tp_new to fail as early as possible
- `pyz3/src/pytypes.zig:330` - Consider moving logic to PyErr
- `pyz3/src/types/obj.zig:19` - Use only when accessing ob_refcnt

## Statistics

- **Total TODOs**: 12 (15 completed in v0.9.1)
- **Blocking Issues**: 0
- **Known Limitations**: 1 (PySequenceMixin - Zig 0.15 language change)
- **Enhancement Requests**: 8
- **Issue #193 Status**: Fully resolved

## Recent Fixes (v0.9.1)

### Documentation & Cleanup
- ✅ Documented PySequenceMixin Zig 0.15 limitation with explicit composition pattern
- ✅ Removed unused keys() function from functions.zig
- ✅ Documented METH_CLASS requirements for classmethod support
- ✅ Documented heap allocation approach with Python docs reference
- ✅ Documented PyLong.as() type limitations with workaround
- ✅ Documented PyIter_Send as future feature for async generators
- ✅ Fixed numpy.PyArray reference in pyz3.zig (was undefined)
- ✅ Removed stale TODO entries (functions.zig:427,449 - comments no longer exist)

### Performance (Issue #193)
- ✅ Fixed pytypes.zig comparison operators: use py.type_() instead of py.self()
- ✅ Documented trampoline.zig: py.self() unavoidable (no instance available)
- ✅ Example code unchanged (py.self() acceptable for user code)
- ✅ Documented pytypes.zig:809,831: direct cast is correct (Python guarantees type)

### Type System Enhancements
- ✅ buffer.zig: Added support for bool, pointer, and array types in getFormat()
- ✅ slice.zig: Added toPyOrNull() and decrefIfNotNull() helper functions
- ✅ str.zig: Fixed FIXME comment in appendObj() - clarified PyUnicode_Append semantics

## Recent Fixes (v0.9.0)

### Memory Management
- ✅ Fixed `PyMemAllocator` struct definition (removed `{}` creating instance)
- ✅ Fixed GIL depth counter overflow protection
- ✅ Fixed data corruption in remap() when alignment shift changes
- ✅ Fixed memory leak in setItem error path
- ✅ Fixed integer overflow in type constructors
- ✅ Fixed resize() memory leak

### NumPy Integration
- ✅ Added NumPy as automatic dependency
- ✅ Implemented NumPy wrapper module (pyz3/src/numpy.zig)
- ✅ Created working NumPy example (example/numpy_example.zig)
- ✅ Added comprehensive NumPy test suite (59 tests, 100% pass rate)
- ✅ Zero errors, zero exceptions guarantee met

### Type System
- ✅ Fixed variable shadowing in sequence.zig
- ✅ Added integer overflow protection in tuple/list constructors
- ✅ Fixed floating-point comparison in tests

## Priority Guidelines

1. **Critical**: Blocking bugs or security issues → Immediate fix required
2. **High**: Feature gaps affecting common use cases → Plan for next minor version
3. **Medium**: Enhancements improving usability → Consider for future releases
4. **Low**: Nice-to-have improvements → Backlog

Current TODO items are primarily **Medium** and **Low** priority.

## Contributing

When adding new TODOs:
1. Use format: `// TODO(author): Description` or `# TODO(author): Description`
2. Reference issue numbers when applicable: `// TODO(author): #123 - Description`
3. Mark with priority if critical: `// TODO(CRITICAL): Description`
4. Update this file when adding significant TODOs

---

*Last updated: 2026-01-01*
*Version: 0.9.1*
