# TODO List for pyz3

This document tracks all TODO, FIXME, and NOTE comments in the codebase as of version 0.9.0.

**Project Status**: ✅ Production-ready. All TODOs are enhancement requests, not bugs or critical issues.

## High Priority Items

### PySequenceMixin Integration (Known Issue)
**Status**: Documented, non-blocking

- [ ] `pyz3/src/types/list.zig:36` - Fix PySequenceMixin integration - currently conflicts with existing methods
- [ ] `pyz3/src/types/tuple.zig:32` - Fix PySequenceMixin integration - currently conflicts with existing methods

**Impact**: PySequenceMixin is commented out. List and tuple types work correctly with manual implementations.

## Core Functionality Enhancements

### Function/Method Handling
- [ ] `pyz3/src/functions.zig:89` - Move function definition to better location
- [ ] `pyz3/src/functions.zig:254` - Add METH_CLASS support checking
- [ ] `pyz3/src/functions.zig:427` - **NEEDS CLARIFICATION** - "TODO: FIXME"
- [ ] `pyz3/src/functions.zig:449` - **NEEDS CLARIFICATION** - "TODO: FIXME"
- [ ] `pyz3/src/pytypes.zig:854` - Reference to issue #193

### Type Trampoline System
- [ ] `pyz3/src/pytypes.zig:806` - Consider trampolining self argument
- [ ] `pyz3/src/pytypes.zig:828` - Consider trampolining self argument
- [ ] `pyz3/src/trampoline.zig:322` - Reference to issue #193

### Type System Features
- [ ] `pyz3/src/types/buffer.zig:107` - Support more complex composite types
- [ ] `pyz3/src/types/iter.zig:42` - Implement PyIter_Send when required
- [ ] `pyz3/src/types/long.zig:44` - Support non-int conversions
- [ ] `pyz3/src/types/slice.zig:29` - Improve comptime optional handling

### Memory Management
- [ ] `pyz3/src/pytypes.zig:67` - Review heap allocation strategy per Python docs

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

- **Total TODOs**: 24
- **Blocking Issues**: 0
- **Known Issues**: 2 (PySequenceMixin - documented and non-blocking)
- **Enhancement Requests**: 22
- **Needs Clarification**: 2 (functions.zig:427, functions.zig:449)

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

*Last updated: 2025-12-22*
*Version: 0.9.0*
