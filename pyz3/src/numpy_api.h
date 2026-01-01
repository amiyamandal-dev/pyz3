// NumPy C API bindings for pyz3
// Comprehensive interface to NumPy's C API
// Based on NumPy 2.0+ API
//
// Performance optimizations:
//   - Minimal includes (only Python.h, NumPy headers loaded at runtime)
//   - Forward declarations to reduce compile-time dependencies
//   - Explicit enum values for better code generation
//   - Type-safe definitions with proper alignment hints

#ifndef PYZ3_NUMPY_API_H
#define PYZ3_NUMPY_API_H

// ============================================================================
// NumPy API Version Configuration
// ============================================================================

// Use NumPy 2.0+ API (latest stable)
// This disables deprecated APIs and ensures forward compatibility
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION

// Unique symbols for array and ufunc APIs
// Prevents symbol conflicts when multiple modules use NumPy
#define PY_ARRAY_UNIQUE_SYMBOL PYZ3_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL PYZ3_UFUNC_API

// ============================================================================
// Dependencies
// ============================================================================

// Include Python.h (required for PyObject and basic types)
// Note: NumPy headers are NOT included here - they're loaded at runtime
// via import_array() to avoid compile-time dependency on NumPy headers
#include <Python.h>

// ============================================================================
// Forward Declarations
// ============================================================================

// Forward declarations for NumPy opaque types
// These allow type-safe usage without including full NumPy headers
// (which would require NumPy development headers at compile time)
typedef struct PyArrayObject PyArrayObject;
typedef struct PyArray_Descr PyArray_Descr;
typedef struct PyArrayIterObject PyArrayIterObject;

// ============================================================================
// NumPy Type System
// ============================================================================

/// NumPy data type enumeration
/// Maps to NumPy's internal type numbering system
/// Values match NumPy's actual enum values for compatibility
typedef enum {
    // Integer types
    NPY_BOOL = 0,           ///< Boolean type (1 byte)
    NPY_BYTE = 1,          ///< Signed 8-bit integer
    NPY_UBYTE = 2,         ///< Unsigned 8-bit integer
    NPY_SHORT = 3,         ///< Signed 16-bit integer
    NPY_USHORT = 4,        ///< Unsigned 16-bit integer
    NPY_INT = 5,           ///< Signed 32-bit integer
    NPY_UINT = 6,          ///< Unsigned 32-bit integer
    NPY_LONG = 7,          ///< Signed 64-bit integer (platform-dependent)
    NPY_ULONG = 8,         ///< Unsigned 64-bit integer (platform-dependent)
    NPY_LONGLONG = 9,      ///< Signed 64-bit integer
    NPY_ULONGLONG = 10,    ///< Unsigned 64-bit integer
    
    // Floating point types
    NPY_FLOAT = 11,        ///< 32-bit floating point
    NPY_DOUBLE = 12,       ///< 64-bit floating point
    NPY_LONGDOUBLE = 13,   ///< Extended precision floating point
    
    // Complex types
    NPY_CFLOAT = 14,       ///< Complex 64-bit (2x float32)
    NPY_CDOUBLE = 15,      ///< Complex 128-bit (2x float64)
    NPY_CLONGDOUBLE = 16,  ///< Complex extended precision
    
    // Special types
    NPY_OBJECT = 17,       ///< Python object array
    NPY_STRING = 18,       ///< String array (deprecated, use unicode)
    NPY_UNICODE = 19,      ///< Unicode string array
    NPY_VOID = 20,         ///< Void/structured array
    
    // Date/time types
    NPY_DATETIME = 21,     ///< Datetime type
    NPY_TIMEDELTA = 22,    ///< Timedelta type
    
    // Half precision
    NPY_HALF = 23,         ///< 16-bit floating point
    
    // Sentinel values
    NPY_NTYPES = 24,       ///< Number of built-in types
    NPY_NOTYPE = 25,       ///< Invalid type marker
    
    // User-defined types start here
    NPY_USERDEF = 256      ///< Base value for user-defined types
} NPY_TYPES;

// ============================================================================
// Array Flags and Properties
// ============================================================================

/// NumPy array flags enumeration
/// These flags describe array properties and memory layout
/// Flags can be combined using bitwise OR operations
typedef enum {
    // Memory layout flags
    NPY_ARRAY_C_CONTIGUOUS = 0x0001,      ///< C-style contiguous memory layout
    NPY_ARRAY_F_CONTIGUOUS = 0x0002,      ///< Fortran-style contiguous layout
    NPY_ARRAY_OWNDATA = 0x0004,           ///< Array owns its data (can free)
    
    // Conversion flags
    NPY_ARRAY_FORCECAST = 0x0010,         ///< Force type casting if needed
    NPY_ARRAY_ENSURECOPY = 0x0020,        ///< Always create a copy
    NPY_ARRAY_ENSUREARRAY = 0x0040,       ///< Ensure result is an array
    
    // Memory properties
    NPY_ARRAY_ELEMENTSTRIDES = 0x0080,   ///< Strides are per-element
    NPY_ARRAY_ALIGNED = 0x0100,           ///< Data is properly aligned
    NPY_ARRAY_NOTSWAPPED = 0x0200,        ///< Data is in native byte order
    NPY_ARRAY_WRITEABLE = 0x0400,         ///< Array data can be modified
    
    // Special flags
    NPY_ARRAY_WRITEBACKIFCOPY = 0x2000,   ///< Write back if copy was made
    
    // Composite flags (commonly used combinations)
    NPY_ARRAY_BEHAVED = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
    ///< Well-behaved array (aligned and writeable)
    
    NPY_ARRAY_CARRAY = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
    ///< C-style array (contiguous, aligned, writeable)
    
    NPY_ARRAY_CARRAY_RO = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
    ///< C-style read-only array (contiguous, aligned)
    
    NPY_ARRAY_FARRAY = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
    ///< Fortran-style array (contiguous, aligned, writeable)
    
    NPY_ARRAY_FARRAY_RO = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED,
    ///< Fortran-style read-only array (contiguous, aligned)
    
    // Default flags
    NPY_ARRAY_DEFAULT = NPY_ARRAY_CARRAY,
    ///< Default array flags (C-style, behaved)
    
    // Input/output array flags
    NPY_ARRAY_IN_ARRAY = NPY_ARRAY_CARRAY_RO,
    ///< Input array (read-only, C-contiguous, aligned)
    
    NPY_ARRAY_OUT_ARRAY = NPY_ARRAY_CARRAY,
    ///< Output array (writeable, C-contiguous, behaved)
    
    NPY_ARRAY_INOUT_ARRAY = NPY_ARRAY_CARRAY
    ///< Input/output array (writeable, C-contiguous, behaved)
} NPY_ARRAY_FLAGS;

// ============================================================================
// Type Definitions
// ============================================================================

/// Platform-independent integer type for array dimensions and indices
/// Matches NumPy's npy_intp (signed integer large enough for array sizes)
/// On 64-bit platforms: 64-bit signed integer
/// On 32-bit platforms: 32-bit signed integer
typedef long long npy_intp;

/// Platform-independent unsigned integer type for array dimensions
/// Matches NumPy's npy_uintp (unsigned integer large enough for array sizes)
typedef unsigned long long npy_uintp;

// ============================================================================
// Compile-time Assertions (if supported)
// ============================================================================

// Ensure type sizes match expectations
// These help catch platform-specific issues at compile time
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: Use static_assert (C11) or _Static_assert
    #if __STDC_VERSION__ >= 201112L
        _Static_assert(sizeof(npy_intp) >= sizeof(void*), 
                       "npy_intp must be at least pointer-sized");
        _Static_assert(sizeof(npy_uintp) >= sizeof(void*), 
                       "npy_uintp must be at least pointer-sized");
    #endif
#endif

// ============================================================================
// Compatibility Macros
// ============================================================================

// Backward compatibility: Allow using enum names as values
// These macros ensure compatibility with code expecting integer constants
#define NPY_TYPES_ENUM NPY_TYPES
#define NPY_ARRAY_FLAGS_ENUM NPY_ARRAY_FLAGS

#endif // PYZ3_NUMPY_API_H
