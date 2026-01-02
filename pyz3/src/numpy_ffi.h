// NumPy C API FFI bindings for pyz3
// This header provides direct access to NumPy's C API structures and functions
//
// Licensed under the Apache License, Version 2.0

#ifndef PYZ3_NUMPY_FFI_H
#define PYZ3_NUMPY_FFI_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

// ============================================================================
// NumPy Type Definitions
// ============================================================================

// npy_intp is the same as Py_ssize_t (pointer-sized signed integer)
typedef Py_ssize_t npy_intp;
typedef size_t npy_uintp;
typedef uint64_t npy_uint64;

// ============================================================================
// NumPy Data Type Enumeration
// ============================================================================

typedef enum {
    NPY_BOOL = 0,
    NPY_BYTE = 1,
    NPY_UBYTE = 2,
    NPY_SHORT = 3,
    NPY_USHORT = 4,
    NPY_INT = 5,
    NPY_UINT = 6,
    NPY_LONG = 7,
    NPY_ULONG = 8,
    NPY_LONGLONG = 9,
    NPY_ULONGLONG = 10,
    NPY_FLOAT = 11,
    NPY_DOUBLE = 12,
    NPY_LONGDOUBLE = 13,
    NPY_CFLOAT = 14,
    NPY_CDOUBLE = 15,
    NPY_CLONGDOUBLE = 16,
    NPY_OBJECT = 17,
    NPY_STRING = 18,
    NPY_UNICODE = 19,
    NPY_VOID = 20,
    NPY_DATETIME = 21,
    NPY_TIMEDELTA = 22,
    NPY_HALF = 23,
    NPY_NTYPES = 24,
    NPY_NOTYPE = 25,
    NPY_USERDEF = 256,
    
    // Aliases for sized types
    NPY_INT8 = NPY_BYTE,
    NPY_UINT8 = NPY_UBYTE,
    NPY_INT16 = NPY_SHORT,
    NPY_UINT16 = NPY_USHORT,
    NPY_INT32 = NPY_INT,
    NPY_UINT32 = NPY_UINT,
    NPY_INT64 = NPY_LONGLONG,
    NPY_UINT64 = NPY_ULONGLONG,
    NPY_FLOAT32 = NPY_FLOAT,
    NPY_FLOAT64 = NPY_DOUBLE
} NPY_TYPES;

// ============================================================================
// NumPy Array Flags
// ============================================================================

typedef enum {
    NPY_ARRAY_C_CONTIGUOUS = 0x0001,
    NPY_ARRAY_F_CONTIGUOUS = 0x0002,
    NPY_ARRAY_OWNDATA = 0x0004,
    NPY_ARRAY_FORCECAST = 0x0010,
    NPY_ARRAY_ENSURECOPY = 0x0020,
    NPY_ARRAY_ENSUREARRAY = 0x0040,
    NPY_ARRAY_ELEMENTSTRIDES = 0x0080,
    NPY_ARRAY_ALIGNED = 0x0100,
    NPY_ARRAY_NOTSWAPPED = 0x0200,
    NPY_ARRAY_WRITEABLE = 0x0400,
    NPY_ARRAY_WRITEBACKIFCOPY = 0x2000,
    
    // Composite flags
    NPY_ARRAY_BEHAVED = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
    NPY_ARRAY_CARRAY = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED,
    NPY_ARRAY_CARRAY_RO = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
    NPY_ARRAY_FARRAY = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
    NPY_ARRAY_FARRAY_RO = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED,
    NPY_ARRAY_DEFAULT = NPY_ARRAY_CARRAY,
    NPY_ARRAY_IN_ARRAY = NPY_ARRAY_CARRAY_RO,
    NPY_ARRAY_OUT_ARRAY = NPY_ARRAY_CARRAY,
    NPY_ARRAY_INOUT_ARRAY = NPY_ARRAY_CARRAY
} NPY_ARRAY_FLAGS;

// ============================================================================
// NumPy Descriptor (dtype) Structure - NumPy 2.0 compatible
// ============================================================================

// Forward declarations
typedef struct _PyArray_Descr PyArray_Descr;
typedef struct tagPyArrayObject_fields PyArrayObject_fields;
typedef struct tagPyArrayObject_fields PyArrayObject;

// NumPy 2.0 descriptor structure
// In NumPy 2.0+, the descriptor layout changed significantly.
// The base PyArray_Descr only has fields up to type_num.
// To access elsize and alignment, we need the full _PyArray_DescrNumPy2 struct.
struct _PyArray_Descr {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;           // 'b', 'i', 'u', 'f', 'c', 'S', 'U', 'V', etc.
    char type;           // unique character for this type
    char byteorder;      // '>' big, '<' little, '|' not-applicable, '=' native
    char _former_flags;  // unused, for ABI compatibility
    int type_num;        // NPY_TYPES enum value
    // NumPy 2.0: elsize and alignment moved to extended struct below
};

// Full NumPy 2.0 descriptor with elsize and alignment
// This matches _PyArray_DescrNumPy2 from ndarraytypes.h
typedef struct {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char _former_flags;
    int type_num;
    npy_uint64 flags;       // New in NumPy 2.0
    npy_intp elsize;        // Moved and now npy_intp (was int)
    npy_intp alignment;     // Moved and now npy_intp (was int)
    PyObject *metadata;
    Py_hash_t hash;
    void *reserved_null[2];
} PyArray_DescrNumPy2;

// ============================================================================
// NumPy Array Object Structure
// ============================================================================

// This is the actual layout of PyArrayObject in memory
struct tagPyArrayObject_fields {
    PyObject_HEAD
    char *data;              // Pointer to raw data buffer
    int nd;                  // Number of dimensions (ndim)
    npy_intp *dimensions;    // Shape array
    npy_intp *strides;       // Strides array (bytes per dimension)
    PyObject *base;          // Base object (for views)
    PyArray_Descr *descr;    // Data type descriptor
    int flags;               // Array flags
    PyObject *weakreflist;   // Weak reference list
    // void *_buffer_info;   // Internal buffer info (NumPy 2.0+)
};

// ============================================================================
// NumPy C API Function Pointer Table
// ============================================================================

// The NumPy C API is accessed through a table of function pointers
// that is loaded at runtime via import_array()

// API slot indices (these are stable across NumPy versions)
#define NPY_API_PyArray_Type_NUM 2
#define NPY_API_PyArrayDescr_Type_NUM 3
#define NPY_API_PyArray_DescrFromType_NUM 45
#define NPY_API_PyArray_NewFromDescr_NUM 94
#define NPY_API_PyArray_New_NUM 93
#define NPY_API_PyArray_SimpleNew_NUM 174
#define NPY_API_PyArray_SimpleNewFromData_NUM 175
#define NPY_API_PyArray_Zeros_NUM 183
#define NPY_API_PyArray_Empty_NUM 184
#define NPY_API_PyArray_CopyInto_NUM 82
#define NPY_API_PyArray_FromAny_NUM 69
#define NPY_API_PyArray_ContiguousFromAny_NUM 70
#define NPY_API_PyArray_CheckFromAny_NUM 73
#define NPY_API_PyArray_INCREF_NUM 109
#define NPY_API_PyArray_XDECREF_NUM 110
#define NPY_API_PyArray_SetBaseObject_NUM 282
#define NPY_API_PyArray_SIZE_NUM 179
#define NPY_API_PyArray_NBYTES_NUM 180

// Function pointer types
typedef PyObject* (*PyArray_SimpleNew_t)(int nd, npy_intp const *dims, int typenum);
typedef PyObject* (*PyArray_SimpleNewFromData_t)(int nd, npy_intp const *dims, int typenum, void *data);
typedef PyObject* (*PyArray_Zeros_t)(int nd, npy_intp const *dims, PyArray_Descr *descr, int fortran);
typedef PyObject* (*PyArray_Empty_t)(int nd, npy_intp const *dims, PyArray_Descr *descr, int fortran);
typedef PyArray_Descr* (*PyArray_DescrFromType_t)(int typenum);
typedef PyObject* (*PyArray_FromAny_t)(PyObject *op, PyArray_Descr *dtype, int min_depth, int max_depth, int requirements, PyObject *context);
typedef PyObject* (*PyArray_ContiguousFromAny_t)(PyObject *op, int typenum, int min_depth, int max_depth);
typedef int (*PyArray_CopyInto_t)(PyArrayObject *dest, PyArrayObject *src);
typedef int (*PyArray_SetBaseObject_t)(PyArrayObject *arr, PyObject *base);

// ============================================================================
// Inline Accessor Functions
// ============================================================================

// These match NumPy's inline functions exactly

static inline int
pyz3_PyArray_NDIM(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->nd;
}

static inline void *
pyz3_PyArray_DATA(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->data;
}

static inline char *
pyz3_PyArray_BYTES(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->data;
}

static inline npy_intp *
pyz3_PyArray_DIMS(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->dimensions;
}

static inline npy_intp *
pyz3_PyArray_SHAPE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->dimensions;
}

static inline npy_intp *
pyz3_PyArray_STRIDES(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->strides;
}

static inline npy_intp
pyz3_PyArray_DIM(const PyArrayObject *arr, int idim)
{
    return ((PyArrayObject_fields *)arr)->dimensions[idim];
}

static inline npy_intp
pyz3_PyArray_STRIDE(const PyArrayObject *arr, int istride)
{
    return ((PyArrayObject_fields *)arr)->strides[istride];
}

static inline npy_intp
pyz3_PyArray_ITEMSIZE(const PyArrayObject *arr)
{
    // NumPy 2.0: must cast to PyArray_DescrNumPy2 to access elsize
    return ((PyArray_DescrNumPy2 *)((PyArrayObject_fields *)arr)->descr)->elsize;
}

static inline PyObject *
pyz3_PyArray_BASE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->base;
}

static inline PyArray_Descr *
pyz3_PyArray_DESCR(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->descr;
}

static inline int
pyz3_PyArray_FLAGS(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->flags;
}

static inline int
pyz3_PyArray_TYPE(const PyArrayObject *arr)
{
    return ((PyArrayObject_fields *)arr)->descr->type_num;
}

static inline int
pyz3_PyArray_CHKFLAGS(const PyArrayObject *arr, int flags)
{
    return (pyz3_PyArray_FLAGS(arr) & flags) == flags;
}

// Compute total size (number of elements)
static inline npy_intp
pyz3_PyArray_SIZE(const PyArrayObject *arr)
{
    npy_intp size = 1;
    int nd = pyz3_PyArray_NDIM(arr);
    npy_intp *dims = pyz3_PyArray_DIMS(arr);
    for (int i = 0; i < nd; i++) {
        size *= dims[i];
    }
    return size;
}

// Compute total bytes
static inline npy_intp
pyz3_PyArray_NBYTES(const PyArrayObject *arr)
{
    return pyz3_PyArray_SIZE(arr) * pyz3_PyArray_ITEMSIZE(arr);
}

// Contiguity checks
static inline int
pyz3_PyArray_IS_C_CONTIGUOUS(const PyArrayObject *arr)
{
    return pyz3_PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS);
}

static inline int
pyz3_PyArray_IS_F_CONTIGUOUS(const PyArrayObject *arr)
{
    return pyz3_PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS);
}

static inline int
pyz3_PyArray_ISCONTIGUOUS(const PyArrayObject *arr)
{
    return pyz3_PyArray_IS_C_CONTIGUOUS(arr);
}

static inline int
pyz3_PyArray_ISWRITEABLE(const PyArrayObject *arr)
{
    return pyz3_PyArray_CHKFLAGS(arr, NPY_ARRAY_WRITEABLE);
}

static inline int
pyz3_PyArray_ISALIGNED(const PyArrayObject *arr)
{
    return pyz3_PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED);
}

#endif // PYZ3_NUMPY_FFI_H
