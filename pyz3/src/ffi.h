

// Export the Limited Python C API for use within PyDust.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// From 3.12 onwards, structmember.h is fixed to be including in Python.h
// See https://github.com/python/cpython/pull/99014
#include <structmember.h>
