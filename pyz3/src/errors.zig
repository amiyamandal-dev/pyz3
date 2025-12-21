const Allocator = @import("std").mem.Allocator;

pub const PyError = error{
    // PyError.PyRaised should be returned when an exception has been set but not caught in
    // the Python interpreter. This tells Pydust to return PyNULL and allow Python to raise
    // the exception to the end user.
    PyRaised,
} || Allocator.Error;
