const std = @import("std");
const py = @import("pyz3");

const root = @This();

pub fn line_number() u32 {
    return py.PyFrame(root).get().?.lineNumber();
}

pub fn function_name() !py.PyString {
    return py.PyFrame(root).get().?.code().name();
}

pub fn file_name() !py.PyString {
    return py.PyFrame(root).get().?.code().fileName();
}

pub fn first_line_number() !u32 {
    return py.PyFrame(root).get().?.code().firstLineNumber();
}

comptime {
    py.rootmodule(root);
}
