const py = @import("pyz3");

const root = @This();

// --8<-- [start:function]
pub fn double(args: struct { x: i64 }) i64 {
    return args.x * 2;
}
// --8<-- [end:function]

// --8<-- [start:kwargs]
pub fn with_kwargs(args: struct { x: f64, y: f64 = 42.42 }) f64 {
    return if (args.x < args.y) args.x * 2 else args.y;
}
// --8<-- [end:kwargs]

// --8<-- [start:varargs]
pub fn variadic(args: struct { hello: py.PyString, args: py.Args(), kwargs: py.Kwargs() }) !py.PyString {
    return py.PyString.createFmt(
        "Hello {s} with {} varargs and {} kwargs",
        .{ try args.hello.asSlice(), args.args.len, args.kwargs.count() },
    );
}
// --8<-- [end:varargs]

comptime {
    py.rootmodule(root);
}
