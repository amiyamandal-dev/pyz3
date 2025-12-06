const std = @import("std");
const py = @import("./pyz3.build.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptionsQueryOnly(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run library tests");

    const pyz3 = py.addPyZ3(b, .{
        .test_step = test_step,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.args_types",
        .root_source_file = b.path("example/args_types.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.exceptions",
        .root_source_file = b.path("example/exceptions.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.hello",
        .root_source_file = b.path("example/hello.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.gil",
        .root_source_file = b.path("example/gil.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.memory",
        .root_source_file = b.path("example/memory.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.modules",
        .root_source_file = b.path("example/modules.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.pytest",
        .root_source_file = b.path("example/pytest.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.result_types",
        .root_source_file = b.path("example/result_types.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.functions",
        .root_source_file = b.path("example/functions.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.classes",
        .root_source_file = b.path("example/classes.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.buffers",
        .root_source_file = b.path("example/buffers.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.iterators",
        .root_source_file = b.path("example/iterators.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.operators",
        .root_source_file = b.path("example/operators.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.code",
        .root_source_file = b.path("example/code.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.new_container_types",
        .root_source_file = b.path("example/new_container_types.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
    });


}

