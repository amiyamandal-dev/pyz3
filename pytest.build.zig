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
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.exceptions",
        .root_source_file = b.path("example/exceptions.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.hello",
        .root_source_file = b.path("example/hello.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.gil",
        .root_source_file = b.path("example/gil.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.memory",
        .root_source_file = b.path("example/memory.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.modules",
        .root_source_file = b.path("example/modules.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.pytest",
        .root_source_file = b.path("example/pytest.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.result_types",
        .root_source_file = b.path("example/result_types.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.functions",
        .root_source_file = b.path("example/functions.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.classes",
        .root_source_file = b.path("example/classes.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.buffers",
        .root_source_file = b.path("example/buffers.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.iterators",
        .root_source_file = b.path("example/iterators.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.operators",
        .root_source_file = b.path("example/operators.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.code",
        .root_source_file = b.path("example/code.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.new_container_types",
        .root_source_file = b.path("example/new_container_types.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.c_integration",
        .root_source_file = b.path("example/c_integration.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{ "example/c_math_helper.c" },
        .c_include_dirs = &.{ "example/" },
        .c_libraries = &.{},
        .c_flags = &.{ "-O2" },
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.fastpath_bench",
        .root_source_file = b.path("example/fastpath_bench.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.gil_bench",
        .root_source_file = b.path("example/gil_bench.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.simd_example",
        .root_source_file = b.path("example/simd_example.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.native_collections_example",
        .root_source_file = b.path("example/native_collections_example.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{ "pyz3/src/native/native_dict.c", "pyz3/src/native/native_array.c" },
        .c_include_dirs = &.{ "pyz3/src/native/" },
        .c_libraries = &.{},
        .c_flags = &.{ "-std=c99" },
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.list_conversion_example",
        .root_source_file = b.path("example/list_conversion_example.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });

    _ = pyz3.addPythonModule(.{
        .name = "example.opaque_state",
        .root_source_file = b.path("example/opaque_state.zig"),
        .limited_api = true,
        .target = target,
        .optimize = optimize,
        .c_sources = &.{},
        .c_include_dirs = &.{},
        .c_libraries = &.{},
        .c_flags = &.{},
        .ld_flags = &.{},
    });


}

