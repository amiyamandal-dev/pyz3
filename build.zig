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

