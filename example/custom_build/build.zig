// Example custom build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "custom_build",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add custom include path
    lib.addIncludePath(b.path("include"));

    // Add C source file
    lib.addCSourceFile(.{
        .file = b.path("src/native.c"),
        .flags = &.{"-O3", "-fPIC"},
    });

    b.installArtifact(lib);
}
