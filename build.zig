// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const builtin = @import("builtin");
const std = @import("std");

pub fn build(b: *std.Build) void {
    // Support cross-compilation via ZIG_TARGET environment variable
    const target = getTargetFromEnv(b) orelse b.standardTargetOptions(.{});

    // Support optimization level via PYZ3_OPTIMIZE environment variable
    const optimize = getOptimizeFromEnv(b) orelse b.standardOptimizeOption(.{});

    const python_exe = b.option([]const u8, "python-exe", "Python executable to use") orelse "python";

    const pythonInc = getPythonIncludePath(python_exe, b.allocator) catch @panic("Missing python");
    const pythonLib = getPythonLibraryPath(python_exe, b.allocator) catch @panic("Missing python");
    const pythonVer = getPythonLDVersion(python_exe, b.allocator) catch @panic("Missing python");
    const pythonLibName = std.fmt.allocPrint(b.allocator, "python{s}", .{pythonVer}) catch @panic("Missing python");

    const test_step = b.step("test", "Run library tests");
    const docs_step = b.step("docs", "Generate docs");

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("pyz3/src/ffi.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_c.defineCMacro("Py_LIMITED_API", "0x030D0000");
    translate_c.addIncludePath(.{ .cwd_relative = pythonInc });

    // We never build this lib, but we use it to generate docs.
    const pyz3_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "pyz3",
        .root_module = b.createModule(.{
            .root_source_file = b.path("pyz3/src/pyz3.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const pyz3_lib_mod = b.createModule(.{ .root_source_file = b.path("./pyconf.dummy.zig") });
    pyz3_lib_mod.addIncludePath(.{ .cwd_relative = pythonInc });
    pyz3_lib.root_module.addImport("ffi", translate_c.createModule());
    pyz3_lib.root_module.addImport("pyconf", pyz3_lib_mod);

    const pyz3_docs = b.addInstallDirectory(.{
        .source_dir = pyz3_lib.getEmittedDocs(),
        // Emit the Zig docs into zig-out/../docs/zig
        .install_dir = .{ .custom = "../docs" },
        .install_subdir = "zig",
    });
    docs_step.dependOn(&pyz3_docs.step);

    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("pyz3/src/pyz3.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    main_tests.linkLibC();
    main_tests.addLibraryPath(.{ .cwd_relative = pythonLib });
    main_tests.linkSystemLibrary(pythonLibName);
    main_tests.addRPath(.{ .cwd_relative = pythonLib });
    const main_tests_mod = b.createModule(.{ .root_source_file = b.path("./pyconf.dummy.zig") });
    main_tests_mod.addIncludePath(.{ .cwd_relative = pythonInc });
    main_tests.root_module.addImport("ffi", translate_c.createModule());
    main_tests.root_module.addImport("pyconf", main_tests_mod);

    const run_main_tests = b.addRunArtifact(main_tests);
    test_step.dependOn(&run_main_tests.step);

    // Setup a library target to trick the Zig Language Server into providing completions for @import("pyz3")
    const example_lib = b.addLibrary(.{
        .name = "example",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("example/hello.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    example_lib.linkLibC();
    example_lib.addLibraryPath(.{ .cwd_relative = pythonLib });
    example_lib.linkSystemLibrary(pythonLibName);
    example_lib.addRPath(.{ .cwd_relative = pythonLib });
    const example_lib_mod = b.createModule(.{ .root_source_file = b.path("pyz3/src/pyz3.zig") });
    example_lib_mod.addIncludePath(.{ .cwd_relative = pythonInc });
    example_lib.root_module.addImport("ffi", translate_c.createModule());
    example_lib.root_module.addImport("pyz3", example_lib_mod);
    example_lib.root_module.addImport(
        "pyconf",
        b.createModule(.{ .root_source_file = b.path("./pyconf.dummy.zig") }),
    );

    // Option for emitting test binary based on the given root source.
    // This is used for debugging as in .vscode/tasks.json
    const test_debug_root = b.option([]const u8, "test-debug-root", "The root path of a file emitted as a binary for use with the debugger");
    if (test_debug_root) |root| {
        main_tests.root_module.root_source_file = b.path(root);
        const test_bin_install = b.addInstallBinFile(main_tests.getEmittedBin(), "test.bin");
        b.getInstallStep().dependOn(&test_bin_install.step);
    }
}

fn getPythonIncludePath(
    python_exe: []const u8,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const includeResult = try runProcess(.{
        .allocator = allocator,
        .argv = &.{ python_exe, "-c", "import sysconfig; print(sysconfig.get_path('include'), end='')" },
    });
    defer allocator.free(includeResult.stderr);
    return includeResult.stdout;
}

fn getPythonLibraryPath(python_exe: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const includeResult = try runProcess(.{
        .allocator = allocator,
        .argv = &.{ python_exe, "-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'), end='')" },
    });
    defer allocator.free(includeResult.stderr);
    return includeResult.stdout;
}

fn getPythonLDVersion(python_exe: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const includeResult = try runProcess(.{
        .allocator = allocator,
        .argv = &.{ python_exe, "-c", "import sysconfig; print(sysconfig.get_config_var('LDVERSION'), end='')" },
    });
    defer allocator.free(includeResult.stderr);
    return includeResult.stdout;
}

const runProcess = if (builtin.zig_version.minor >= 12) std.process.Child.run else std.process.Child.exec;

/// Get target from ZIG_TARGET environment variable if set
fn getTargetFromEnv(b: *std.Build) ?std.Build.ResolvedTarget {
    const zig_target_str = std.process.getEnvVarOwned(b.allocator, "ZIG_TARGET") catch return null;
    defer b.allocator.free(zig_target_str);

    if (zig_target_str.len == 0) return null;

    const query = std.Target.Query.parse(.{ .arch_os_abi = zig_target_str }) catch |err| {
        std.debug.print("Warning: Invalid ZIG_TARGET '{s}': {}\n", .{ zig_target_str, err });
        return null;
    };

    return b.resolveTargetQuery(query);
}

/// Get optimization mode from PYZ3_OPTIMIZE environment variable if set
fn getOptimizeFromEnv(b: *std.Build) ?std.builtin.OptimizeMode {
    const optimize_str = std.process.getEnvVarOwned(b.allocator, "PYZ3_OPTIMIZE") catch return null;
    defer b.allocator.free(optimize_str);

    if (optimize_str.len == 0) return null;

    if (std.mem.eql(u8, optimize_str, "Debug")) return .Debug;
    if (std.mem.eql(u8, optimize_str, "ReleaseSafe")) return .ReleaseSafe;
    if (std.mem.eql(u8, optimize_str, "ReleaseFast")) return .ReleaseFast;
    if (std.mem.eql(u8, optimize_str, "ReleaseSmall")) return .ReleaseSmall;

    std.debug.print("Warning: Invalid PYZ3_OPTIMIZE '{s}', using default\n", .{optimize_str});
    return null;
}
