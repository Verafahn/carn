const std = @import("std");
const translate_c = @import("translate_c");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const carn = b.dependency("carn", .{
        .target = target,
        .optimize = optimize,
    });
    const glfw = b.dependency("glfw_zig", .{
        .target = target,
        .optimize = optimize,
    });

    // GLFW's native window accessors are exposed with platform macros, so
    // translate a small wrapper header instead of importing the C headers.
    const glfw_header: []const u8 = switch (target.result.os.tag) {
        .windows =>
        \\#define GLFW_EXPOSE_NATIVE_WIN32
        \\#include <GLFW/glfw3.h>
        \\#include <GLFW/glfw3native.h>
        ,
        .linux =>
        \\#define GLFW_EXPOSE_NATIVE_X11
        \\#define GLFW_EXPOSE_NATIVE_WAYLAND
        \\#include <GLFW/glfw3.h>
        \\#include <GLFW/glfw3native.h>
        ,
        else =>
        \\#include <GLFW/glfw3.h>
        ,
    };
    const glfw_bindings: translate_c.Translator = .init(
        b.dependency("translate_c", .{}),
        .{
            .name = "glfw_bindings",
            .c_source_file = b.addWriteFiles().add(
                "glfw_bindings.h",
                glfw_header,
            ),
            .target = target,
            .optimize = optimize,
        },
    );
    glfw_bindings.addIncludePath(glfw.path("glfw/include"));
    glfw_bindings.linkLibrary(glfw.artifact("glfw"));

    // Shaders are ordinary Zig programs compiled for the SPIR-V target.
    const spirv = b.resolveTargetQuery(.{
        .cpu_arch = .spirv32,
        .os_tag = .vulkan,
    });
    const vertex = b.addObject(.{
        .name = "triangle_vertex",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vertex.zig"),
            .target = spirv,
            .optimize = .ReleaseSmall,
        }),
        // The SPIR-V code generator lives in the self-hosted backend.
        .use_llvm = false,
    });
    const fragment = b.addObject(.{
        .name = "triangle_fragment",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/fragment.zig"),
            .target = spirv,
            .optimize = .ReleaseSmall,
        }),
        .use_llvm = false,
    });

    // Bundle the compiled shaders so the executable can embed them.
    const generated = b.addWriteFiles();
    _ = generated.addCopyFile(vertex.getEmittedBin(), "vertex.spv");
    _ = generated.addCopyFile(fragment.getEmittedBin(), "fragment.spv");
    const shaders = generated.add("shaders.zig",
        \\pub const vertex = @embedFile("vertex.spv");
        \\pub const fragment = @embedFile("fragment.spv");
        \\
    );

    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "carn", .module = carn.module("carn") },
            .{ .name = "glfw", .module = glfw_bindings.mod },
        },
    });
    exe_module.addImport("shaders", b.createModule(.{
        .root_source_file = shaders,
        .target = target,
        .optimize = optimize,
    }));

    const exe = b.addExecutable(.{
        .name = "triangle",
        .root_module = exe_module,
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();

    const run_step = b.step("run", "Run the colored triangle example.");
    run_step.dependOn(&run.step);
}
