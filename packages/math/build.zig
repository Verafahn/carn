const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const math = b.addModule("math", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_math = b.addTest(.{ .root_module = math });
    const run_test_math = b.addRunArtifact(test_math);
    const test_step = b.step("test", "Test the math module.");
    test_step.dependOn(&run_test_math.step);

    const spirv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv32,
        .os_tag = .vulkan,
    });
    const spirv_check = b.addObject(.{
        .name = "math_spirv_check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/spirv.zig"),
            .target = spirv_target,
            .optimize = .ReleaseSmall,
        }),
        // The SPIR-V code generator lives in the
        // self-hosted backend.
        .use_llvm = false,
    });
    const spirv_step = b.step(
        "check-spirv",
        "Compile the shared math code for SPIR-V.",
    );
    spirv_step.dependOn(&spirv_check.step);
    test_step.dependOn(&spirv_check.step);
}
