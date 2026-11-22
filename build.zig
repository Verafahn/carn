const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const render = b.dependency("render", .{
        .target = target,
        .optimize = optimize,
        .vulkan = true,
    });

    const carn = b.addModule("carn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "rhi", .module = render.module("rhi") },
        },
    });

    const carn_test = b.addTest(.{ .root_module = carn });
    const run_carn_test = b.addRunArtifact(carn_test);
    const carn_test_step = b.step("test", "Test module carn.");
    carn_test_step.dependOn(&run_carn_test.step);

    const test_render = b.step("test-render", "Test submodule 'render'.");
    test_render.dependOn(&render.builder.top_level_steps.get("test").?.step);
}
