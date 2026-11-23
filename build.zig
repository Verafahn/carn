const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const render = b.dependency("render", .{
        .target = target,
        .optimize = optimize,
        .vulkan = true,
    });
    const math = b.dependency("math", .{
        .target = target,
        .optimize = optimize,
    });

    const carn = b.addModule("carn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "render/rhi", .module = render.module("rhi") },
            .{ .name = "render/null", .module = render.module("null") },
            .{ .name = "render/vulkan", .module = render.module("vulkan") },
            .{ .name = "math", .module = math.module("math") },
        },
    });

    const carn_test = b.addTest(.{ .root_module = carn });
    const run_carn_test = b.addRunArtifact(carn_test);
    const carn_test_step = b.step("test", "Test module carn.");
    carn_test_step.dependOn(&run_carn_test.step);

    const test_render = b.step("test-render", "Test submodule 'render'.");
    test_render.dependOn(&render.builder.top_level_steps.get("test").?.step);
    const test_math = b.step("test-math", "Test submodule 'math'.");
    test_math.dependOn(&math.builder.top_level_steps.get("test").?.step);
}
