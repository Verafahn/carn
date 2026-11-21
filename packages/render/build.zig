const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rhi = b.addModule("rhi", .{
        .root_source_file = b.path("src/rhi/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const null = b.addModule("null", .{
        .root_source_file = b.path("src/backend/null/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{.name = "rhi", .module = rhi},
        },
    });

    const vulkan = b.addModule("vulkan", .{
        .root_source_file = b.path("src/backend/vulkan/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{.name = "rhi", .module = rhi},
        },
    });

    const test_render = b.addTest(.{ .root_module = rhi });
    const run_test_render = b.addRunArtifact(test_render);
    const test_step = b.step("test", "Test this module.");
    test_step.dependOn(&run_test_render.step);
}
