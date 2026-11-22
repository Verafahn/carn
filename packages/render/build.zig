const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rhi = b.addModule("rhi", .{
        .root_source_file = b.path("src/rhi/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_render = b.addTest(.{ .root_module = rhi });
    const run_test_render = b.addRunArtifact(test_render);

    const nil = b.addModule("null", .{
        .root_source_file = b.path("src/backend/null/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "rhi", .module = rhi },
        },
    });
    const test_nil = b.addTest(.{ .root_module = nil });
    const run_test_nil = b.addRunArtifact(test_nil);

    const test_step = b.step("test", "Test this module.");
    test_step.dependOn(&run_test_render.step);
    test_step.dependOn(&run_test_nil.step);

    if (b.option(bool, "vulkan", "") orelse b.isRoot()) {
        const vk = b.dependency("vulkan", .{
            .registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml"),
        }).module("vulkan-zig");
        
        const vulkan = b.addModule("vulkan", .{
            .root_source_file = b.path("src/backend/vulkan/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "rhi", .module = rhi },
                .{ .name = "vk", .module = vk },
            },
        });
        const test_vulkan = b.addTest(.{ .root_module = vulkan });
        const run_test_vulkan = b.addRunArtifact(test_vulkan);

        test_step.dependOn(&run_test_vulkan.step);
    }
}
