const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const render = b.addModule("render", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_render = b.addTest(.{ .root_module = render });
    const run_test_render = b.addRunArtifact(test_render);
    const test_step = b.step("test", "Test this module.");
    test_step.dependOn(&run_test_render.step);
}
