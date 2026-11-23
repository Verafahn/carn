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
    const test_step = b.step("test", "");
    test_step.dependOn(&run_test_math.step);
}
