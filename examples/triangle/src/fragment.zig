//! Fragment shader for the colored triangle.

const shader = @import("carn").utils.shader;
const shared = @import("shared.zig");

const ColorStream = shader.DataStream(shared.Varying, .{ .input = true });
const UniformStream = shader.Uniform(shared.UniformData);

const color_out = @extern(*addrspace(.output) @Vector(4, f32), .{
    .name = "fragment_color",
    .decoration = .{ .location = 0 },
});

export fn main() callconv(.{ .spirv_fragment = .{} }) void {
    const varying = ColorStream.read();
    const uniform_value = UniformStream.read();
    color_out.* = varying.color * uniform_value.tint;
}
