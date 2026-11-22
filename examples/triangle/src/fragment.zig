//! Fragment shader for the colored triangle.

const Color = @Vector(4, f32);

const color_in = @extern(*addrspace(.input) Color, .{
    .name = "color",
    .decoration = .{ .location = 0 },
});

const color_out = @extern(*addrspace(.output) Color, .{
    .name = "fragment_color",
    .decoration = .{ .location = 0 },
});

export fn main() callconv(.{ .spirv_fragment = .{} }) void {
    color_out.* = color_in.*;
}
