//! Vertex shader for the colored triangle.
//!
//! Positions and colors are derived from the vertex index, so the
//! pipeline needs no vertex buffers.

const spirv = @import("std").spirv;

const Color = @Vector(4, f32);

const color_out = @extern(*addrspace(.output) Color, .{
    .name = "color",
    .decoration = .{ .location = 0 },
});

export fn main() callconv(.spirv_vertex) void {
    const positions = [3]@Vector(4, f32){
        .{ 0.0, -0.5, 0.0, 1.0 },
        .{ 0.5, 0.5, 0.0, 1.0 },
        .{ -0.5, 0.5, 0.0, 1.0 },
    };
    const colors = [3]Color{
        .{ 1.0, 0.0, 0.0, 1.0 },
        .{ 0.0, 1.0, 0.0, 1.0 },
        .{ 0.0, 0.0, 1.0, 1.0 },
    };

    const index = @min(spirv.vertex_index, 2);

    spirv.position_out.* = positions[index];
    color_out.* = colors[index];
}
