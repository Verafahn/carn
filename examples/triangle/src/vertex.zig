//! Vertex shader for the colored triangle.
//!
//! Positions and colors come from the vertex buffer.

const spirv = @import("std").spirv;
const shader = @import("carn").utils.shader;
const shared = @import("shared.zig");

const VertexStream = shader.Vertex(shared.VertexData);
const ColorStream = shader.DataStream(shared.Varying, .{ .output = true });

export fn main() callconv(.spirv_vertex) void {
    const vertex = VertexStream.load();
    spirv.position_out.* = vertex.position;
    ColorStream.write(.{ .color = vertex.color });
}
