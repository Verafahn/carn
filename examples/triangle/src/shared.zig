//! Data shared by the host and both triangle shaders.

/// One vertex submitted through the shader utilities.
pub const VertexData = struct {
    /// Clip-space position.
    position: @Vector(4, f32),
    /// Vertex color.
    color: @Vector(4, f32),
};

/// Uniform data read by the fragment shader.
pub const UniformData = struct {
    /// Color multiplier.
    tint: @Vector(4, f32),
};

/// Data passed from the vertex shader to the
/// fragment shader.
pub const Varying = struct {
    /// Interpolated vertex color.
    color: @Vector(4, f32),
};
