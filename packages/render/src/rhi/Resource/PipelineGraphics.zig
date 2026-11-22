//! A graphics pipeline.
//!
//! A graphics pipeline captures the fixed state used by
//! draw commands: shaders, vertex layout, primitive and
//! raster state, depth-stencil state and blending.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const PipelineLayout = @import("PipelineLayout.zig");
const Shader = @import("Shader.zig");
const Texture = @import("Texture.zig");
const PipelineGraphics = @This();
const std = @import("std");

/// Fixed state of a graphics pipeline.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Bindings and push constants of the pipeline.
    layout: PipelineLayout,
    /// Vertex shader.
    vertex: Shader,
    /// Fragment shader; null disables the stage.
    fragment: ?Shader = null,
    /// Vertex buffer layouts, in input slot order.
    vertex_buffers: []const VertexBuffer = &.{},
    /// Primitive assembly state.
    primitive: Primitive = .{},
    /// Rasterization state.
    raster: Raster = .{},
    /// Depth-stencil state; null disables the test.
    depth_stencil: ?DepthStencil = null,
    /// Color attachment state.
    color: Color = .{},
    /// Depth-stencil attachment format; null disables
    /// the depth-stencil attachment.
    depth_format: ?Resource.Format = null,
    /// Samples per pixel of the render pass.
    samples: Texture.SampleCount = .x1,
    /// Writes alpha coverage to the color target.
    alpha_to_coverage: bool = false,
};

/// Layout of one vertex buffer.
pub const VertexBuffer = struct {
    /// Bytes between consecutive vertices.
    stride: u32,
    /// Attributes read from this buffer.
    attributes: []const VertexAttribute = &.{},
};

/// One vertex attribute.
pub const VertexAttribute = struct {
    /// Shader input location.
    location: u32,
    /// Byte offset inside the vertex.
    offset: u32 = 0,
    /// Type and component count of the attribute.
    format: VertexFormat,
};

/// Type and component count of a vertex attribute.
pub const VertexFormat = enum {
    /// One 32-bit float.
    f32,
    /// Two 32-bit floats.
    f32x2,
    /// Three 32-bit floats.
    f32x3,
    /// Four 32-bit floats.
    f32x4,
    /// Two 8-bit unsigned integers.
    u8x2,
    /// Four 8-bit unsigned integers.
    u8x4,
    /// Four 8-bit unsigned normalized values.
    u8x4_unorm,
    /// Two 8-bit signed integers.
    i8x2,
    /// Four 8-bit signed integers.
    i8x4,
    /// Two 16-bit unsigned integers.
    u16x2,
    /// Four 16-bit unsigned integers.
    u16x4,
    /// Two 16-bit unsigned normalized values.
    u16x2_unorm,
    /// Four 16-bit unsigned normalized values.
    u16x4_unorm,
    /// Two 16-bit signed integers.
    i16x2,
    /// Four 16-bit signed integers.
    i16x4,
    /// One 32-bit unsigned integer.
    u32,
    /// Two 32-bit unsigned integers.
    u32x2,
    /// Three 32-bit unsigned integers.
    u32x3,
    /// Four 32-bit unsigned integers.
    u32x4,
};

/// Primitive assembly state.
pub const Primitive = struct {
    /// How vertices form primitives.
    topology: Topology = .triangle_list,
    /// Winding considered front-facing.
    front_face: FrontFace = .counter_clockwise,
    /// Faces removed before rasterization.
    cull: Cull = .back,
    /// How primitives are rasterized.
    polygon: Polygon = .fill,
};

/// How vertices form primitives.
pub const Topology = enum {
    /// Independent points.
    point_list,
    /// A connected line strip.
    line_strip,
    /// Independent lines.
    line_list,
    /// A connected triangle strip.
    triangle_strip,
    /// Independent triangles.
    triangle_list,
    /// A fan around the first vertex.
    triangle_fan,
};

/// Winding order of a front face.
pub const FrontFace = enum {
    /// Counter-clockwise vertices are front.
    counter_clockwise,
    /// Clockwise vertices are front.
    clockwise,
};

/// Faces removed before rasterization.
pub const Cull = enum {
    /// Keep every face.
    none,
    /// Remove front faces.
    front,
    /// Remove back faces.
    back,
};

/// Rasterization mode of primitives.
pub const Polygon = enum {
    /// Fill the primitive area.
    fill,
    /// Draw primitive edges as lines.
    line,
    /// Draw primitive vertices as points.
    point,
};

/// Rasterization state.
pub const Raster = struct {
    /// Clamp depth instead of discarding the fragment.
    depth_clamp: bool = false,
    /// Skip rasterization; no fragments are produced.
    discard: bool = false,
    /// Width of rasterized lines, in pixels.
    line_width: f32 = 1,
    /// Depth bias applied to fragments.
    depth_bias: DepthBias = .{},
};

/// Constant and slope depth bias.
pub const DepthBias = struct {
    /// Constant bias added to every fragment.
    constant: f32 = 0,
    /// Bias proportional to the depth slope.
    slope: f32 = 0,
    /// Largest bias magnitude.
    clamp: f32 = 0,
};

/// Depth and stencil test state.
pub const DepthStencil = struct {
    /// Write passing depth values.
    write: bool = true,
    /// Comparison used by the depth test.
    compare: Resource.CompareOp = .less,
    /// Stencil test state; null disables it.
    stencil: ?Stencil = null,
};

/// Stencil test state.
pub const Stencil = struct {
    /// State for front-facing primitives.
    front: StencilFace = .{},
    /// State for back-facing primitives.
    back: StencilFace = .{},
    /// Bits used when comparing the reference.
    compare_mask: u8 = 0xff,
    /// Bits written by stencil operations.
    write_mask: u8 = 0xff,
    /// Reference value used by compare and replace.
    reference: u8 = 0,
};

/// Stencil state of one face.
pub const StencilFace = struct {
    /// Comparison against the reference value.
    compare: Resource.CompareOp = .always,
    /// Applied when the stencil test fails.
    fail: StencilOp = .keep,
    /// Applied when depth fails after stencil.
    depth_fail: StencilOp = .keep,
    /// Applied when both tests pass.
    pass: StencilOp = .keep,
};

/// Operation applied to a stencil value.
pub const StencilOp = enum {
    /// Keep the current value.
    keep,
    /// Replace the value with zero.
    zero,
    /// Replace the value with the reference.
    replace,
    /// Increment, clamping to the maximum.
    increment_clamp,
    /// Decrement, clamping to zero.
    decrement_clamp,
    /// Bitwise invert the value.
    invert,
    /// Increment, wrapping to zero.
    increment_wrap,
    /// Decrement, wrapping to the maximum.
    decrement_wrap,
};

/// Color attachment state.
pub const Color = struct {
    /// Formats of the color attachments, in order.
    formats: []const Resource.Format = &.{},
    /// Blend state per attachment; empty or as long as
    /// `formats`.
    targets: []const ColorTarget = &.{},
    /// Blend constants used by `.constant` factors.
    constants: [4]f32 = .{ 0, 0, 0, 0 },
};

/// Blend state of one color attachment.
pub const ColorTarget = struct {
    /// Enables blending; when false the source value
    /// replaces the destination.
    blend: bool = false,
    /// Blend factors of the color channels.
    color: BlendComponent = .{},
    /// Blend factors of the alpha channel.
    alpha: BlendComponent = .{},
    /// Channels written by the pipeline.
    write_mask: ColorMask = .{
        .r = true,
        .g = true,
        .b = true,
        .a = true,
    },
};

/// Channels written by a pipeline.
pub const ColorMask = packed struct {
    /// The red channel.
    r: bool = false,
    /// The green channel.
    g: bool = false,
    /// The blue channel.
    b: bool = false,
    /// The alpha channel.
    a: bool = false,
};

/// One blend equation.
pub const BlendComponent = struct {
    /// Factor applied to the source value.
    src: BlendFactor = .one,
    /// Factor applied to the destination value.
    dst: BlendFactor = .zero,
    /// Operation combining the two factors.
    op: BlendOp = .add,
};

/// Factors that scale a blend operand.
pub const BlendFactor = enum {
    /// Zero.
    zero,
    /// One.
    one,
    /// Source color.
    src_color,
    /// One minus the source color.
    one_minus_src_color,
    /// Destination color.
    dst_color,
    /// One minus the destination color.
    one_minus_dst_color,
    /// Source alpha.
    src_alpha,
    /// One minus the source alpha.
    one_minus_src_alpha,
    /// Destination alpha.
    dst_alpha,
    /// One minus the destination alpha.
    one_minus_dst_alpha,
    /// The pipeline blend constant.
    constant,
    /// One minus the pipeline blend constant.
    one_minus_constant,
    /// Smaller of source alpha and one minus
    /// destination alpha.
    src_alpha_saturate,
};

/// Operations that combine blend operands.
pub const BlendOp = enum {
    /// Source plus destination.
    add,
    /// Source minus destination.
    subtract,
    /// Destination minus source.
    reverse_subtract,
    /// Smaller of the two values.
    min,
    /// Larger of the two values.
    max,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a graphics pipeline on `device`.
///
/// # Args
/// - `device`: the device that owns the pipeline.
/// - `desc`: shaders, layout and fixed state.
///
/// # Return
/// A live pipeline; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the state is inconsistent,
///   for example a blend target without a format.
/// - `error.InvalidResource`: a referenced object is
///   not live or has the wrong kind.
/// - `error.TooManyObjects`: the backend cannot track
///   another pipeline.
/// - `error.Unsupport`: the backend cannot create
///   graphics pipelines.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(
    device: Device,
    desc: Desc,
) Device.CreateError!PipelineGraphics {
    const rc_desc: Resource.Desc = .{ .pipeline_graphics = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the pipeline.
///
/// # Note
/// Submitted work that uses the pipeline must have
/// completed. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `pipeline_graphics`.
pub fn destroy(self: PipelineGraphics) void {
    std.debug.assert(self.handle.kind == .pipeline_graphics);
    self.device.rawDestroy(self.handle, @returnAddress());
}
