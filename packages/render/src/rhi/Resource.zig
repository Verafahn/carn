//! GPU resource handles and shared resource value types.
//!
//! A `Resource` is a tagged handle owned by a backend. The
//! typed wrappers re-exported here add a kind at the Zig
//! type level; they do not add state.
//!
//! Descriptions are borrowed for the duration of a create
//! call. Every slice inside a description must stay valid
//! until the backend returns.
//!
//! Typed operations come as `x` and `xWithAddr`; the
//! latter takes an explicit call-site address. Prefer
//! `x`, which captures `@returnAddress`.

pub const Buffer = @import("Resource/Buffer.zig");
pub const CommandBuffer = @import("Resource/CommandBuffer.zig");
pub const CommandPool = @import("Resource/CommandPool.zig");
pub const DescriptorSet = @import("Resource/DescriptorSet.zig");
pub const DescriptorSetLayout = @import("Resource/DescriptorSetLayout.zig");
pub const Fence = @import("Resource/Fence.zig");
pub const PipelineCompute = @import("Resource/PipelineCompute.zig");
pub const PipelineGraphics = @import("Resource/PipelineGraphics.zig");
pub const PipelineLayout = @import("Resource/PipelineLayout.zig");
pub const Sampler = @import("Resource/Sampler.zig");
pub const Semaphore = @import("Resource/Semaphore.zig");
pub const Shader = @import("Resource/Shader.zig");
pub const Surface = @import("Resource/Surface.zig");
pub const Swapchain = @import("Resource/Swapchain.zig");
pub const Texture = @import("Resource/Texture.zig");

/// A backend-owned resource handle.
///
/// # Note
/// Handles are values. Copying one does not create a
/// second resource, and every live handle must be
/// destroyed exactly once.
kind: Kind,
/// Backend identifier; interpreted by the owning
/// backend only.
handle: u64,

/// The kind of resource a handle refers to.
pub const Kind = enum {
    /// A linear GPU buffer.
    buffer,
    /// A recorded command list.
    command_buffer,
    /// Allocator for command buffers.
    command_pool,
    /// A set of descriptors bound to a pipeline.
    descriptor_set,
    /// The shape of a descriptor set.
    descriptor_set_layout,
    /// A GPU/CPU synchronization point.
    fence,
    /// A compute pipeline.
    pipeline_compute,
    /// A graphics pipeline.
    pipeline_graphics,
    /// Bindings shared by several pipelines.
    pipeline_layout,
    /// State used when sampling a texture.
    sampler,
    /// A GPU/GPU synchronization point.
    semaphore,
    /// Shader code and its entry point.
    shader,
    /// A presentation surface.
    surface,
    /// A chain of presentable images.
    swapchain,
    /// A GPU image.
    texture,
};

/// Resource descriptions keyed by resource kind.
pub const Desc = union(Kind) {
    /// A linear GPU buffer.
    buffer: Buffer.Desc,
    /// A recorded command list.
    command_buffer: CommandBuffer.Desc,
    /// Allocator for command buffers.
    command_pool: CommandPool.Desc,
    /// A set of descriptors bound to a pipeline.
    descriptor_set: DescriptorSet.Desc,
    /// The shape of a descriptor set.
    descriptor_set_layout: DescriptorSetLayout.Desc,
    /// A GPU/CPU synchronization point.
    fence: Fence.Desc,
    /// A compute pipeline.
    pipeline_compute: PipelineCompute.Desc,
    /// A graphics pipeline.
    pipeline_graphics: PipelineGraphics.Desc,
    /// Bindings shared by several pipelines.
    pipeline_layout: PipelineLayout.Desc,
    /// State used when sampling a texture.
    sampler: Sampler.Desc,
    /// A GPU/GPU synchronization point.
    semaphore: Semaphore.Desc,
    /// Shader code and its entry point.
    shader: Shader.Desc,
    /// A presentation surface.
    surface: Surface.Desc,
    /// A chain of presentable images.
    swapchain: Swapchain.Desc,
    /// A GPU image.
    texture: Texture.Desc,
};

/// Pixel and texel formats understood by the RHI.
///
/// Names list channels from the least significant byte to
/// the most significant byte. The suffix describes how
/// the channel values are interpreted.
pub const Format = enum {
    /// One 8-bit unsigned normalized channel.
    r8_unorm,
    /// One 8-bit signed normalized channel.
    r8_snorm,
    /// One 8-bit unsigned integer channel.
    r8_uint,
    /// One 8-bit signed integer channel.
    r8_sint,
    /// Two 8-bit unsigned normalized channels.
    r8g8_unorm,
    /// Two 8-bit signed normalized channels.
    r8g8_snorm,
    /// Two 8-bit unsigned integer channels.
    r8g8_uint,
    /// Two 8-bit signed integer channels.
    r8g8_sint,
    /// Four 8-bit unsigned normalized channels.
    r8g8b8a8_unorm,
    /// Four 8-bit signed normalized channels.
    r8g8b8a8_snorm,
    /// Four 8-bit unsigned integer channels.
    r8g8b8a8_uint,
    /// Four 8-bit signed integer channels.
    r8g8b8a8_sint,
    /// Four 8-bit sRGB encoded channels.
    r8g8b8a8_srgb,
    /// Four 8-bit channels in BGRA byte order.
    b8g8r8a8_unorm,
    /// Four 8-bit sRGB channels in BGRA byte order.
    b8g8r8a8_srgb,
    /// One 16-bit unsigned integer channel.
    r16_uint,
    /// One 16-bit signed integer channel.
    r16_sint,
    /// One 16-bit float channel.
    r16_float,
    /// Two 16-bit unsigned integer channels.
    r16g16_uint,
    /// Two 16-bit signed integer channels.
    r16g16_sint,
    /// Two 16-bit float channels.
    r16g16_float,
    /// Four 16-bit unsigned integer channels.
    r16g16b16a16_uint,
    /// Four 16-bit signed integer channels.
    r16g16b16a16_sint,
    /// Four 16-bit float channels.
    r16g16b16a16_float,
    /// One 32-bit unsigned integer channel.
    r32_uint,
    /// One 32-bit signed integer channel.
    r32_sint,
    /// One 32-bit float channel.
    r32_float,
    /// Two 32-bit unsigned integer channels.
    r32g32_uint,
    /// Two 32-bit signed integer channels.
    r32g32_sint,
    /// Two 32-bit float channels.
    r32g32_float,
    /// Three 32-bit unsigned integer channels.
    r32g32b32_uint,
    /// Three 32-bit signed integer channels.
    r32g32b32_sint,
    /// Three 32-bit float channels.
    r32g32b32_float,
    /// Four 32-bit unsigned integer channels.
    r32g32b32a32_uint,
    /// Four 32-bit signed integer channels.
    r32g32b32a32_sint,
    /// Four 32-bit float channels.
    r32g32b32a32_float,
    /// Packed 10:10:10:2 unsigned normalized channels.
    r10g10b10a2_unorm,
    /// Packed 10:10:10:2 in BGRA bit order.
    b10g10r10a2_unorm,
    /// Packed unsigned float channels, 11:11:10.
    r11g11b10_float,
    /// A 16-bit normalized depth channel.
    depth16_unorm,
    /// A 24-bit depth channel and 8-bit stencil.
    depth24_plus_stencil8,
    /// A 32-bit float depth channel.
    depth32_float,
    /// An 8-bit stencil channel.
    stencil8,
};

/// A three-axis size in pixels, texels or elements.
pub const Extent = struct {
    /// Size along the x axis.
    width: u32,
    /// Size along the y axis; 1 for 1D resources.
    height: u32 = 1,
    /// Size along the z axis; 1 for 2D resources.
    depth: u32 = 1,
};

/// A scalar comparison used by samplers and depth tests.
pub const CompareOp = enum {
    /// The comparison never passes.
    never,
    /// Passes when the left value is less.
    less,
    /// Passes when the values are equal.
    equal,
    /// Passes when the left value is less or equal.
    less_equal,
    /// Passes when the left value is greater.
    greater,
    /// Passes when the values are not equal.
    not_equal,
    /// Passes when the left value is greater or equal.
    greater_equal,
    /// The comparison always passes.
    always,
};

test {
    _ = Buffer;
    _ = CommandBuffer;
    _ = CommandPool;
    _ = DescriptorSet;
    _ = DescriptorSetLayout;
    _ = Fence;
    _ = PipelineCompute;
    _ = PipelineGraphics;
    _ = PipelineLayout;
    _ = Sampler;
    _ = Semaphore;
    _ = Shader;
    _ = Surface;
    _ = Swapchain;
    _ = Texture;
}
