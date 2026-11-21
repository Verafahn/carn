//! GPU Resource Abstraction Handle

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

kind: Kind,
handle: u64,

pub const Kind = enum {
    buffer,
    command_buffer,
    command_pool,
    descriptor_set,
    descriptor_set_layout,
    fence,
    pipeline_compute,
    pipeline_graphics,
    pipeline_layout,
    sampler,
    semaphore,
    shader,
    surface,
    swapchain,
    texture,
};

pub const Desc = union(Kind) {
    buffer: Buffer.Desc,
    command_buffer: CommandBuffer.Desc,
    command_pool: CommandPool.Desc,
    descriptor_set: DescriptorSet.Desc,
    descriptor_set_layout: DescriptorSetLayout.Desc,
    fence: Fence.Desc,
    pipeline_compute: PipelineCompute.Desc,
    pipeline_graphics: PipelineGraphics.Desc,
    pipeline_layout: PipelineLayout.Desc,
    sampler: Sampler.Desc,
    semaphore: Semaphore.Desc,
    shader: Shader.Desc,
    surface: Surface.Desc,
    swapchain: Swapchain.Desc,
    texture: Texture.Desc,
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
