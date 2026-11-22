//! Backend state owned by one Vulkan resource.
//!
//! The registry in `root.zig` maps an RHI handle to one
//! `Entry`. The active field of `Data` holds the native
//! objects and bookkeeping for that resource.

const std = @import("std");
const rhi = @import("rhi");
const vk = @import("vk");

const Resource = rhi.Resource;
const Device = rhi.Device;
const Texture = Resource.Texture;
const Buffer = Resource.Buffer;
const Shader = Resource.Shader;
const Surface = Resource.Surface;
const DescriptorSetLayout = Resource.DescriptorSetLayout;
const PipelineLayout = Resource.PipelineLayout;
const CommandBuffer = Resource.CommandBuffer;

/// Sentinel used by the free-list of registry entries.
pub const no_index = std.math.maxInt(u32);

/// One live or recycled registry slot.
pub const Entry = struct {
    /// Kind of resource stored in `data`.
    kind: Resource.Kind,
    /// Incremented on every recycle.
    generation: u32,
    /// False once the resource is destroyed.
    live: bool,
    /// Owned copy of the debug label, if any.
    label: ?[]u8,
    /// Native state selected by `kind`.
    data: Data,
    /// Next slot in the free list.
    next_free: u32 = no_index,
};

/// Native state of one resource.
pub const Data = union(Resource.Kind) {
    buffer: BufferState,
    command_buffer: CommandBufferState,
    command_pool: CommandPoolState,
    descriptor_set: DescriptorSetState,
    descriptor_set_layout: DescriptorSetLayoutState,
    fence: FenceState,
    pipeline_compute: PipelineState,
    pipeline_graphics: PipelineState,
    pipeline_layout: PipelineLayoutState,
    sampler: SamplerState,
    semaphore: SemaphoreState,
    shader: ShaderState,
    surface: SurfaceState,
    swapchain: SwapchainState,
    texture: TextureState,
};

/// A buffer and the memory bound to it.
pub const BufferState = struct {
    /// Native buffer.
    buffer: vk.Buffer,
    /// Memory block bound at offset zero.
    memory: vk.DeviceMemory,
    /// Size requested by the description.
    size: u64,
    /// Usage flags requested by the description.
    usage: Buffer.Usage,
    /// Memory domain requested by the description.
    domain: Buffer.Memory,
    /// Properties of the selected memory type.
    memory_flags: vk.MemoryPropertyFlags,
    /// True while the host owns a mapping.
    mapped: bool = false,
    /// Pointer returned by `vkMapMemory`.
    map_ptr: ?[*]u8 = null,
    /// First byte of the mapped range.
    map_offset: u64 = 0,
    /// Length of the mapped range.
    map_size: u64 = 0,
};

/// An image, its view and the memory bound to it.
pub const TextureState = struct {
    /// Native image.
    image: vk.Image,
    /// Memory block; null for swapchain images.
    memory: ?vk.DeviceMemory,
    /// View used for rendering, sampling and copies.
    view: vk.ImageView,
    /// Stencil-only view of a combined depth-stencil
    /// texture, when one exists.
    stencil_view: ?vk.ImageView = null,
    /// Description the image was created from.
    desc: Texture.Desc,
    /// Layout of the whole texture.
    layout: vk.ImageLayout = .undefined,
    /// Swapchain that owns the image, if any.
    owner: ?Resource = null,
};

/// A sampler object.
pub const SamplerState = struct {
    sampler: vk.Sampler,
};

/// A shader module and its entry point.
pub const ShaderState = struct {
    module: vk.ShaderModule,
    stage: Shader.Stage,
    /// Null-terminated entry point name.
    entry: [:0]u8,
};

/// A descriptor set layout and its binding table.
pub const DescriptorSetLayoutState = struct {
    layout: vk.DescriptorSetLayout,
    /// Owned copy of the binding declarations.
    bindings: []DescriptorSetLayout.Binding,
};

/// A descriptor set and the pool backing it.
pub const DescriptorSetState = struct {
    set: vk.DescriptorSet,
    pool: vk.DescriptorPool,
    layout: Resource,
};

/// A pipeline layout and the resources it references.
pub const PipelineLayoutState = struct {
    layout: vk.PipelineLayout,
    sets: []Resource,
    push_constants: []PipelineLayout.PushConstantRange,
};

/// A pipeline object.
pub const PipelineState = struct {
    pipeline: vk.Pipeline,
    layout: Resource,
    bind_point: vk.PipelineBindPoint,
};

/// Phase of a command buffer.
pub const Phase = enum {
    initial,
    recording,
    ended,
};

/// One command stored for a secondary command buffer.
pub const StoredCommand = struct {
    /// Deep copy of the recorded command.
    cmd: Device.Command,
};

/// A command pool and its child buffers.
pub const CommandPoolState = struct {
    pool: vk.CommandPool,
    queue_family: u32,
    reset_individual: bool,
    /// Handles of buffers allocated from this pool.
    buffers: std.ArrayListUnmanaged(Resource) = .empty,
};

/// A command buffer and its recording state.
pub const CommandBufferState = struct {
    /// Native command buffer.
    native: vk.CommandBuffer,
    /// Pool that owns the buffer.
    pool: Resource,
    /// Primary or secondary.
    level: CommandBuffer.Level,
    /// Current recording phase.
    phase: Phase = .initial,
    /// Hint passed to `Command.begin`.
    one_time: bool = false,
    /// True while a dynamic render pass is active.
    in_render_pass: bool = false,
    /// Pipeline bound by the latest `bind_pipeline`.
    pipeline: ?Resource = null,
    /// Layout of `pipeline`.
    pipeline_layout: ?Resource = null,
    /// Kind of `pipeline`.
    pipeline_kind: ?vk.PipelineBindPoint = null,
    /// True once a viewport was set.
    viewport_set: bool = false,
    /// True once a scissor was set.
    scissor_set: bool = false,
    /// True once an index buffer was set.
    index_buffer: bool = false,
    /// Work kinds recorded so far.
    kinds: Device.QueueKind = .{},
    /// Commands of a secondary buffer, replayed by
    /// `execute_commands`.
    stored: std.ArrayListUnmanaged(StoredCommand) = .empty,
    /// Owns the deep copies stored in `stored`.
    arena: std.heap.ArenaAllocator,
};

/// A fence object.
pub const FenceState = struct {
    fence: vk.Fence,
};

/// A semaphore object.
pub const SemaphoreState = struct {
    semaphore: vk.Semaphore,
    kind: Resource.Semaphore.Kind,
};

/// A presentation surface.
pub const SurfaceState = struct {
    surface: vk.SurfaceKHR,
    platform: Surface.Platform,
};

/// A swapchain and the textures wrapping its images.
pub const SwapchainState = struct {
    swapchain: vk.SwapchainKHR,
    surface: Resource,
    format: vk.Format,
    extent: vk.Extent2D,
    usage: Texture.Usage,
    /// Handles of textures owned by this swapchain.
    images: []Resource,
    /// True for images acquired and not presented.
    acquired: []bool,
    /// Family that accepted the swapchain.
    queue_family: u32,
};

/// Maps an RHI handle kind to its state type.
///
/// # Args
/// - `kind`: resource kind.
///
/// # Return
/// The `Data` field type storing that kind.
pub fn StateFor(comptime kind: Resource.Kind) type {
    return @FieldType(Data, @tagName(kind));
}
