//! The backend-independent device.
//!
//! A `Device` is an opaque backend pointer plus a
//! vtable. Resources are created through `rawCreate` and
//! destroyed through `rawDestroy`; the typed wrappers in
//! `Resource` call those functions.
//!
//! Work reaches the GPU through four closed operation
//! sets:
//!
//! - `Command`: recorded into a command buffer.
//! - `DeviceOp`: executed immediately on the device.
//! - `Queue.submit`, `Queue.present` and `Queue.wait`.
//! - `Feature`: capability and limit queries.
//!
//! The sets are exhaustive. Every backend implements
//! every operation; a device that cannot execute one
//! reports `error.Unsupport` instead of silently
//! approximating it. Adding an operation is a breaking
//! interface change.
//!
//! The `raw*` functions do not validate arguments. The
//! typed wrappers check handle kinds in safety builds,
//! and a validation layer owns the full checks. Backends
//! receive `ret_addr` so diagnostics point at the call
//! site.

const std = @import("std");
const Resource = @import("Resource.zig");
const Queue = @import("Queue.zig");
const Device = @This();

const Texture = Resource.Texture;
const Buffer = Resource.Buffer;
const Sampler = Resource.Sampler;
const Swapchain = Resource.Swapchain;
const CommandPool = Resource.CommandPool;
const CommandBuffer = Resource.CommandBuffer;
const DescriptorSet = Resource.DescriptorSet;
const DescriptorSetLayout = Resource.DescriptorSetLayout;
const Fence = Resource.Fence;
const PipelineGraphics = Resource.PipelineGraphics;
const PipelineCompute = Resource.PipelineCompute;
const Semaphore = Resource.Semaphore;
const Shader = Resource.Shader;

/// Errors shared by every operation.
pub const Error = error{
    /// The operation, or a feature it requires, is not
    /// supported by this backend, device, or object.
    Unsupport,

    /// The device has been lost.
    ///
    /// All further use of the device and its resources
    /// is undefined; the application must destroy and
    /// recreate it.
    DeviceLost,

    /// The device failed for a reason other than
    /// being lost.
    DeviceFailure,

    /// An argument failed validation.
    ///
    /// The exact failure is intentionally not exposed:
    /// the RHI validation layer reports the details,
    /// and backends must not invent additional argument
    /// errors.
    InvalidArgument,

    /// The object is in a state that does not permit
    /// this operation, or the operation would violate
    /// the ordering rules of the interface.
    InvalidState,

    /// A backend violated the RHI contract.
    ///
    /// For example by returning a handle whose tag does
    /// not match the supplied description.
    BackendFailure,
} || std.mem.Allocator.Error;

/// Errors that report a failed resource creation.
pub const CreateError = error{
    /// The resource description is invalid.
    InvalidDesc,

    /// A resource referenced by the description is not
    /// live or has the wrong kind.
    InvalidResource,

    /// Device memory is exhausted.
    ///
    /// This is distinct from `OutOfMemory`, which is
    /// about host memory.
    OutOfDeviceMemory,

    /// The backend cannot track another live resource
    /// of this kind.
    TooManyObjects,
} || Error;

/// Errors that report a failed resource mapping.
pub const MapError = error{
    /// The requested mapping is invalid.
    InvalidDesc,

    /// The handle does not refer to a live resource.
    InvalidResource,

    /// The resource was not created with host-visible
    /// memory.
    NotHostVisible,

    /// The resource is already mapped.
    AlreadyMapped,

    /// The requested range lies outside of the
    /// resource.
    OutOfRange,
} || Error;

/// Errors that report a failed command recording.
pub const RecordError = error{
    /// The description of the command is invalid.
    InvalidDesc,

    /// A resource referenced by the command is not
    /// live or has the wrong kind.
    InvalidResource,

    /// The command list is not currently recording.
    CommandBufferState,
} || Error;

/// Errors that report a failed feature query.
pub const QueryError = error{
    /// `feature_index` is outside of the range
    /// accepted by the feature.
    IndexOutOfBounds,

    /// The query's resource handle is not live or has
    /// the wrong kind.
    InvalidResource,

    /// `out` is too small, or a required output is
    /// missing.
    InsufficientBuffer,
} || Error;

/// Errors that report a failed queue lookup.
pub const GetQueueError = error{
    /// The device exposes no queues.
    NoQueues,

    /// `index` is outside of the flat queue list.
    IndexOutOfBounds,
} || Error;

/// Errors that report a failed immediate operation.
pub const DeviceOpError = error{
    /// The operation description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the
    /// wrong kind.
    InvalidResource,

    /// An index into a backend-side list is out of
    /// range.
    IndexOutOfBounds,

    /// An output buffer is too small, or an output was
    /// required and omitted.
    InsufficientBuffer,

    /// The device exposes no queue that can serve the
    /// operation.
    NoQueues,

    /// A command list is not in the state required by
    /// the operation.
    CommandBufferState,

    /// A timed operation expired before it could
    /// complete.
    Timeout,

    /// The current swapchain can no longer be used.
    SwapchainOutOfDate,

    /// The surface backing a swapchain is no longer
    /// available.
    SurfaceLost,
} || Error;

/// Host access requested for a mapped resource.
pub const MapFlags = packed struct {
    /// Allow the host to read the range.
    read: bool = false,
    /// Allow the host to write the range.
    write: bool = false,
    /// Keep the mapping across frames.
    persistent: bool = false,
};

/// A byte range written by the host while mapped.
pub const DirtyRange = struct {
    /// Byte offset from the start of the resource.
    offset: u64,
    /// Number of bytes written.
    size: u64,
};

/// One command recorded into a command buffer.
///
/// Commands execute in recording order. The command
/// and its slices are borrowed for the duration of
/// `CommandBuffer.record`; the backend copies what it
/// needs. Resources the command names must stay alive
/// until the recorded work completes.
///
/// # Note
/// The set of commands is exhaustive. Adding a variant
/// is a breaking interface change.
pub const Command = union(enum) {
    /// Starts recording on the current command buffer.
    begin: Begin,
    /// Finishes recording.
    end: void,
    /// Starts a dynamic render pass.
    begin_render_pass: RenderPass,
    /// Ends the active render pass.
    end_render_pass: void,
    /// Binds a graphics or compute pipeline.
    bind_pipeline: BindPipeline,
    /// Binds a descriptor set at a layout index.
    bind_descriptor_set: BindDescriptorSet,
    /// Binds vertex buffers starting at `first`.
    bind_vertex_buffers: BindVertexBuffers,
    /// Binds an index buffer.
    bind_index_buffer: BindIndexBuffer,
    /// Writes push constants.
    push_constants: PushConstants,
    /// Sets the dynamic viewport.
    set_viewport: Viewport,
    /// Sets the dynamic scissor rectangle.
    set_scissor: Rect,
    /// Draws non-indexed geometry.
    draw: Draw,
    /// Draws indexed geometry.
    draw_indexed: DrawIndexed,
    /// Draws non-indexed geometry from a buffer.
    draw_indirect: DrawIndirect,
    /// Draws indexed geometry from a buffer.
    draw_indexed_indirect: DrawIndirect,
    /// Executes recorded secondary command buffers.
    execute_commands: ExecuteCommands,
    /// Dispatches a compute grid.
    dispatch: Dispatch,
    /// Dispatches a compute grid from a buffer.
    dispatch_indirect: DispatchIndirect,
    /// Copies between buffers.
    copy_buffer: CopyBuffer,
    /// Copies a buffer into a texture.
    copy_buffer_to_texture: CopyBufferTexture,
    /// Copies a texture into a buffer.
    copy_texture_to_buffer: CopyTextureBuffer,
    /// Copies between textures.
    copy_texture: CopyTexture,
    /// Scales and converts between textures.
    blit_texture: BlitTexture,
    /// Resolves a multisampled texture.
    resolve_texture: ResolveTexture,
    /// Fills a buffer with a 32-bit value.
    fill_buffer: FillBuffer,
    /// Inserts memory and layout barriers.
    barrier: Barrier,

    /// Parameters for `begin`.
    pub const Begin = struct {
        /// Hint that the buffer is submitted once; the
        /// backend may use a one-shot allocation path.
        one_time: bool = false,
    };

    /// An integer rectangle.
    pub const Rect = struct {
        /// Left edge, in pixels.
        x: i32 = 0,
        /// Top edge, in pixels.
        y: i32 = 0,
        /// Rectangle width, in pixels.
        width: u32,
        /// Rectangle height, in pixels.
        height: u32,
    };

    /// A floating-point viewport.
    pub const Viewport = struct {
        /// Left edge, in pixels.
        x: f32 = 0,
        /// Top edge, in pixels.
        y: f32 = 0,
        /// Viewport width, in pixels.
        width: f32,
        /// Viewport height, in pixels.
        height: f32,
        /// Minimum depth written by clipping.
        depth_min: f32 = 0,
        /// Maximum depth written by clipping.
        depth_max: f32 = 1,
    };

    /// What a render pass does with previous contents.
    pub const Load = enum {
        /// Preserve the previous contents.
        load,
        /// Clear to the attachment's clear value.
        clear,
        /// Leave the contents undefined.
        dont_care,
    };

    /// What a render pass does with its output.
    pub const Store = enum {
        /// Keep the rendered contents.
        store,
        /// Leave the contents undefined.
        dont_care,
    };

    /// A color attachment clear value.
    pub const ClearColor = union(enum) {
        /// Floating-point channels.
        float: [4]f32,
        /// Unsigned integer channels.
        uint: [4]u32,
        /// Signed integer channels.
        sint: [4]i32,
    };

    /// A color attachment of a dynamic render pass.
    pub const ColorAttachment = struct {
        /// Texture receiving the rendered color.
        texture: Texture,
        /// Single-sampled texture that receives the
        /// resolved contents of a multisampled
        /// `texture`; null skips the resolve.
        resolve: ?Texture = null,
        /// Treatment of the previous contents.
        load: Load = .load,
        /// Treatment of the rendered contents.
        store: Store = .store,
        /// Clear value when `load` is `clear`.
        clear: ClearColor = .{ .float = .{ 0, 0, 0, 0 } },
    };

    /// A depth-stencil attachment of a render pass.
    pub const DepthStencilAttachment = struct {
        /// Texture receiving depth and stencil.
        texture: Texture,
        /// Treatment of the previous contents.
        load: Load = .load,
        /// Treatment of the rendered contents.
        store: Store = .store,
        /// Depth clear value when `load` is `clear`.
        depth_clear: f32 = 1,
        /// Stencil clear value when `load` is `clear`.
        stencil_clear: u8 = 0,
    };

    /// Parameters for `begin_render_pass`.
    pub const RenderPass = struct {
        /// Rendered area, in pixels.
        area: Rect,
        /// Color attachments, in slot order.
        color: []const ColorAttachment = &.{},
        /// Depth-stencil attachment; null renders
        /// without one.
        depth_stencil: ?DepthStencilAttachment = null,
    };

    /// Parameters for `bind_pipeline`.
    pub const BindPipeline = union(enum) {
        /// A graphics pipeline.
        graphics: PipelineGraphics,
        /// A compute pipeline.
        compute: PipelineCompute,
    };

    /// Parameters for `bind_descriptor_set`.
    pub const BindDescriptorSet = struct {
        /// Layout index the set is bound to.
        index: u32,
        /// The set to bind.
        set: DescriptorSet,
    };

    /// Parameters for `bind_vertex_buffers`.
    pub const BindVertexBuffers = struct {
        /// First vertex input slot written.
        first: u32 = 0,
        /// Buffers and offsets to bind in order.
        buffers: []const Binding,

        /// One vertex buffer binding.
        pub const Binding = struct {
            /// Buffer holding vertex data.
            buffer: Buffer,
            /// Byte offset of the first vertex.
            offset: u64 = 0,
        };
    };

    /// Parameters for `bind_index_buffer`.
    pub const BindIndexBuffer = struct {
        /// Buffer holding index data.
        buffer: Buffer,
        /// Byte offset of the first index.
        offset: u64 = 0,
        /// Width of one index.
        format: IndexFormat = .u32,
    };

    /// Width of one index buffer element.
    pub const IndexFormat = enum {
        /// 16-bit unsigned index.
        u16,
        /// 32-bit unsigned index.
        u32,
    };

    /// Parameters for `push_constants`.
    pub const PushConstants = struct {
        /// Stages that read the constants.
        stages: Shader.StageFlags,
        /// Byte offset inside the push constant window.
        offset: u32 = 0,
        /// Constant bytes; borrowed by the command.
        data: []const u8,
    };

    /// A half-open numeric range.
    pub const Range = struct {
        /// First element.
        first: u32 = 0,
        /// Number of elements.
        count: u32,
    };

    /// Parameters for `draw`.
    pub const Draw = struct {
        /// Vertex range to draw.
        vertices: Range,
        /// Instance range to draw.
        instances: Range = .{ .first = 0, .count = 1 },
    };

    /// Parameters for `draw_indexed`.
    pub const DrawIndexed = struct {
        /// Index range to draw.
        indices: Range,
        /// Value added to every index.
        base_vertex: i32 = 0,
        /// Instance range to draw.
        instances: Range = .{ .first = 0, .count = 1 },
    };

    /// Parameters for indirect draws.
    pub const DrawIndirect = struct {
        /// Buffer holding draw records.
        buffer: Buffer,
        /// Byte offset of the first record.
        offset: u64 = 0,
        /// Number of records to consume.
        count: u32,
        /// Byte stride between records; 0 selects the
        /// natural stride of the command.
        stride: u32 = 0,
    };

    /// Parameters for `execute_commands`.
    pub const ExecuteCommands = struct {
        /// Secondary buffers executed in order.
        commands: []const CommandBuffer,
    };

    /// Parameters for `dispatch`.
    pub const Dispatch = struct {
        /// Workgroup counts, per axis.
        groups: [3]u32,
    };

    /// Parameters for `dispatch_indirect`.
    pub const DispatchIndirect = struct {
        /// Buffer holding one dispatch record.
        buffer: Buffer,
        /// Byte offset of the record.
        offset: u64 = 0,
    };

    /// Parameters for `copy_buffer`.
    pub const CopyBuffer = struct {
        /// Buffer read by the copy.
        src: Buffer,
        /// Buffer written by the copy.
        dst: Buffer,
        /// Byte regions to copy.
        regions: []const BufferCopy,
    };

    /// One byte range of a buffer copy.
    pub const BufferCopy = struct {
        /// First byte read from the source.
        src_offset: u64 = 0,
        /// First byte written in the target.
        dst_offset: u64 = 0,
        /// Number of bytes to copy.
        size: u64,
    };

    /// A region of one texture subresource.
    pub const TextureRegion = struct {
        /// Mip level.
        mip: u16 = 0,
        /// Array layer or depth slice.
        layer: u16 = 0,
        /// Texel offset inside the subresource.
        offset: [3]u32 = .{ 0, 0, 0 },
        /// Texel size of the region, per axis.
        extent: [3]u32,
    };

    /// Parameters for `copy_buffer_to_texture`.
    pub const CopyBufferTexture = struct {
        /// Buffer read by the copy.
        src: Buffer,
        /// Texture written by the copy.
        dst: Texture,
        /// Regions to copy.
        regions: []const Region,

        /// One buffer-to-texture region.
        pub const Region = struct {
            /// First source byte in `src`.
            buffer_offset: u64 = 0,
            /// Byte stride between texel rows; 0 means
            /// tightly packed.
            buffer_stride: u32 = 0,
            /// Destination subresource and extent.
            region: TextureRegion,
        };
    };

    /// Parameters for `copy_texture_to_buffer`.
    pub const CopyTextureBuffer = struct {
        /// Texture read by the copy.
        src: Texture,
        /// Buffer written by the copy.
        dst: Buffer,
        /// Regions to copy.
        regions: []const Region,

        /// One texture-to-buffer region.
        pub const Region = struct {
            /// First target byte in `dst`.
            buffer_offset: u64 = 0,
            /// Byte stride between texel rows; 0 means
            /// tightly packed.
            buffer_stride: u32 = 0,
            /// Source subresource and extent.
            region: TextureRegion,
        };
    };

    /// One texture-to-texture region pair.
    pub const TextureCopy = struct {
        /// Source subresource and extent.
        src: TextureRegion,
        /// Destination subresource and extent.
        dst: TextureRegion,
    };

    /// Parameters for `copy_texture`.
    pub const CopyTexture = struct {
        /// Texture read by the copy.
        src: Texture,
        /// Texture written by the copy.
        dst: Texture,
        /// Region pairs to copy.
        regions: []const TextureCopy,
    };

    /// Parameters for `blit_texture`.
    pub const BlitTexture = struct {
        /// Texture read by the blit.
        src: Texture,
        /// Texture written by the blit.
        dst: Texture,
        /// Region pairs to blit.
        regions: []const TextureCopy,
        /// Interpolation used when scaling.
        filter: Sampler.Filter = .linear,
    };

    /// Parameters for `resolve_texture`.
    pub const ResolveTexture = struct {
        /// Multisampled source texture.
        src: Texture,
        /// Single-sampled destination texture.
        dst: Texture,
        /// Region pairs to resolve.
        regions: []const TextureCopy,
    };

    /// Parameters for `fill_buffer`.
    pub const FillBuffer = struct {
        /// Buffer to fill.
        buffer: Buffer,
        /// First byte to write.
        offset: u64 = 0,
        /// Bytes to fill; 0 means the rest of the
        /// buffer.
        size: u64 = 0,
        /// 32-bit value written repeatedly.
        value: u32 = 0,
    };

    /// Parameters for `barrier`.
    pub const Barrier = struct {
        /// Buffer barriers, applied in order.
        buffers: []const BufferBarrier = &.{},
        /// Texture barriers, applied after buffers.
        textures: []const TextureBarrier = &.{},
    };

    /// A memory and execution barrier for a buffer.
    pub const BufferBarrier = struct {
        /// Buffer covered by the barrier.
        buffer: Buffer,
        /// First byte covered.
        offset: u64 = 0,
        /// Bytes covered; 0 means the rest.
        size: u64 = 0,
        /// State before the barrier.
        src: Sync,
        /// State after the barrier.
        dst: Sync,
    };

    /// A memory, execution and layout barrier for a
    /// texture.
    pub const TextureBarrier = struct {
        /// Texture covered by the barrier.
        texture: Texture,
        /// Subresources covered.
        range: Texture.SubresourceRange = .{},
        /// State before the barrier.
        src: Sync,
        /// State after the barrier.
        dst: Sync,
        /// Layout after the barrier.
        layout: Texture.Layout = .general,
    };

    /// Stage and access scopes of a barrier.
    pub const Sync = struct {
        /// Pipeline stages in the scope.
        stages: Stage = .{},
        /// Memory accesses in the scope.
        access: Access = .{},
    };

    /// Pipeline stages used by a barrier.
    pub const Stage = packed struct {
        /// Covers every stage when true.
        all: bool = false,
        /// Vertex shader stage.
        vertex: bool = false,
        /// Fragment shader stage.
        fragment: bool = false,
        /// Compute shader stage.
        compute: bool = false,
        /// Copy, blit and resolve stages.
        transfer: bool = false,
        /// Host reads and writes.
        host: bool = false,
        /// Color attachment output.
        color_attachment: bool = false,
        /// Depth-stencil attachment output.
        depth_stencil: bool = false,
    };

    /// Memory accesses used by a barrier.
    pub const Access = packed struct {
        /// Covers every access when true.
        all: bool = false,
        /// Uniform buffer reads.
        uniform_read: bool = false,
        /// Storage buffer or image reads.
        storage_read: bool = false,
        /// Storage buffer or image writes.
        storage_write: bool = false,
        /// Vertex buffer reads.
        vertex_read: bool = false,
        /// Index buffer reads.
        index_read: bool = false,
        /// Indirect argument reads.
        indirect_read: bool = false,
        /// Transfer reads.
        transfer_read: bool = false,
        /// Transfer writes.
        transfer_write: bool = false,
        /// Color attachment reads.
        color_read: bool = false,
        /// Color attachment writes.
        color_write: bool = false,
        /// Depth-stencil reads.
        depth_read: bool = false,
        /// Depth-stencil writes.
        depth_write: bool = false,
        /// Host reads of coherent memory.
        host_read: bool = false,
        /// Host writes of coherent memory.
        host_write: bool = false,
    };
};

/// A capability or limit query.
///
/// The active field selects what is tested or reported.
/// `query` returns the verdict and optionally writes a
/// result structure into its `out` buffer.
///
/// # Note
/// The set of queries is exhaustive. Adding a variant
/// is a breaking interface change.
pub const Feature = union(enum) {
    /// Tests whether a format supports a usage set.
    ///
    /// `feature_index` must be zero. `out` is ignored
    /// and may be null.
    format: FormatQuery,
    /// Tests whether a texture description would be
    /// supported.
    ///
    /// `feature_index` must be zero. `out` is ignored
    /// and may be null.
    texture: Texture.Desc,
    /// Tests whether a surface accepts a presentation
    /// format and mode.
    ///
    /// `feature_index` must be zero. `handle` must be
    /// a live surface; `out` is ignored and may be
    /// null.
    swapchain: SwapchainQuery,
    /// Reports properties of queue family
    /// `feature_index`.
    ///
    /// `out` is required and receives a
    /// `FeatureOut.queue_family` value;
    /// `error.IndexOutOfBounds` reports a missing
    /// family.
    queue_family: void,
    /// Reports the device limits.
    ///
    /// `feature_index` must be zero. `out` is required
    /// and receives a `FeatureOut.limits` value.
    limits: void,
};

/// Parameters of a `Feature.format` query.
pub const FormatQuery = struct {
    /// Format to test.
    format: Resource.Format,
    /// Usages that must all be supported.
    usage: Texture.Usage,
};

/// Parameters of a `Feature.swapchain` query.
pub const SwapchainQuery = struct {
    /// Presentation format to test.
    format: Resource.Format,
    /// Presentation mode to test.
    mode: Swapchain.PresentMode,
};

/// The value `query` writes into its output buffer.
///
/// # Note
/// The active field must match the active `Feature`
/// field. Features that only return a `Support` verdict
/// have a `void` field and need no output buffer.
pub const FeatureOut = union(enum) {
    /// No output value.
    format: void,
    /// No output value.
    texture: void,
    /// No output value.
    swapchain: void,
    /// Properties of one queue family.
    queue_family: QueueFamily,
    /// Device limits.
    limits: Limits,
};

/// Work kinds a queue family accepts.
pub const QueueKind = packed struct {
    /// Graphics and compute pipeline work.
    graphics: bool = false,
    /// Compute pipeline work.
    compute: bool = false,
    /// Copy, blit and resolve work.
    transfer: bool = false,
    /// Presentation to a surface.
    present: bool = false,
};

/// Properties of one queue family.
pub const QueueFamily = struct {
    /// Number of queues in the family.
    count: u32,
    /// Work kinds the family accepts.
    kinds: QueueKind,
};

/// Limits reported by `Feature.limits`.
pub const Limits = struct {
    /// Largest texture extent along x for 1D.
    max_texture_extent_1d: u32,
    /// Largest texture extent per axis for 2D.
    max_texture_extent_2d: u32,
    /// Largest texture extent per axis for 3D.
    max_texture_extent_3d: u32,
    /// Largest number of array layers.
    max_texture_layers: u32,
    /// Largest number of mip levels.
    max_texture_mips: u16,
    /// Largest vertex buffer stride, in bytes.
    max_vertex_stride: u32,
    /// Largest number of vertex buffers.
    max_vertex_buffers: u32,
    /// Largest number of descriptor sets per pipeline
    /// layout.
    max_descriptor_sets: u32,
    /// Largest push constant window, in bytes.
    max_push_constant_size: u32,
    /// Largest number of storage buffer bindings.
    max_storage_buffers: u32,
    /// Largest number of sampled textures.
    max_sampled_textures: u32,
    /// Largest compute workgroup size, per axis.
    max_workgroup_size: [3]u32,
    /// Largest workgroup storage, in bytes.
    max_workgroup_memory: u32,
};

/// The support verdict of a feature query.
pub const Support = enum {
    /// The backend cannot answer without more
    /// context; the output buffer is untouched.
    unknown,
    /// The feature is unavailable.
    unsupported,
    /// The feature is available.
    supported,
};

/// An operation executed immediately on the device.
///
/// Immediate operations are not recorded into a command
/// buffer. They complete before the call returns, or
/// report `error.Timeout` when they are timed.
///
/// # Note
/// The set of operations is exhaustive. Adding a
/// variant is a breaking interface change.
pub const DeviceOp = union(enum) {
    /// Blocks until all submitted work completes.
    wait_idle: void,
    /// Acquires the next image of a swapchain.
    ///
    /// `out` receives the image index as a `u32`.
    acquire: Acquire,
    /// Returns the texture that wraps a swapchain image.
    ///
    /// The texture is owned by the swapchain and must
    /// not be destroyed. `out` receives a `Resource`.
    swapchain_image: SwapchainImage,
    /// Resets a pool and every buffer allocated from it.
    ///
    /// The pool must have no pending work.
    reset_command_pool: CommandPool,
    /// Resets one command buffer to its initial state.
    ///
    /// The buffer must have no pending execution.
    reset_command_buffer: CommandBuffer,
    /// Resets a fence to the unsignalled state.
    ///
    /// No thread may be waiting on the fence.
    reset_fence: Fence,
    /// Writes descriptors into a set.
    update_descriptor_set: DescriptorWrite,
};

/// Parameters of `DeviceOp.acquire`.
pub const Acquire = struct {
    /// Swapchain to acquire from.
    swapchain: Swapchain,
    /// Semaphore signalled when the image is
    /// available; null signals nothing.
    signal: ?Semaphore = null,
    /// Fence signalled when the image is available;
    /// null signals nothing.
    fence: ?Fence = null,
    /// Nanosecond timeout; null waits forever.
    timeout_ns: ?u64 = null,
};

/// Parameters of `DeviceOp.update_descriptor_set`.
pub const DescriptorWrite = struct {
    /// Set to update; it must not be in use.
    set: DescriptorSet,
    /// Writes applied in order.
    writes: []const DescriptorSet.Write,
};

/// Parameters of `DeviceOp.swapchain_image`.
pub const SwapchainImage = struct {
    /// Swapchain that owns the image.
    swapchain: Swapchain,
    /// Image index in `0..image_count`.
    index: u32,
};

/// Backend state; interpreted only by `vtable`.
vptr: *anyopaque,
/// Backend implementation of the RHI.
vtable: *const VTable,

/// The backend implementation of the device.
///
/// # Note
/// Every entry receives the backend pointer stored in
/// `Device.vptr`. Entries must not retain borrowed
/// slices after returning, except where documented on
/// the individual entry.
pub const VTable = struct {
    /// Creates a resource and returns its handle.
    ///
    /// # Args
    /// - `desc`: description to interpret; its tag
    ///   selects the resource kind.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Return
    /// A live handle whose `kind` matches `desc`.
    ///
    /// # Error
    /// - `error.InvalidDesc`: the description is
    ///   rejected.
    /// - `error.InvalidResource`: a resource
    ///   referenced by `desc` is not live or has the
    ///   wrong kind.
    /// - `error.OutOfDeviceMemory`: device memory is
    ///   exhausted.
    /// - `error.TooManyObjects`: the backend cannot
    ///   track another handle.
    /// - `error.Unsupport`: the backend cannot create
    ///   this kind.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    create: *const fn (
        *anyopaque,
        desc: *const Resource.Desc,
        ret_addr: usize,
    ) CreateError!Resource,

    /// Destroys a resource and releases backend state.
    ///
    /// # Args
    /// - `handle`: live handle returned by `create`.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Note
    /// The caller guarantees that no pending work
    /// references the resource. The backend may defer
    /// release internally; the handle is dead after
    /// this call.
    destroy: *const fn (
        *anyopaque,
        handle: Resource,
        ret_addr: usize,
    ) void,

    /// Maps a host-visible resource into memory.
    ///
    /// # Args
    /// - `handle`: live resource to map.
    /// - `offset`: byte offset of the range.
    /// - `size`: byte length of the range.
    /// - `flags`: requested host access.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Return
    /// Host memory for the range; it stays valid until
    /// `unmap`.
    ///
    /// # Error
    /// - `error.InvalidDesc`: the range or flags are
    ///   malformed.
    /// - `error.InvalidResource`: the handle is not
    ///   live.
    /// - `error.NotHostVisible`: the resource cannot be
    ///   mapped.
    /// - `error.AlreadyMapped`: the resource is mapped.
    /// - `error.OutOfRange`: the range leaves the
    ///   resource.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    map: *const fn (
        *anyopaque,
        handle: Resource,
        offset: u64,
        size: u64,
        flags: MapFlags,
        ret_addr: usize,
    ) MapError![]u8,

    /// Releases a mapping created by `map`.
    ///
    /// # Args
    /// - `handle`: mapped resource.
    /// - `dirty`: ranges written by the host, relative
    ///   to the start of the resource. May be empty.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Note
    /// The pointer returned by `map` is invalid after
    /// this call. `dirty` is a hint; coherent memory
    /// ignores it.
    unmap: *const fn (
        *anyopaque,
        handle: Resource,
        dirty: []const DirtyRange,
        ret_addr: usize,
    ) void,

    /// Appends one command to a command buffer.
    ///
    /// # Args
    /// - `handle`: command buffer receiving the
    ///   command.
    /// - `cmd`: command to record.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Error
    /// - `error.InvalidDesc`: the command payload is
    ///   malformed.
    /// - `error.InvalidResource`: a referenced resource
    ///   is not live or has the wrong kind.
    /// - `error.CommandBufferState`: the buffer is not
    ///   recording.
    /// - `error.Unsupport`: the device cannot execute
    ///   the command.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    ///
    /// # Note
    /// The command and its slices are borrowed only
    /// for this call. Resources the command names must
    /// stay alive until the recorded work completes.
    record: *const fn (
        *anyopaque,
        handle: Resource,
        cmd: *const Command,
        ret_addr: usize,
    ) RecordError!void,

    /// Answers a capability or limit query.
    ///
    /// # Args
    /// - `feature`: query to perform.
    /// - `feature_index`: index inside a repeated
    ///   result; 0 when the feature has no index.
    /// - `handle`: optional resource the query is
    ///   scoped to; null queries the device.
    /// - `out`: optional output buffer; required by
    ///   features that produce a value.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Return
    /// The support verdict. `unknown` means the backend
    /// cannot answer and leaves `out` untouched.
    ///
    /// # Error
    /// - `error.IndexOutOfBounds`: `feature_index`
    ///   selects a missing entry.
    /// - `error.InsufficientBuffer`: `out` is missing
    ///   or too small for the result.
    /// - `error.InvalidResource`: `handle` is not live
    ///   or has the wrong kind.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    query: *const fn (
        *anyopaque,
        feature: Feature,
        feature_index: u32,
        handle: ?Resource,
        out: ?[]u8,
        ret_addr: usize,
    ) QueryError!Support,

    /// Returns a queue by its flat index.
    ///
    /// # Args
    /// - `index`: position in the flat queue list.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Return
    /// A queue whose `family_index` and `queue_index`
    /// identify it inside the device.
    ///
    /// # Error
    /// - `error.NoQueues`: the device exposes no
    ///   queues.
    /// - `error.IndexOutOfBounds`: `index` is past the
    ///   queue list.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    get_queue: *const fn (
        *anyopaque,
        index: u32,
        ret_addr: usize,
    ) GetQueueError!Queue,

    /// Executes an immediate operation.
    ///
    /// # Args
    /// - `op`: operation to execute.
    /// - `out`: optional output buffer; required by
    ///   operations that return a value.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Error
    /// - `error.InvalidDesc`: the operation payload is
    ///   malformed.
    /// - `error.InvalidResource`: a referenced resource
    ///   is not live or has the wrong kind.
    /// - `error.IndexOutOfBounds`: an index selects a
    ///   missing entry.
    /// - `error.InsufficientBuffer`: `out` is missing
    ///   or too small for the result.
    /// - `error.NoQueues`: the device exposes no
    ///   queues.
    /// - `error.CommandBufferState`: a command buffer
    ///   is in the wrong state.
    /// - `error.Timeout`: a timed operation expired.
    /// - `error.SwapchainOutOfDate`: the swapchain is
    ///   stale.
    /// - `error.SurfaceLost`: the surface is gone.
    /// - `error.Unsupport`: the device cannot execute
    ///   the operation.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    ///
    /// # Note
    /// For `acquire`, `out` receives the image index
    /// as a `u32`; for `swapchain_image`, `out`
    /// receives a `Resource`.
    device_op: *const fn (
        *anyopaque,
        op: DeviceOp,
        out: ?[]u8,
        ret_addr: usize,
    ) DeviceOpError!void,
};

/// Creates a resource through the backend.
///
/// # Args
/// - `desc`: description passed to the backend.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The handle created by the backend.
///
/// # Error
/// - `error.InvalidDesc`, `error.InvalidResource`,
///   `error.OutOfDeviceMemory`,
///   `error.TooManyObjects` and `error.Unsupport`
///   when the backend rejects `desc`.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// This function does not validate `desc`. Prefer
/// `Resource.*.create` and the validation layer.
pub inline fn rawCreate(
    self: Device,
    desc: *const Resource.Desc,
    ret_addr: usize,
) CreateError!Resource {
    return try self.vtable.create(self.vptr, desc, ret_addr);
}

/// Destroys a resource through the backend.
///
/// # Args
/// - `handle`: live handle returned by `rawCreate`.
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// This function does not validate `handle`. Prefer
/// `Resource.*.destroy`.
pub inline fn rawDestroy(
    self: Device,
    handle: Resource,
    ret_addr: usize,
) void {
    self.vtable.destroy(self.vptr, handle, ret_addr);
}

/// Maps a host-visible resource through the backend.
///
/// # Args
/// - `handle`: live resource to map.
/// - `offset`: byte offset of the range.
/// - `size`: byte length of the range.
/// - `flags`: requested host access.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// Host memory for the range; valid until `rawUnmap`.
///
/// # Error
/// - `error.InvalidResource`, `error.NotHostVisible`,
///   `error.AlreadyMapped` and `error.OutOfRange`
///   when the mapping is rejected.
/// - `error.InvalidDesc`: the range or flags are
///   malformed.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// This function does not validate `handle`. Prefer
/// `Buffer.map`.
pub inline fn rawMap(
    self: Device,
    handle: Resource,
    offset: u64,
    size: u64,
    flags: MapFlags,
    ret_addr: usize,
) MapError![]u8 {
    return try self.vtable.map(
        self.vptr,
        handle,
        offset,
        size,
        flags,
        ret_addr,
    );
}

/// Releases a mapping through the backend.
///
/// # Args
/// - `handle`: mapped resource.
/// - `dirty`: ranges written by the host.
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// This function does not validate `handle`. Prefer
/// `Buffer.unmap`.
pub inline fn rawUnmap(
    self: Device,
    handle: Resource,
    dirty: []const DirtyRange,
    ret_addr: usize,
) void {
    self.vtable.unmap(self.vptr, handle, dirty, ret_addr);
}

/// Records a command through the backend.
///
/// # Args
/// - `handle`: command buffer receiving the command.
/// - `cmd`: command to record.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.InvalidDesc`, `error.InvalidResource`,
///   `error.CommandBufferState` and `error.Unsupport`
///   when the command cannot be recorded.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// This function does not validate `handle`. Prefer
/// `CommandBuffer.record`.
pub inline fn rawRecord(
    self: Device,
    handle: Resource,
    cmd: *const Command,
    ret_addr: usize,
) RecordError!void {
    try self.vtable.record(self.vptr, handle, cmd, ret_addr);
}

/// Queries a feature through the backend.
///
/// ```zig
/// var family: Device.QueueFamily = undefined;
/// _ = try device.rawQuery(
///     .queue_family,
///     0,
///     null,
///     std.mem.asBytes(&family),
///     @returnAddress(),
/// );
/// ```
///
/// # Args
/// - `feature`: query to perform.
/// - `feature_index`: index inside a repeated
///   result; 0 when the feature has no index.
/// - `handle`: optional resource the query is
///   scoped to; null queries the device.
/// - `out`: optional output buffer; required by
///   features that produce a value.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The support verdict. `unknown` leaves `out`
/// untouched.
///
/// # Error
/// - `error.IndexOutOfBounds`: `feature_index`
///   selects a missing entry.
/// - `error.InsufficientBuffer`: `out` is missing
///   or too small for the result.
/// - `error.InvalidResource`: `handle` is not live
///   or has the wrong kind.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// This function does not validate `handle`. Size
/// `out` with the `FeatureOut` field that matches
/// `feature`.
pub inline fn rawQuery(
    self: Device,
    feature: Feature,
    feature_index: u32,
    handle: ?Resource,
    out: ?[]u8,
    ret_addr: usize,
) QueryError!Support {
    return try self.vtable.query(
        self.vptr,
        feature,
        feature_index,
        handle,
        out,
        ret_addr,
    );
}

/// Looks up a queue through the backend.
///
/// # Args
/// - `index`: position in the flat queue list.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A queue that identifies its family and index.
///
/// # Error
/// - `error.NoQueues`: the device exposes no
///   queues.
/// - `error.IndexOutOfBounds`: `index` is past the
///   queue list.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub inline fn rawGetQueue(
    self: Device,
    index: u32,
    ret_addr: usize,
) GetQueueError!Queue {
    return try self.vtable.get_queue(self.vptr, index, ret_addr);
}

/// Executes an immediate operation through the
/// backend.
///
/// # Args
/// - `op`: operation to execute.
/// - `out`: optional output buffer for operations
///   that return a value.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.InvalidDesc`, `error.InvalidResource`,
///   `error.IndexOutOfBounds`, `error.Timeout`,
///   `error.SwapchainOutOfDate`, `error.SurfaceLost`
///   and `error.InsufficientBuffer` when `op` cannot
///   complete.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// For `acquire`, `out` receives the image index as a
/// `u32`; for `swapchain_image`, `out` receives a
/// `Resource`.
pub inline fn rawDeviceOp(
    self: Device,
    op: DeviceOp,
    out: ?[]u8,
    ret_addr: usize,
) DeviceOpError!void {
    try self.vtable.device_op(self.vptr, op, out, ret_addr);
}

/// Blocks until all submitted work completes.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub inline fn waitIdle(
    self: Device,
    ret_addr: usize,
) DeviceOpError!void {
    try self.rawDeviceOp(.wait_idle, null, ret_addr);
}

pub const createBuffer = Resource.Buffer.create;
pub const createCommandBuffer = Resource.CommandBuffer.create;
pub const createCommandPool = Resource.CommandPool.create;
pub const createDescriptorSet = Resource.DescriptorSet.create;
pub const createDescriptorSetLayout = Resource.DescriptorSetLayout.create;
pub const createFence = Resource.Fence.create;
pub const createPipelineCompute = Resource.PipelineCompute.create;
pub const createPipelineGraphics = Resource.PipelineGraphics.create;
pub const createPipelineLayout = Resource.PipelineLayout.create;
pub const createSampler = Resource.Sampler.create;
pub const createSemaphore = Resource.Semaphore.create;
pub const createShader = Resource.Shader.create;
pub const createSurface = Resource.Surface.create;
pub const createSwapchain = Resource.Swapchain.create;
pub const createTexture = Resource.Texture.create;

test {
    _ = Resource;
}
