const std = @import("std");
const Resource = @import("Resource.zig");
const Queue = @import("Queue.zig");
const Device = @This();

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

    /// A backend violated the RHI contract
    ///
    /// For example by returning a handle whose tag does
    /// not match the supplied description.
    BackendFailure,
} || std.mem.Allocator.Error;

pub const CreateError = error{
    /// The resource description is invalid.
    InvalidDesc,

    /// Device memory is exhausted.
    ///
    /// This is distinct from `OutOfMemory`, which is about
    /// host memory.
    OutOfDeviceMemory,

    /// The backend cannot track another live resource of
    /// this kind.
    TooManyObjects,
} || Error;

pub const MapError = error{
    /// The requested mapping is invalid.
    InvalidDesc,

    /// The handle does not refer to a live resource.
    InvalidResource,

    /// The resource was not created with `host_visible` memory.
    NotHostVisible,

    /// The resource is already mapped.
    AlreadyMapped,

    /// The requested range lies outside of the resource.
    OutOfRange,
} || Error;

pub const RecordError = error{
    /// The description of the command is invalid.
    InvalidDesc,

    /// A resource referenced by the command is not live or has
    /// the wrong kind.
    InvalidResource,

    /// The command list is not currently recording.
    CommandBufferState,
} || Error;

pub const QueryError = error{
    /// `feature_index` is outside of the range accepted by the
    /// feature.
    IndexOutOfBounds,

    /// `out_len` is too small for the feature's output
    /// structure.
    InsufficientBuffer,
} || Error;

pub const GetQueueError = error{
    /// The device exposes no queues
    /// (a headless compute-only device).
    NoQueues,

    /// `index` is outside of the flat queue list.
    IndexOutOfBounds,
} || Error;

pub const DeviceOpError = error{
    /// The operation description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the wrong kind.
    InvalidResource,

    /// An index into a backend-side list is out of range.
    IndexOutOfBounds,

    /// An output buffer is too small, or an output was
    /// required and omitted.
    InsufficientBuffer,

    /// The device has no queue with the requested index.
    NoQueues,

    /// A command list is not in the state required by the
    /// operation.
    CommandBufferState,

    /// A timed operation expired before it could complete.
    Timeout,

    /// The current swapchain can no longer be used.
    SwapchainOutOfDate,

    /// The surface backing a swapchain is no longer available.
    SurfaceLost,
} || Error;

pub const MapFlags = packed struct {
    read: bool = false,
    write: bool = false,
    persistent: bool = false,
};

pub const DirtyRange = struct {
    offset: u64,
    size: u64,
};

pub const Command = union(enum) {};

pub const Feature = union {};

pub const Support = enum {
    unknown,
    unsupported,
    supported,
};

pub const DeviceOp = enum {};

vptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    create: *const fn (
        *anyopaque,
        desc: *const Resource.Desc,
        ret_addr: usize,
    ) CreateError!Resource,

    destory: *const fn (
        *anyopaque,
        handle: Resource,
        ret_addr: usize,
    ) void,

    map: *const fn (
        *anyopaque,
        handle: Resource,
        offset: u64,
        size: u64,
        flags: MapFlags,
        ret_addr: usize,
    ) MapError![]u8,

    unmap: *const fn (
        *anyopaque,
        handle: Resource,
        dirty: []const DirtyRange,
        ret_addr: usize,
    ) void,

    record: *const fn (
        *anyopaque,
        handle: Resource,
        cmd: *const Command,
        ret_addr: usize,
    ) RecordError!void,

    query: *const fn (
        *anyopaque,
        feature: Feature,
        feature_index: u32,
        handle: ?Resource,
        out: ?[]u8,
        ret_addr: usize,
    ) QueryError!Support,

    get_queue: *const fn (
        *anyopaque,
        index: u32,
        ret_addr: usize,
    ) GetQueueError!Queue,

    device_op: *const fn (
        *anyopaque,
        op: DeviceOp,
        out: ?[]u8,
        ret_addr: usize,
    ) DeviceOpError!void,
};

pub inline fn rawCreate(self: Device, desc: *const Resource.Desc, ret_addr: usize) CreateError!Resource {
    return try self.vtable.create(self.vptr, desc, ret_addr);
}

pub inline fn rawDestroy(self: Device, handle: Resource, ret_addr: usize) void {
    self.vtable.destory(self.vptr, handle, ret_addr);
}

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

pub inline fn rawUnmap(self: Device, handle: Resource, dirty: []const DirtyRange, ret_addr: usize) void {
    self.vtable.unmap(self.vptr, handle, dirty, ret_addr);
}

pub inline fn rawRecord(self: Device, handle: Resource, cmd: *const Command, ret_addr: usize) RecordError!void {
    try self.vtable.record(self.vptr, handle, cmd, ret_addr);
}

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

pub inline fn rawGetQueue(self: Device, index: u32, ret_addr: usize) GetQueueError!Queue {
    return try self.vtable.get_queue(self.vptr, index, ret_addr);
}

pub inline fn rawDeviceOp(self: Device, op: DeviceOp, out: ?[]u8, ret_addr: usize) DeviceOpError!void {
    try self.vtable.device_op(self.vptr, op, out, ret_addr);
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
