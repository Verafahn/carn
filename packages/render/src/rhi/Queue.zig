const Device = @import("Device.zig");
const Error = Device.Error;
const Resource = @import("Resource.zig");
const Queue = @This();

vptr: *anyopaque,
vtable: *const VTable,
/// The queue family this queue belongs to.
family_index: u32,
/// The index of this queue inside its family.
queue_index: u32,

pub const VTable = struct {
    submit: *const fn (
        *anyopaque,
        submits: []const SubmitInfo,
        ret_addr: usize,
    ) SubmitError!void,

    present: *const fn (
        *anyopaque,
        swapchain: Resource,
        image_index: u32,
        wait_ops: []const SyncOp,
        ret_addr: usize,
    ) PresentError!PresentResult,

    wait: *const fn (
        *anyopaque,
        syncs: []const SyncOp,
        wait_any: bool,
        timeout_ns: ?u64,
        ret_addr: usize,
    ) WaitError!WaitResult,
};

pub inline fn submit(self: Queue, submits: []const SubmitInfo, ret_addr: usize) SubmitError!void {
    try self.vtable.submit(self.vptr, submits, ret_addr);
}

pub inline fn present(
    self: Queue,
    swapchain: Resource,
    image_index: u32,
    wait_ops: []const SyncOp,
    ret_addr: usize,
) PresentError!PresentResult {
    return try self.vtable.present(
        self.vptr,
        swapchain,
        image_index,
        wait_ops,
        ret_addr,
    );
}

pub inline fn wait(
    self: Queue,
    syncs: []const SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
    ret_addr: usize,
) WaitError!WaitResult {
    return try self.vtable.wait(
        self.vptr,
        syncs,
        wait_any,
        timeout_ns,
        ret_addr,
    );
}

pub const SubmitInfo = struct {};

pub const SyncOp = struct {};

pub const PresentResult = struct {};

pub const WaitResult = struct {};

pub const SubmitError = error{
    /// A submit description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the wrong kind.
    InvalidResource,

    /// The queue is not able to accept submissions
    /// (for example because it was destroyed with its device).
    QueueState,
} || Error;

pub const PresentError = error{
    /// The swapchain handle is not live or is not a swapchain.
    InvalidResource,

    /// The surface backing the swapchain has been lost.
    SurfaceLost,

    /// The surface's presentation engine does not support the
    /// request.
    SurfaceNotSupported,
} || Error;

pub const WaitError = error{
    /// A referenced synchronisation object is not live or has
    /// the wrong kind.
    InvalidResource,
} || Error;
