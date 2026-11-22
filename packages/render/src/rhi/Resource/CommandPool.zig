//! Allocator for command buffers.
//!
//! A pool groups command buffers that share a queue
//! family and a reset policy.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const CommandPool = @This();
const std = @import("std");

/// Queue family and reset policy of a command pool.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Queue family that executes the pool's command
    /// buffers.
    queue_family: u32 = 0,
    /// Allow resetting one buffer at a time.
    reset_individual: bool = false,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a command pool and attributes diagnostics
/// to `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the pool.
/// - `desc`: queue family and reset policy.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live pool; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!CommandPool {
    const rc_desc: Resource.Desc = .{ .command_pool = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a command pool on `device`.
///
/// # Args
/// - `device`: the device that owns the pool.
/// - `desc`: queue family and reset policy.
///
/// # Return
/// A live pool; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: `queue_family` is not
///   exposed by the device.
/// - `error.TooManyObjects`: the backend cannot track
///   another pool.
/// - `error.Unsupport`: the backend cannot create
///   command pools.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!CommandPool {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the pool and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// The lifetime contract matches `destroy`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_pool`.
pub fn destroyWithAddr(self: CommandPool, ret_addr: usize) void {
    std.debug.assert(self.handle.kind == .command_pool);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the pool and every buffer allocated from it.
///
/// # Note
/// Buffers from the pool must not be pending
/// execution. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_pool`.
pub fn destroy(self: CommandPool) void {
    self.destroyWithAddr(@returnAddress());
}

/// Resets the pool and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// The error contract matches `reset`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_pool`.
pub fn resetWithAddr(
    self: CommandPool,
    ret_addr: usize,
) Device.DeviceOpError!void {
    std.debug.assert(self.handle.kind == .command_pool);
    try self.device.rawDeviceOp(
        .{ .reset_command_pool = self },
        null,
        ret_addr,
    );
}

/// Resets the pool and every buffer allocated from it.
///
/// # Error
/// - `error.InvalidResource`: the handle is not live.
/// - `error.InvalidState`: a buffer from the pool is
///   still in use.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_pool`.
pub fn reset(self: CommandPool) Device.DeviceOpError!void {
    try self.resetWithAddr(@returnAddress());
}
