//! A GPU/CPU synchronization point.
//!
//! A fence is signalled when submitted work completes.
//! The host waits on it with `Queue.wait` and resets it
//! with `reset`.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Fence = @This();
const std = @import("std");

/// Initial state of a fence.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Start in the signalled state.
    signaled: bool = false,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a fence on `device`.
///
/// # Args
/// - `device`: the device that owns the fence.
/// - `desc`: initial state.
///
/// # Return
/// A live fence; release it with `destroy`.
///
/// # Error
/// - `error.TooManyObjects`: the backend cannot track
///   another fence.
/// - `error.Unsupport`: the backend cannot create
///   fences.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Fence {
    const rc_desc: Resource.Desc = .{ .fence = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the fence.
///
/// # Note
/// The fence must not be waited on by another thread.
/// The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `fence`.
pub fn destroy(self: Fence) void {
    std.debug.assert(self.handle.kind == .fence);
    self.device.rawDestroy(self.handle, @returnAddress());
}

/// Resets the fence to the unsignalled state.
///
/// # Error
/// - `error.InvalidResource`: the handle is not live.
/// - `error.InvalidState`: the fence is referenced
///   by pending work or waiters.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Panic
/// In safety builds, the handle kind is not `fence`.
pub fn reset(self: Fence) Device.DeviceOpError!void {
    std.debug.assert(self.handle.kind == .fence);
    try self.device.rawDeviceOp(
        .{ .reset_fence = self },
        null,
        @returnAddress(),
    );
}
