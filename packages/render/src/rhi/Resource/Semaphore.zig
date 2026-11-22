//! A GPU/GPU synchronization point.
//!
//! A semaphore orders work across queues. Binary
//! semaphores signal one submission at a time; timeline
//! semaphores carry a monotonically increasing value.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Semaphore = @This();
const std = @import("std");

/// Kind and initial value of a semaphore.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Semaphore behavior.
    kind: Kind = .binary,
    /// Initial timeline value; ignored by binary
    /// semaphores.
    value: u64 = 0,
};

/// Synchronization behavior of a semaphore.
pub const Kind = enum {
    /// Signals one operation and resets itself.
    binary,
    /// Holds a monotonically increasing value.
    timeline,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a semaphore on `device`.
///
/// # Args
/// - `device`: the device that owns the semaphore.
/// - `desc`: kind and initial value.
///
/// # Return
/// A live semaphore; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the kind or initial value
///   is rejected.
/// - `error.TooManyObjects`: the backend cannot track
///   another semaphore.
/// - `error.Unsupport`: the backend cannot create
///   timeline semaphores.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Semaphore {
    const rc_desc: Resource.Desc = .{ .semaphore = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the semaphore.
///
/// # Note
/// The semaphore must not be waited on or signalled by
/// pending work. The handle must not be used
/// afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `semaphore`.
pub fn destroy(self: Semaphore) void {
    std.debug.assert(self.handle.kind == .semaphore);
    self.device.rawDestroy(self.handle, @returnAddress());
}
