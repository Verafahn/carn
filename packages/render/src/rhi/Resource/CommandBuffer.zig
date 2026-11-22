//! A recorded command list.
//!
//! Commands are appended with `record`, starting with a
//! `begin` command and ending with an `end` command. The
//! buffer is then submitted through a queue.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const CommandPool = @import("CommandPool.zig");
const CommandBuffer = @This();
const std = @import("std");

/// Pool and level of a command buffer.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Pool that owns the buffer.
    pool: CommandPool,
    /// Level that controls where the buffer is used.
    level: Level = .primary,
};

/// Where a command buffer may be executed.
pub const Level = enum {
    /// Submitted directly to a queue.
    primary,
    /// Recorded inside a primary buffer with
    /// `Device.Command.execute_commands`.
    secondary,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a command buffer and attributes
/// diagnostics to `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the buffer.
/// - `desc`: pool and level.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live buffer; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!CommandBuffer {
    const rc_desc: Resource.Desc = .{ .command_buffer = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a command buffer on `device`.
///
/// # Args
/// - `device`: the device that owns the buffer.
/// - `desc`: pool and level.
///
/// # Return
/// A live buffer; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the pool level or queue
///   family is rejected.
/// - `error.InvalidResource`: `pool` is not live or
///   has the wrong kind.
/// - `error.TooManyObjects`: the backend cannot track
///   another command buffer.
/// - `error.Unsupport`: the backend cannot create
///   command buffers.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(
    device: Device,
    desc: Desc,
) Device.CreateError!CommandBuffer {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the buffer and attributes diagnostics to
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
/// `command_buffer`.
pub fn destroyWithAddr(
    self: CommandBuffer,
    ret_addr: usize,
) void {
    std.debug.assert(self.handle.kind == .command_buffer);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the buffer.
///
/// # Note
/// The buffer must not be pending execution. The
/// handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_buffer`.
pub fn destroy(self: CommandBuffer) void {
    self.destroyWithAddr(@returnAddress());
}

/// Records one command and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `cmd`: command to append; borrowed only for
///   this call.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// The error contract matches `record`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_buffer`.
pub fn recordWithAddr(
    self: CommandBuffer,
    cmd: *const Device.Command,
    ret_addr: usize,
) Device.RecordError!void {
    std.debug.assert(self.handle.kind == .command_buffer);
    try self.device.rawRecord(self.handle, cmd, ret_addr);
}

/// Records one command into the buffer.
///
/// # Args
/// - `cmd`: command to append; borrowed only for
///   this call. Resources it names must stay alive
///   until the recorded work completes.
///
/// # Error
/// - `error.InvalidDesc`: the command payload is
///   malformed.
/// - `error.InvalidResource`: a referenced resource is
///   not live or has the wrong kind.
/// - `error.CommandBufferState`: the buffer is not
///   recording.
/// - `error.Unsupport`: the device cannot execute the
///   command.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Note
/// Recording starts with `Device.Command.begin` and
/// stops with `Device.Command.end`.
/// `error.CommandBufferState` reports a command
/// recorded outside that window.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_buffer`.
pub fn record(
    self: CommandBuffer,
    cmd: *const Device.Command,
) Device.RecordError!void {
    try self.recordWithAddr(cmd, @returnAddress());
}

/// Resets the buffer and attributes diagnostics to
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
/// `command_buffer`.
pub fn resetWithAddr(
    self: CommandBuffer,
    ret_addr: usize,
) Device.DeviceOpError!void {
    std.debug.assert(self.handle.kind == .command_buffer);
    try self.device.rawDeviceOp(
        .{ .reset_command_buffer = self },
        null,
        ret_addr,
    );
}

/// Resets the buffer to its initial, empty state.
///
/// # Error
/// - `error.InvalidResource`: the handle is not live.
/// - `error.InvalidState`: the buffer is still in use
///   or its pool forbids individual resets.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `command_buffer`.
pub fn reset(self: CommandBuffer) Device.DeviceOpError!void {
    try self.resetWithAddr(@returnAddress());
}
