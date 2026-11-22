//! A linear GPU buffer.
//!
//! Buffers are typed handles. Create one with `create`,
//! release it with `destroy`, and reach host-visible
//! memory with `map` and `unmap`.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Buffer = @This();
const std = @import("std");

/// Size, usage and memory domain of a buffer.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Size in bytes; must be greater than zero.
    size: u64,
    /// Operations the buffer may be used for.
    usage: Usage = .{},
    /// Memory domain that backs the buffer.
    memory: Memory = .device,
};

/// The set of operations a buffer may be used for.
pub const Usage = packed struct {
    /// Source of vertex data.
    vertex: bool = false,
    /// Source of index data.
    index: bool = false,
    /// Uniform buffer binding.
    uniform: bool = false,
    /// Storage buffer binding.
    storage: bool = false,
    /// Source of indirect draw or dispatch arguments.
    indirect: bool = false,
    /// Source of transfer operations.
    transfer_src: bool = false,
    /// Destination of transfer operations.
    transfer_dst: bool = false,
};

/// The memory domain that backs a buffer.
pub const Memory = enum {
    /// Device-local memory.
    ///
    /// Fast for the GPU and not mappable by the host.
    device,
    /// Host-visible memory.
    ///
    /// Mappable with `map`, but usually not device-local.
    host,
    /// Device-local and host-visible memory.
    unified,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a buffer on `device`.
///
/// # Args
/// - `device`: the device that owns the buffer.
/// - `desc`: size, usage and memory domain.
///
/// # Return
/// A live buffer; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: `size` is zero or `usage` is
///   empty.
/// - `error.OutOfDeviceMemory`: device memory is
///   exhausted.
/// - `error.TooManyObjects`: the backend cannot track
///   another buffer.
/// - `error.Unsupport`: the backend cannot honour
///   `usage` or `memory`.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Buffer {
    const rc_desc: Resource.Desc = .{ .buffer = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the buffer and releases its device memory.
///
/// # Note
/// Submitted work that reads or writes the buffer must
/// have completed before this call. The handle must not
/// be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `buffer`.
pub fn destroy(self: Buffer) void {
    std.debug.assert(self.handle.kind == .buffer);
    self.device.rawDestroy(self.handle, @returnAddress());
}

/// Maps the buffer into host address space.
///
/// # Args
/// - `offset`: byte offset of the mapped range.
/// - `size`: byte length of the mapped range.
/// - `flags`: requested host access.
///
/// # Return
/// Host memory for `offset...offset + size`; it stays
/// valid until `unmap`.
///
/// # Error
/// - `error.InvalidDesc`: the range or flags are
///   malformed.
/// - `error.InvalidResource`: the handle is not live.
/// - `error.NotHostVisible`: `Buffer.Desc.memory` is
///   not `host` or `unified`.
/// - `error.AlreadyMapped`: the buffer is mapped.
/// - `error.OutOfRange`: the range leaves the buffer.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Note
/// The returned slice is owned by the backend and must
/// not be freed. Write it only when `flags.write` is
/// set.
///
/// # Panic
/// In safety builds, the handle kind is not `buffer`.
pub fn map(
    self: Buffer,
    offset: u64,
    size: u64,
    flags: Device.MapFlags,
) Device.MapError![]u8 {
    std.debug.assert(self.handle.kind == .buffer);
    return try self.device.rawMap(
        self.handle,
        offset,
        size,
        flags,
        @returnAddress(),
    );
}

/// Releases a mapping created by `map`.
///
/// # Args
/// - `dirty`: ranges written by the host, relative to
///   the start of the buffer. May be empty.
///
/// # Note
/// The pointer returned by `map` is invalid after this
/// call. Coherent memory ignores `dirty`.
///
/// # Panic
/// In safety builds, the handle kind is not `buffer`.
pub fn unmap(self: Buffer, dirty: []const Device.DirtyRange) void {
    std.debug.assert(self.handle.kind == .buffer);
    self.device.rawUnmap(self.handle, dirty, @returnAddress());
}
