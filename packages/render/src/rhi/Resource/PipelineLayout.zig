//! Bindings shared by pipelines.
//!
//! A pipeline layout fixes the descriptor set layouts
//! and push constant windows that a pipeline uses.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const DescriptorSetLayout = @import("DescriptorSetLayout.zig");
const Shader = @import("Shader.zig");
const PipelineLayout = @This();
const std = @import("std");

/// Descriptor sets and push constants of a layout.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Descriptor set layouts at indices 0..N.
    sets: []const DescriptorSetLayout = &.{},
    /// Push constant windows, in declaration order.
    push_constants: []const PushConstantRange = &.{},
};

/// One push constant window.
pub const PushConstantRange = struct {
    /// Stages that may read the window.
    stages: Shader.StageFlags,
    /// Byte offset of the window.
    offset: u32 = 0,
    /// Window length in bytes.
    size: u32,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a pipeline layout on `device`.
///
/// # Args
/// - `device`: the device that owns the layout.
/// - `desc`: set layouts and push constant windows.
///
/// # Return
/// A live layout; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: a set layout is invalid or
///   two push constant windows overlap.
/// - `error.InvalidResource`: a set layout is not
///   live or has the wrong kind.
/// - `error.TooManyObjects`: the backend cannot track
///   another layout.
/// - `error.Unsupport`: the backend cannot create
///   pipeline layouts.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!PipelineLayout {
    const rc_desc: Resource.Desc = .{ .pipeline_layout = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the layout.
///
/// # Note
/// Pipelines created from the layout must not be in
/// use. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `pipeline_layout`.
pub fn destroy(self: PipelineLayout) void {
    std.debug.assert(self.handle.kind == .pipeline_layout);
    self.device.rawDestroy(self.handle, @returnAddress());
}
