//! A compute pipeline.
//!
//! A compute pipeline binds one shader and one pipeline
//! layout, and executes through `dispatch` commands.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const PipelineLayout = @import("PipelineLayout.zig");
const Shader = @import("Shader.zig");
const PipelineCompute = @This();
const std = @import("std");

/// Shader and layout of a compute pipeline.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Bindings and push constants of the pipeline.
    layout: PipelineLayout,
    /// Compute shader executed by the pipeline.
    shader: Shader,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a compute pipeline on `device`.
///
/// # Args
/// - `device`: the device that owns the pipeline.
/// - `desc`: shader and pipeline layout.
///
/// # Return
/// A live pipeline; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the shader entry point or
///   layout is rejected.
/// - `error.InvalidResource`: a referenced object is
///   not live or has the wrong kind.
/// - `error.TooManyObjects`: the backend cannot track
///   another pipeline.
/// - `error.Unsupport`: the backend cannot create
///   compute pipelines.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(
    device: Device,
    desc: Desc,
) Device.CreateError!PipelineCompute {
    const rc_desc: Resource.Desc = .{ .pipeline_compute = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the pipeline.
///
/// # Note
/// Submitted work that uses the pipeline must have
/// completed. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `pipeline_compute`.
pub fn destroy(self: PipelineCompute) void {
    std.debug.assert(self.handle.kind == .pipeline_compute);
    self.device.rawDestroy(self.handle, @returnAddress());
}
