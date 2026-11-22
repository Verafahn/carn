//! The shape of a descriptor set.
//!
//! A layout lists the bindings a set exposes and the
//! shader stages that may access each binding.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Shader = @import("Shader.zig");
const DescriptorSetLayout = @This();
const std = @import("std");

/// Binding declarations of a descriptor set layout.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Bindings exposed by the set.
    bindings: []const Binding = &.{},
};

/// One binding declaration.
pub const Binding = struct {
    /// Binding index used by shaders.
    binding: u32,
    /// Kind of descriptor stored at the binding.
    kind: Kind,
    /// Number of array elements; 1 for a single
    /// descriptor.
    count: u32 = 1,
    /// Stages that may access the binding.
    stages: Shader.StageFlags,
    /// Restricts storage descriptors to reads.
    ///
    /// Ignored by non-storage kinds.
    read_only: bool = false,
};

/// Kinds of descriptor a binding may expose.
pub const Kind = enum {
    /// A uniform buffer.
    uniform_buffer,
    /// A storage buffer.
    storage_buffer,
    /// A sampled texture.
    sampled_texture,
    /// A storage texture.
    storage_texture,
    /// A standalone sampler.
    sampler,
    /// A texture and sampler pair.
    combined_sampler,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a descriptor set layout on `device`.
///
/// # Args
/// - `device`: the device that owns the layout.
/// - `desc`: binding declarations.
///
/// # Return
/// A live layout; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: a binding index is repeated
///   or a declaration is rejected.
/// - `error.TooManyObjects`: the backend cannot track
///   another layout.
/// - `error.Unsupport`: the backend cannot create
///   this kind of binding.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(
    device: Device,
    desc: Desc,
) Device.CreateError!DescriptorSetLayout {
    const rc_desc: Resource.Desc = .{
        .descriptor_set_layout = desc,
    };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the layout.
///
/// # Note
/// Descriptor sets created from the layout must not be
/// in use. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `descriptor_set_layout`.
pub fn destroy(self: DescriptorSetLayout) void {
    std.debug.assert(self.handle.kind == .descriptor_set_layout);
    self.device.rawDestroy(self.handle, @returnAddress());
}
