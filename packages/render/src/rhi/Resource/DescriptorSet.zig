//! A set of descriptors bound to a pipeline.
//!
//! A set is created from a `DescriptorSetLayout` and
//! filled with `update` before it is recorded into a
//! command buffer.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const DescriptorSetLayout = @import("DescriptorSetLayout.zig");
const Buffer = @import("Buffer.zig");
const Texture = @import("Texture.zig");
const Sampler = @import("Sampler.zig");
const DescriptorSet = @This();
const std = @import("std");

/// Layout of a descriptor set.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Shape of the set.
    layout: DescriptorSetLayout,
};

/// One descriptor write.
pub const Write = union(enum) {
    /// A buffer binding.
    buffer: BufferWrite,
    /// A sampled or storage texture binding.
    texture: TextureWrite,
    /// A sampler binding.
    sampler: SamplerWrite,
    /// A combined texture and sampler binding.
    combined: CombinedWrite,
};

/// A buffer written into a set.
pub const BufferWrite = struct {
    /// Binding index to write.
    binding: u32,
    /// Array element index; 0 for a single binding.
    index: u32 = 0,
    /// Buffer to bind.
    buffer: Buffer,
    /// Byte offset of the visible range.
    offset: u64 = 0,
    /// Byte length of the visible range; 0 means the
    /// rest of the buffer.
    size: u64 = 0,
};

/// A texture written into a set.
pub const TextureWrite = struct {
    /// Binding index to write.
    binding: u32,
    /// Array element index; 0 for a single binding.
    index: u32 = 0,
    /// Texture to bind.
    texture: Texture,
    /// Subresource visible to the shader.
    range: Texture.SubresourceRange = .{},
};

/// A sampler written into a set.
pub const SamplerWrite = struct {
    /// Binding index to write.
    binding: u32,
    /// Array element index; 0 for a single binding.
    index: u32 = 0,
    /// Sampler to bind.
    sampler: Sampler,
};

/// A texture and sampler written as one descriptor.
pub const CombinedWrite = struct {
    /// Binding index to write.
    binding: u32,
    /// Array element index; 0 for a single binding.
    index: u32 = 0,
    /// Texture to bind.
    texture: Texture,
    /// Sampler to bind.
    sampler: Sampler,
    /// Subresource visible to the shader.
    range: Texture.SubresourceRange = .{},
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a descriptor set and attributes
/// diagnostics to `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the set.
/// - `desc`: layout of the set.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live set; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!DescriptorSet {
    const rc_desc: Resource.Desc = .{ .descriptor_set = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a descriptor set on `device`.
///
/// # Args
/// - `device`: the device that owns the set.
/// - `desc`: layout of the set.
///
/// # Return
/// A live set; release it with `destroy`.
///
/// # Error
/// - `error.InvalidResource`: `layout` is not live or
///   has the wrong kind.
/// - `error.OutOfDeviceMemory`: descriptor memory is
///   exhausted.
/// - `error.TooManyObjects`: the backend cannot track
///   another set.
/// - `error.Unsupport`: the backend cannot create
///   descriptor sets.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(
    device: Device,
    desc: Desc,
) Device.CreateError!DescriptorSet {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the set and attributes diagnostics to
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
/// `descriptor_set`.
pub fn destroyWithAddr(
    self: DescriptorSet,
    ret_addr: usize,
) void {
    std.debug.assert(self.handle.kind == .descriptor_set);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the descriptor set.
///
/// # Note
/// The set must not be referenced by pending work. The
/// handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `descriptor_set`.
pub fn destroy(self: DescriptorSet) void {
    self.destroyWithAddr(@returnAddress());
}

/// Writes descriptors and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `writes`: writes applied in order.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// The error contract matches `update`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `descriptor_set`.
pub fn updateWithAddr(
    self: DescriptorSet,
    writes: []const Write,
    ret_addr: usize,
) Device.DeviceOpError!void {
    std.debug.assert(self.handle.kind == .descriptor_set);
    try self.device.rawDeviceOp(
        .{ .update_descriptor_set = .{
            .set = self,
            .writes = writes,
        } },
        null,
        ret_addr,
    );
}

/// Writes descriptors into the set.
///
/// # Args
/// - `writes`: writes applied in order; every binding
///   must match the layout.
///
/// # Error
/// - `error.InvalidDesc`: a binding index, array
///   index or write kind is rejected by the layout.
/// - `error.InvalidResource`: a written resource is not
///   live or has the wrong kind.
/// - `error.InvalidState`: the set is used by pending
///   work.
/// - `error.Unsupport`: the backend cannot update the
///   set.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Note
/// Written resources must stay alive until every
/// command that reads the set has completed.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `descriptor_set`.
pub fn update(
    self: DescriptorSet,
    writes: []const Write,
) Device.DeviceOpError!void {
    try self.updateWithAddr(writes, @returnAddress());
}
