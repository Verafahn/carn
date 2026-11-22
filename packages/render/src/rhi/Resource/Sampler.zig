//! State used when sampling a texture.
//!
//! Samplers are typed handles. Create one with `create`
//! and release it with `destroy`.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Sampler = @This();
const std = @import("std");

/// Filtering and addressing state of a sampler.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Filter used when magnifying a texture.
    mag_filter: Filter = .linear,
    /// Filter used when minifying a texture.
    min_filter: Filter = .linear,
    /// Filter used between mip levels.
    mip_filter: Filter = .linear,
    /// Addressing mode per axis.
    address: Address = .{},
    /// Lowest mip level that may be sampled.
    lod_min: f32 = 0,
    /// Highest mip level that may be sampled.
    lod_max: f32 = std.math.inf(f32),
    /// Comparison for shadow samplers; null samples
    /// values directly.
    compare: ?Resource.CompareOp = null,
    /// Maximum anisotropy; 1 disables it.
    max_anisotropy: f32 = 1,
};

/// Filtering applied to sampled texels.
pub const Filter = enum {
    /// Select the nearest texel.
    nearest,
    /// Blend the nearest texels.
    linear,
};

/// Addressing mode of one texture axis.
pub const Wrap = enum {
    /// Repeat the texture.
    repeat,
    /// Repeat the texture with mirrored copies.
    mirror_repeat,
    /// Clamp to the edge texel.
    clamp_to_edge,
    /// Clamp to the border color.
    clamp_to_border,
    /// Mirror once, then clamp.
    mirror_clamp_to_edge,
};

/// Addressing mode per texture axis.
pub const Address = struct {
    /// Mode along the x axis.
    u: Wrap = .repeat,
    /// Mode along the y axis.
    v: Wrap = .repeat,
    /// Mode along the z axis.
    w: Wrap = .repeat,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a sampler and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the sampler.
/// - `desc`: filtering, addressing and comparison
///   state.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live sampler; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!Sampler {
    const rc_desc: Resource.Desc = .{ .sampler = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a sampler on `device`.
///
/// # Args
/// - `device`: the device that owns the sampler.
/// - `desc`: filtering, addressing and comparison
///   state.
///
/// # Return
/// A live sampler; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the lod range or anisotropy
///   is rejected.
/// - `error.TooManyObjects`: the backend cannot track
///   another sampler.
/// - `error.Unsupport`: the backend cannot create
///   samplers.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Sampler {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the sampler and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// The lifetime contract matches `destroy`.
///
/// # Panic
/// In safety builds, the handle kind is not `sampler`.
pub fn destroyWithAddr(self: Sampler, ret_addr: usize) void {
    std.debug.assert(self.handle.kind == .sampler);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the sampler.
///
/// # Note
/// Descriptor sets that reference the sampler must not
/// be in use. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `sampler`.
pub fn destroy(self: Sampler) void {
    self.destroyWithAddr(@returnAddress());
}
