//! A GPU image.
//!
//! Textures are typed handles. Create one with `create`
//! and release it with `destroy`; contents are written
//! with recorded copy or render commands.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Texture = @This();
const std = @import("std");

/// Shape, format and usage of a texture.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Dimensionality of the texture.
    kind: Kind = .d2,
    /// Size of mip level zero.
    extent: Resource.Extent,
    /// Format of every texel.
    format: Resource.Format,
    /// Mip levels; 1 disables mipmapping.
    mips: u16 = 1,
    /// Array layers; 1 is a plain texture.
    layers: u16 = 1,
    /// Samples per texel.
    samples: SampleCount = .x1,
    /// Operations the texture may be used for.
    usage: Usage = .{},
};

/// Dimensionality of a texture.
pub const Kind = enum {
    /// A 1D texture.
    d1,
    /// A 2D texture or 2D array when `layers` is set.
    d2,
    /// A 3D texture.
    d3,
};

/// Samples per texel.
pub const SampleCount = enum(u8) {
    /// One sample per texel.
    x1 = 1,
    /// Two samples per texel.
    x2 = 2,
    /// Four samples per texel.
    x4 = 4,
    /// Eight samples per texel.
    x8 = 8,
    /// Sixteen samples per texel.
    x16 = 16,
};

/// The set of operations a texture may be used for.
pub const Usage = packed struct {
    /// Read by a shader sampler.
    sampled: bool = false,
    /// Read or written as a storage image.
    storage: bool = false,
    /// Used as a color render target.
    color_attachment: bool = false,
    /// Used as a depth-stencil render target.
    depth_stencil_attachment: bool = false,
    /// Source of transfer operations.
    transfer_src: bool = false,
    /// Destination of transfer operations.
    transfer_dst: bool = false,
};

/// Colour and depth-stencil planes of a format.
pub const Aspect = packed struct {
    /// The color plane.
    color: bool = false,
    /// The depth plane.
    depth: bool = false,
    /// The stencil plane.
    stencil: bool = false,
};

/// The layout of a texture subresource.
pub const Layout = enum {
    /// Contents have no defined layout.
    undefined,
    /// General-purpose layout.
    general,
    /// Layout used by a color attachment.
    color_attachment,
    /// Layout used by a depth-stencil attachment.
    depth_stencil_attachment,
    /// Layout used by shader reads.
    shader_read,
    /// Layout used by a transfer source.
    transfer_src,
    /// Layout used by a transfer destination.
    transfer_dst,
    /// Layout used by presentation.
    present,
};

/// A range of mip levels and array layers.
pub const SubresourceRange = struct {
    /// First mip level.
    mip: u16 = 0,
    /// Mip levels covered; 0 means the rest.
    mip_count: u16 = 0,
    /// First array layer.
    layer: u16 = 0,
    /// Array layers covered; 0 means the rest.
    layer_count: u16 = 0,
    /// Planes covered. Empty means every plane of the
    /// format.
    aspects: Aspect = .{},
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a texture on `device`.
///
/// # Args
/// - `device`: the device that owns the texture.
/// - `desc`: shape, format and usage.
///
/// # Return
/// A live texture; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the extent, format, usage
///   or sample count is rejected.
/// - `error.OutOfDeviceMemory`: device memory is
///   exhausted.
/// - `error.TooManyObjects`: the backend cannot track
///   another texture.
/// - `error.Unsupport`: the backend cannot create
///   textures.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Texture {
    const rc_desc: Resource.Desc = .{ .texture = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the texture and releases its device memory.
///
/// # Note
/// Submitted work that reads or writes the texture must
/// have completed before this call. The handle must not
/// be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `texture`.
pub fn destroy(self: Texture) void {
    std.debug.assert(self.handle.kind == .texture);
    self.device.rawDestroy(self.handle, @returnAddress());
}
