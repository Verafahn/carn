//! A chain of presentable images.
//!
//! A swapchain owns the images that are presented to a
//! surface. Acquire an image with `acquire`, render into
//! it, then hand it to `Queue.present`.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Surface = @import("Surface.zig");
const Texture = @import("Texture.zig");
const Semaphore = @import("Semaphore.zig");
const Fence = @import("Fence.zig");
const Swapchain = @This();
const std = @import("std");

/// Surface, format and presentation policy.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Surface the images are presented to.
    surface: Surface,
    /// Format of every image.
    format: Resource.Format,
    /// Size of every image in pixels.
    ///
    /// `extent.depth` must be 1.
    extent: Resource.Extent,
    /// How images are handed to the display.
    mode: PresentMode = .fifo,
    /// Requested image count; 0 lets the backend pick.
    images: u32 = 0,
    /// Usages of every image.
    usage: Texture.Usage = .{ .color_attachment = true },
};

/// How a presented image reaches the display.
pub const PresentMode = enum {
    /// Vertical sync; blocks while the queue is full.
    fifo,
    /// Vertical sync, but late frames present at once.
    fifo_relaxed,
    /// Vertical sync with a replaceable pending frame.
    mailbox,
    /// Present without waiting for vertical sync.
    immediate,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a swapchain and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the swapchain.
/// - `desc`: surface, format and presentation
///   policy.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live swapchain; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!Swapchain {
    const rc_desc: Resource.Desc = .{ .swapchain = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a swapchain on `device`.
///
/// # Args
/// - `device`: the device that owns the swapchain.
/// - `desc`: surface, format and presentation policy.
///
/// # Return
/// A live swapchain; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: the format, extent or mode
///   is not supported by the surface.
/// - `error.InvalidResource`: `surface` is not live
///   or has the wrong kind.
/// - `error.OutOfDeviceMemory`: image memory is
///   exhausted.
/// - `error.TooManyObjects`: the backend cannot track
///   another swapchain.
/// - `error.Unsupport`: the backend cannot present.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Swapchain {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the swapchain and attributes diagnostics
/// to `ret_addr`.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// The lifetime contract matches `destroy`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn destroyWithAddr(self: Swapchain, ret_addr: usize) void {
    std.debug.assert(self.handle.kind == .swapchain);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the swapchain and its images.
///
/// # Note
/// Every acquired image must have been presented, and
/// pending work must have completed. The handle must
/// not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn destroy(self: Swapchain) void {
    self.destroyWithAddr(@returnAddress());
}

/// Acquires the next image and attributes
/// diagnostics to `ret_addr`.
///
/// # Args
/// - `signal`: semaphore signalled when the image is
///   available; null signals nothing.
/// - `fence`: fence signalled when the image is
///   available; null signals nothing.
/// - `timeout_ns`: nanosecond timeout; null waits
///   forever.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The index of the acquired image.
///
/// # Error
/// The error contract matches `acquire`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn acquireWithAddr(
    self: Swapchain,
    signal: ?Semaphore,
    fence: ?Fence,
    timeout_ns: ?u64,
    ret_addr: usize,
) Device.DeviceOpError!u32 {
    std.debug.assert(self.handle.kind == .swapchain);
    var index: u32 = undefined;
    try self.device.rawDeviceOp(
        .{ .acquire = .{
            .swapchain = self,
            .signal = signal,
            .fence = fence,
            .timeout_ns = timeout_ns,
        } },
        std.mem.asBytes(&index),
        ret_addr,
    );
    return index;
}

/// Acquires the next image for rendering.
///
/// # Args
/// - `signal`: semaphore signalled when the image is
///   available; null signals nothing.
/// - `fence`: fence signalled when the image is
///   available; null signals nothing.
/// - `timeout_ns`: nanosecond timeout; null waits
///   forever.
///
/// # Return
/// The index of the acquired image, usable as the
/// `image_index` argument of `Queue.present`.
///
/// # Error
/// - `error.InvalidResource`: the handle is not live.
/// - `error.SwapchainOutOfDate`: the swapchain no
///   longer matches the surface.
/// - `error.SurfaceLost`: the surface is gone.
/// - `error.Timeout`: `timeout_ns` elapsed first.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Note
/// The application owns the image until it is
/// presented, which returns it to the presentation
/// engine. Acquiring every image without presenting
/// blocks.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn acquire(
    self: Swapchain,
    signal: ?Semaphore,
    fence: ?Fence,
    timeout_ns: ?u64,
) Device.DeviceOpError!u32 {
    return self.acquireWithAddr(
        signal,
        fence,
        timeout_ns,
        @returnAddress(),
    );
}

/// Returns image `index` and attributes diagnostics
/// to `ret_addr`.
///
/// # Args
/// - `index`: image index in `0..image_count`.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A texture owned by the swapchain.
///
/// # Error
/// The error contract matches `image`.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn imageWithAddr(
    self: Swapchain,
    index: u32,
    ret_addr: usize,
) Device.DeviceOpError!Texture {
    std.debug.assert(self.handle.kind == .swapchain);
    var handle: Resource = undefined;
    try self.device.rawDeviceOp(
        .{ .swapchain_image = .{
            .swapchain = self,
            .index = index,
        } },
        std.mem.asBytes(&handle),
        ret_addr,
    );
    return .{
        .handle = handle,
        .device = self.device,
    };
}

/// Returns the texture that wraps image `index`.
///
/// # Args
/// - `index`: image index in `0..image_count`.
///
/// # Return
/// A texture owned by the swapchain. It stays valid
/// until the swapchain is destroyed.
///
/// # Error
/// - `error.InvalidResource`: the handle is not live.
/// - `error.IndexOutOfBounds`: `index` is not an image
///   of the swapchain.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
///
/// # Note
/// Do not destroy the returned texture. The image must
/// be acquired before it is rendered into.
///
/// # Panic
/// In safety builds, the handle kind is not
/// `swapchain`.
pub fn image(
    self: Swapchain,
    index: u32,
) Device.DeviceOpError!Texture {
    return self.imageWithAddr(index, @returnAddress());
}
