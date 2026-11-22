//! A presentation surface.
//!
//! A surface wraps a native window or layer. Swapchains
//! present into a surface; the surface itself owns no
//! images.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Surface = @This();
const std = @import("std");

/// Native window and platform of a surface.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Raw native handle; ignored by `headless`.
    window: usize,
    /// Platform API that owns `window`.
    platform: Platform,
};

/// Platform APIs accepted by `Surface.Desc`.
pub const Platform = enum {
    /// A Win32 `HWND`.
    win32,
    /// An Xlib `Window`.
    xlib,
    /// An XCB `xcb_window_t`.
    xcb,
    /// A Wayland `wl_surface`.
    wayland,
    /// A Metal `CAMetalLayer`.
    metal,
    /// An Android `ANativeWindow`.
    android,
    /// No window; presentation is unsupported.
    headless,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a surface on `device`.
///
/// # Args
/// - `device`: the device that owns the surface.
/// - `desc`: native window and platform.
///
/// # Return
/// A live surface; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: `window` is zero on a
///   non-headless platform, or it does not belong to
///   `platform`.
/// - `error.TooManyObjects`: the backend cannot track
///   another surface.
/// - `error.Unsupport`: the backend cannot create
///   surfaces for `platform`.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Surface {
    const rc_desc: Resource.Desc = .{ .surface = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Destroys the surface.
///
/// # Note
/// Swapchains created for the surface must be destroyed
/// first. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `surface`.
pub fn destroy(self: Surface) void {
    std.debug.assert(self.handle.kind == .surface);
    self.device.rawDestroy(self.handle, @returnAddress());
}
