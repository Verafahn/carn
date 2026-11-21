const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Swapchain = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Swapchain {
    const rc_desc: Resource.Desc = .{ .swapchain = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Swapchain) void {
    std.debug.assert(self.handle.kind == .swapchain);
    self.device.rawDestroy(self.handle, @returnAddress());
}
