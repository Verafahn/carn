const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Fence = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Fence {
    const rc_desc: Resource.Desc = .{ .fence = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Fence) void {
    std.debug.assert(self.handle.kind == .fence);
    self.device.rawDestroy(self.handle, @returnAddress());
}
