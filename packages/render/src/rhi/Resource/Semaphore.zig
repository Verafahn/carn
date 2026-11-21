const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Semaphore = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Semaphore {
    const rc_desc: Resource.Desc = .{ .semaphore = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Semaphore) void {
    std.debug.assert(self.handle.kind == .semaphore);
    self.device.rawDestroy(self.handle, @returnAddress());
}
