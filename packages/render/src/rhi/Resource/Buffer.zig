const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Buffer = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Buffer {
    const rc_desc: Resource.Desc = .{ .buffer = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Buffer) void {
    std.debug.assert(self.handle.kind == .buffer);
    self.device.rawDestroy(self.handle, @returnAddress());
}
