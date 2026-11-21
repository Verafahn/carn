const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Texture = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Texture {
    const rc_desc: Resource.Desc = .{ .texture = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Texture) void {
    std.debug.assert(self.handle.kind == .texture);
    self.device.rawDestroy(self.handle, @returnAddress());
}
