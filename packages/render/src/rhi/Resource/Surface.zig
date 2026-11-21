const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Surface = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Surface {
    const rc_desc: Resource.Desc = .{ .surface = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Surface) void {
    std.debug.assert(self.handle.kind == .surface);
    self.device.rawDestroy(self.handle, @returnAddress());
}
