const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Shader = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Shader {
    const rc_desc: Resource.Desc = .{ .shader = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Shader) void {
    std.debug.assert(self.handle.kind == .shader);
    self.device.rawDestroy(self.handle, @returnAddress());
}
