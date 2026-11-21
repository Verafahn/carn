const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Sampler = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!Sampler {
    const rc_desc: Resource.Desc = .{ .sampler = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: Sampler) void {
    std.debug.assert(self.handle.kind == .sampler);
    self.device.rawDestroy(self.handle, @returnAddress());
}
