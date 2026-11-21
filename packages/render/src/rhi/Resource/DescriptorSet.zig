const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const DescriptorSet = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!DescriptorSet {
    const rc_desc: Resource.Desc = .{ .descriptor_set = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: DescriptorSet) void {
    std.debug.assert(self.handle.kind == .descriptor_set);
    self.device.rawDestroy(self.handle, @returnAddress());
}
