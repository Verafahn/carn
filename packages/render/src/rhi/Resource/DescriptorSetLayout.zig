const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const DescriptorSetLayout = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!DescriptorSetLayout {
    const rc_desc: Resource.Desc = .{ .descriptor_set_layout = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: DescriptorSetLayout) void {
    std.debug.assert(self.handle.kind == .descriptor_set_layout);
    self.device.rawDestroy(self.handle, @returnAddress());
}
