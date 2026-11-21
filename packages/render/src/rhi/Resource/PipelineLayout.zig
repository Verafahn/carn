const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const PipelineLayout = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!PipelineLayout {
    const rc_desc: Resource.Desc = .{ .pipeline_layout = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: PipelineLayout) void {
    std.debug.assert(self.handle.kind == .pipeline_layout);
    self.device.rawDestroy(self.handle, @returnAddress());
}
