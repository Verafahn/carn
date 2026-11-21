const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const PipelineGraphics = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!PipelineGraphics {
    const rc_desc: Resource.Desc = .{ .pipeline_graphics = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: PipelineGraphics) void {
    std.debug.assert(self.handle.kind == .pipeline_graphics);
    self.device.rawDestroy(self.handle, @returnAddress());
}
