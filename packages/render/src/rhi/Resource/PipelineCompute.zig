const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const PipelineCompute = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!PipelineCompute {
    const rc_desc: Resource.Desc = .{ .pipeline_compute = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: PipelineCompute) void {
    std.debug.assert(self.handle.kind == .pipeline_compute);
    self.device.rawDestroy(self.handle, @returnAddress());
}
