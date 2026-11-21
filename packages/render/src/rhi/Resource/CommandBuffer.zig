const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const CommandBuffer = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!CommandBuffer {
    const rc_desc: Resource.Desc = .{ .command_buffer = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: CommandBuffer) void {
    std.debug.assert(self.handle.kind == .command_buffer);
    self.device.rawDestroy(self.handle, @returnAddress());
}
