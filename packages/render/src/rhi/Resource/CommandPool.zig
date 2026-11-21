const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const CommandPool = @This();
const std = @import("std");

pub const Desc = struct {};

handle: Resource,
device: Device,

pub fn create(device: Device, desc: Desc) Device.CreateError!CommandPool {
    const rc_desc: Resource.Desc = .{ .command_pool = desc };
    const handle = try device.rawCreate(&rc_desc, @returnAddress());
    return .{
        .handle = handle,
        .device = device,
    };
}

pub fn destroy(self: CommandPool) void {
    std.debug.assert(self.handle.kind == .command_pool);
    self.device.rawDestroy(self.handle, @returnAddress());
}
