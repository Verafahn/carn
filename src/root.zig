pub const utils = @import("utils/root.zig");

pub const render = struct {
    pub const rhi = @import("render/rhi");
    pub const nil = @import("render/null");
    pub const vulkan = @import("render/vulkan");
};

test {
    _ = utils;
    _ = render.rhi;
    _ = render.nil;
    _ = render.vulkan;
}
