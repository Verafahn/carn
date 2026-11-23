//! Spatial primitives, queries, cameras, and
//! coordinate-space transforms.

pub const plane = @import("plane.zig");
pub const sphere = @import("sphere.zig");
pub const aabb = @import("aabb.zig");
pub const obb = @import("obb.zig");
pub const frustum = @import("frustum.zig");
pub const capsule = @import("capsule.zig");
pub const segment = @import("segment.zig");
pub const triangle = @import("triangle.zig");
pub const ray = @import("ray.zig");
pub const camera = @import("camera.zig");
pub const transform = @import("transform.zig");

test {
    _ = plane;
    _ = sphere;
    _ = aabb;
    _ = obb;
    _ = frustum;
    _ = capsule;
    _ = segment;
    _ = triangle;
    _ = ray;
    _ = camera;
    _ = transform;
}
