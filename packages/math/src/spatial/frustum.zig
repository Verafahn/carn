//! Viewing frusta made of six planes.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const matrix = @import("../linear/matrix.zig");
const plane = @import("plane.zig");
const sphere = @import("sphere.zig");
const aabb = @import("aabb.zig");
const obb = @import("obb.zig");

/// A viewing frustum made of six planes.
///
/// # Args
/// - `T`: the component type.
pub fn Frustum(comptime T: type) type {
    return struct {
        /// Left clipping plane.
        left: plane.Plane(T),
        /// Right clipping plane.
        right: plane.Plane(T),
        /// Bottom clipping plane.
        bottom: plane.Plane(T),
        /// Top clipping plane.
        top: plane.Plane(T),
        /// Near clipping plane.
        near: plane.Plane(T),
        /// Far clipping plane.
        far: plane.Plane(T),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Extracts a frustum from a
        /// view-projection matrix.
        ///
        /// # Args
        /// - `view_projection`: a row-vector matrix
        ///   that maps view space to clip space.
        ///
        /// # Return
        /// The six normalized clipping planes.
        ///
        /// # Note
        /// Extraction assumes clip depth in `0..1`.
        /// With reversed depth the `near` and `far`
        /// fields hold the opposite planes, which does
        /// not affect the tests.
        pub fn fromMatrix(
            view_projection: matrix.Mat4(T),
        ) Self {
            const c0 = view_projection.column(0);
            const c1 = view_projection.column(1);
            const c2 = view_projection.column(2);
            const c3 = view_projection.column(3);

            const left = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{
                    c3[0] + c0[0],
                    c3[1] + c0[1],
                    c3[2] + c0[2],
                }),
                .distance = c3[3] + c0[3],
            };
            const right = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{
                    c3[0] - c0[0],
                    c3[1] - c0[1],
                    c3[2] - c0[2],
                }),
                .distance = c3[3] - c0[3],
            };
            const bottom = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{
                    c3[0] + c1[0],
                    c3[1] + c1[1],
                    c3[2] + c1[2],
                }),
                .distance = c3[3] + c1[3],
            };
            const top = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{
                    c3[0] - c1[0],
                    c3[1] - c1[1],
                    c3[2] - c1[2],
                }),
                .distance = c3[3] - c1[3],
            };
            const near = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{ c2[0], c2[1], c2[2] }),
                .distance = c2[3],
            };
            const far = plane.Plane(T){
                .normal = vector.Vector(T, 3)
                    .fromElements(.{
                    c3[0] - c2[0],
                    c3[1] - c2[1],
                    c3[2] - c2[2],
                }),
                .distance = c3[3] - c2[3],
            };

            return .{
                .left = left.normalized(),
                .right = right.normalized(),
                .bottom = bottom.normalized(),
                .top = top.normalized(),
                .near = near.normalized(),
                .far = far.normalized(),
            };
        }

        /// Tests whether a point is inside.
        ///
        /// # Args
        /// - `point`: the point to test.
        ///
        /// # Return
        /// `true` when the point is inside all planes.
        pub fn containsPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) bool {
            inline for (self.planes()) |current| {
                if (current.signedDistance(point) < 0) {
                    return false;
                }
            }
            return true;
        }

        /// Tests whether a sphere overlaps.
        ///
        /// # Args
        /// - `other`: the sphere to test.
        ///
        /// # Return
        /// `false` only when the sphere is fully
        /// outside one plane.
        ///
        /// # Note
        /// A sphere outside the corners may still
        /// report `true`; the test is conservative.
        pub fn intersectsSphere(
            self: Self,
            other: sphere.Sphere(T),
        ) bool {
            inline for (self.planes()) |current| {
                if (current.signedDistance(other.center) <
                    -other.radius)
                {
                    return false;
                }
            }
            return true;
        }

        /// Tests whether a box overlaps.
        ///
        /// # Args
        /// - `box`: the aligned box to test.
        ///
        /// # Return
        /// `false` only when the box is fully outside
        /// one plane.
        ///
        /// # Note
        /// The test is conservative; it may accept
        /// boxes that only touch the corners.
        pub fn intersectsAabb(
            self: Self,
            box: aabb.Aabb(T),
        ) bool {
            inline for (self.planes()) |current| {
                const positive = vector.Vector(T, 3)
                    .fromElements(@select(
                    T,
                    current.normal.elements >=
                        @as(@Vector(3, T), @splat(0)),
                    box.max.elements,
                    box.min.elements,
                ));
                if (current.signedDistance(positive) < 0) {
                    return false;
                }
            }
            return true;
        }

        /// Tests whether an oriented box overlaps.
        ///
        /// # Args
        /// - `box`: the oriented box to test.
        ///
        /// # Return
        /// `false` only when the box is fully outside
        /// one plane.
        pub fn intersectsObb(
            self: Self,
            box: obb.Obb(T),
        ) bool {
            const transposed = box.rotation.transpose();
            inline for (self.planes()) |current| {
                const local = current.normal.mulMatrix(
                    3,
                    transposed,
                );
                const radius = box.half_extents.dot(
                    vector.Vector(T, 3)
                        .fromElements(@abs(local.elements)),
                );
                if (current.signedDistance(box.center) <
                    -radius)
                {
                    return false;
                }
            }
            return true;
        }

        fn planes(self: Self) [6]plane.Plane(T) {
            return .{
                self.left,
                self.right,
                self.bottom,
                self.top,
                self.near,
                self.far,
            };
        }
    };
}

test "frustum from identity" {
    const Vec3f = vector.Vec3(f32);
    const projection = matrix.Mat4(f32).identity();
    const frustum = Frustum(f32).fromMatrix(projection);
    try std.testing.expect(frustum.containsPoint(Vec3f.zero()));
    try std.testing.expect(!frustum.containsPoint(
        Vec3f.fromElements(.{ 2, 0, 0 }),
    ));
    try std.testing.expect(frustum.intersectsSphere(
        sphere.Sphere(f32){ .center = Vec3f.zero(), .radius = 0.5 },
    ));
    try std.testing.expect(!frustum.intersectsSphere(
        sphere.Sphere(f32){
            .center = Vec3f.fromElements(.{ 5, 0, 0 }),
            .radius = 0.5,
        },
    ));
    try std.testing.expect(frustum.intersectsAabb(
        aabb.Aabb(f32){
            .min = Vec3f.splat(-0.5),
            .max = Vec3f.splat(0.5),
        },
    ));
    try std.testing.expect(!frustum.intersectsAabb(
        aabb.Aabb(f32){
            .min = Vec3f.splat(2),
            .max = Vec3f.splat(3),
        },
    ));
    try std.testing.expect(frustum.intersectsObb(
        obb.Obb(f32).fromAabb(aabb.Aabb(f32){
            .min = Vec3f.splat(-0.5),
            .max = Vec3f.splat(0.5),
        }),
    ));
    try std.testing.expect(!frustum.intersectsObb(
        obb.Obb(f32).fromAabb(aabb.Aabb(f32){
            .min = Vec3f.splat(2),
            .max = Vec3f.splat(3),
        }),
    ));
}

test "frustum from perspective" {
    const Vec3f = vector.Vec3(f32);
    const projection = matrix.Mat4(f32).fromColumns(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 2 },
        .{ 0, 0, -1, 0 },
    });
    const frustum = Frustum(f32).fromMatrix(projection);
    try std.testing.expect(frustum.containsPoint(
        Vec3f.fromElements(.{ 0, 0, -1.5 }),
    ));
    try std.testing.expect(!frustum.containsPoint(
        Vec3f.fromElements(.{ 0, 0, -0.5 }),
    ));
    try std.testing.expect(!frustum.containsPoint(
        Vec3f.fromElements(.{ 0, 0, -3 }),
    ));
    try std.testing.expect(!frustum.containsPoint(
        Vec3f.fromElements(.{ 3, 0, -2 }),
    ));
    try std.testing.expect(frustum.containsPoint(
        Vec3f.fromElements(.{ 1.9, 0, -2 }),
    ));
    try std.testing.expect(frustum.intersectsAabb(
        aabb.Aabb(f32){
            .min = Vec3f.fromElements(.{ -0.5, -0.5, -2.5 }),
            .max = Vec3f.fromElements(.{ 0.5, 0.5, -1.5 }),
        },
    ));
    try std.testing.expect(!frustum.intersectsAabb(
        aabb.Aabb(f32){
            .min = Vec3f.fromElements(.{ -0.5, -0.5, -0.5 }),
            .max = Vec3f.zero(),
        },
    ));
}
