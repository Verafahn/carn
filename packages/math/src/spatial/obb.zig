//! Oriented bounding boxes.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const matrix = @import("../linear/matrix.zig");
const aabb = @import("aabb.zig");

/// An oriented bounding box.
///
/// # Args
/// - `T`: the component type.
pub fn Obb(comptime T: type) type {
    return struct {
        /// Center of the box.
        center: vector.Vector(T, 3),
        /// Half size along each local axis.
        half_extents: vector.Vector(T, 3),
        /// Local-to-world rotation.
        rotation: matrix.Mat3(T),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds an oriented box from an aligned one.
        ///
        /// # Args
        /// - `box`: the source box.
        ///
        /// # Return
        /// An oriented box with no rotation.
        pub fn fromAabb(box: aabb.Aabb(T)) Self {
            return .{
                .center = box.center(),
                .half_extents = box.size().scale(0.5),
                .rotation = matrix.Mat3(T).identity(),
            };
        }

        /// Tests whether a point is inside.
        ///
        /// # Args
        /// - `point`: the point to test.
        ///
        /// # Return
        /// `true` when the point is inside or on the
        /// surface.
        pub fn containsPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) bool {
            const offset = point.sub(self.center);
            const local = offset.mulMatrix(
                3,
                self.rotation.transpose(),
            );
            return @reduce(.And, @abs(local.elements) <=
                self.half_extents.elements);
        }

        /// Returns one corner.
        ///
        /// # Args
        /// - `index`: corner index in `0..8`.
        ///
        /// # Return
        /// The world-space corner; bit 0 selects the X
        /// sign, bit 1 the Y sign, and bit 2 the Z sign.
        ///
        /// # Panic
        /// In safety-checked builds, `index` is 8 or
        /// larger.
        pub fn corner(
            self: Self,
            index: usize,
        ) vector.Vector(T, 3) {
            std.debug.assert(index < 8);
            const x: T = if (index & 1 == 0) -1 else 1;
            const y: T = if (index & 2 == 0) -1 else 1;
            const z: T = if (index & 4 == 0) -1 else 1;
            const local = vector.Vector(T, 3)
                .fromElements(.{
                x * self.half_extents.at(0),
                y * self.half_extents.at(1),
                z * self.half_extents.at(2),
            });
            return self.center.add(
                local.mulMatrix(3, self.rotation),
            );
        }

        /// Computes a conservative aligned box.
        ///
        /// # Return
        /// An axis-aligned box containing the rotated
        /// box.
        pub fn toAabb(self: Self) aabb.Aabb(T) {
            var extents = vector.Vector(T, 3).zero();
            inline for (0..3) |axis| {
                var sum: T = 0;
                inline for (0..3) |component| {
                    sum += self.half_extents.at(component) *
                        @abs(self.rotation.at(component, axis));
                }
                extents.set(axis, sum);
            }
            return .{
                .min = self.center.sub(extents),
                .max = self.center.add(extents),
            };
        }
    };
}

test "obb from aabb" {
    const Vec3f = vector.Vec3(f32);
    const box = aabb.Aabb(f32){
        .min = Vec3f.zero(),
        .max = Vec3f.fromElements(.{ 2, 4, 6 }),
    };
    const oriented = Obb(f32).fromAabb(box);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        oriented.center.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        oriented.half_extents.at(1),
        1e-6,
    );
    try std.testing.expect(oriented.containsPoint(
        Vec3f.fromElements(.{ 2, 4, 6 }),
    ));
    try std.testing.expect(!oriented.containsPoint(
        Vec3f.fromElements(.{ 3, 0, 0 }),
    ));

    const corner = oriented.corner(7);
    try std.testing.expectApproxEqAbs(@as(f32, 2), corner.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), corner.at(2), 1e-6);

    const aligned = oriented.toAabb();
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        aligned.min.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 6),
        aligned.max.at(2),
        1e-6,
    );
}

test "obb rotation" {
    const Vec3f = vector.Vec3(f32);
    const half_turn = 0.5 * @as(f32, 3.141592653589793);
    const rotation = matrix.Mat3(f32).fromColumns(.{
        .{ @cos(half_turn), @sin(half_turn), 0 },
        .{ -@sin(half_turn), @cos(half_turn), 0 },
        .{ 0, 0, 1 },
    });
    const oriented = Obb(f32){
        .center = Vec3f.zero(),
        .half_extents = Vec3f.fromElements(.{ 2, 1, 1 }),
        .rotation = rotation,
    };
    // The local X axis now points along world -X/Y.
    try std.testing.expect(oriented.containsPoint(
        Vec3f.fromElements(.{ 0, 2, 0.5 }),
    ));
    try std.testing.expect(!oriented.containsPoint(
        Vec3f.fromElements(.{ 2, 0, 0.5 }),
    ));
    const aligned = oriented.toAabb();
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        aligned.max.at(0),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        aligned.max.at(1),
        1e-5,
    );
}
