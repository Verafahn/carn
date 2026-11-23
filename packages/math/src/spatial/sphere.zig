//! Spheres.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// A sphere with a center and a radius.
///
/// # Args
/// - `T`: the component type.
pub fn Sphere(comptime T: type) type {
    return struct {
        /// Center of the sphere.
        center: vector.Vector(T, 3),
        /// Radius of the sphere.
        radius: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;

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
            return self.center.distanceSquared(point) <=
                self.radius * self.radius;
        }

        /// Tests whether two spheres overlap.
        ///
        /// # Args
        /// - `other`: the other sphere.
        ///
        /// # Return
        /// `true` when the surfaces touch or overlap.
        pub fn intersects(self: Self, other: Self) bool {
            const radius = self.radius + other.radius;
            return self.center.distanceSquared(other.center) <=
                radius * radius;
        }

        /// Computes the smallest sphere enclosing both.
        ///
        /// # Args
        /// - `other`: the other sphere.
        ///
        /// # Return
        /// The merged sphere.
        ///
        /// # Note
        /// When one sphere contains the other, that
        /// sphere is returned unchanged.
        pub fn merge(self: Self, other: Self) Self {
            const offset = other.center.sub(self.center);
            const distance = offset.length();
            if (self.radius >= distance + other.radius) {
                return self;
            }
            if (other.radius >= distance + self.radius) {
                return other;
            }
            const radius = (distance + self.radius +
                other.radius) * 0.5;
            return .{
                .center = self.center.add(offset.scale(
                    (radius - self.radius) / distance,
                )),
                .radius = radius,
            };
        }
    };
}

test "sphere tests" {
    const Vec3f = vector.Vec3(f32);
    const unit = Sphere(f32){
        .center = Vec3f.zero(),
        .radius = 1,
    };
    try std.testing.expect(unit.containsPoint(
        Vec3f.fromElements(.{ 0, 0, 1 }),
    ));
    try std.testing.expect(!unit.containsPoint(
        Vec3f.fromElements(.{ 0, 0, 2 }),
    ));

    const other = Sphere(f32){
        .center = Vec3f.fromElements(.{ 2, 0, 0 }),
        .radius = 1,
    };
    try std.testing.expect(unit.intersects(other));
    const far = Sphere(f32){
        .center = Vec3f.fromElements(.{ 5, 0, 0 }),
        .radius = 1,
    };
    try std.testing.expect(!unit.intersects(far));

    const merged = unit.merge(far);
    try std.testing.expectApproxEqAbs(
        @as(f32, 2.5),
        merged.center.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3.5),
        merged.radius,
        1e-6,
    );
    try std.testing.expect(merged.containsPoint(
        Vec3f.fromElements(.{ 0, 0, 1 }),
    ));
    try std.testing.expect(merged.containsPoint(
        Vec3f.fromElements(.{ 5, 0, 1 }),
    ));

    const contained = unit.merge(Sphere(f32){
        .center = Vec3f.zero(),
        .radius = 0.5,
    });
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        contained.radius,
        1e-6,
    );
}
