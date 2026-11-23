//! Capsules around a line segment.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const sphere = @import("sphere.zig");

/// A capsule around a line segment.
///
/// # Args
/// - `T`: the component type.
pub fn Capsule(comptime T: type) type {
    return struct {
        /// First endpoint of the core segment.
        a: vector.Vector(T, 3),
        /// Second endpoint of the core segment.
        b: vector.Vector(T, 3),
        /// Distance from the segment to the surface.
        radius: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a capsule from a segment.
        ///
        /// # Args
        /// - `a`: first segment endpoint.
        /// - `b`: second segment endpoint.
        /// - `radius`: distance from the segment to
        ///   the surface.
        ///
        /// # Return
        /// The capsule around the segment.
        pub fn fromSegment(
            a: vector.Vector(T, 3),
            b: vector.Vector(T, 3),
            radius: T,
        ) Self {
            return .{ .a = a, .b = b, .radius = radius };
        }

        /// Tests whether a point is inside.
        ///
        /// # Args
        /// - `point`: the point to test.
        ///
        /// # Return
        /// `true` when the point is within the radius
        /// of the core segment.
        pub fn containsPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) bool {
            return self.distanceTo(point) <= self.radius;
        }

        /// Tests whether a sphere overlaps.
        ///
        /// # Args
        /// - `other`: the sphere to test.
        ///
        /// # Return
        /// `true` when the surfaces touch or overlap.
        pub fn intersectsSphere(
            self: Self,
            other: sphere.Sphere(T),
        ) bool {
            return self.distanceTo(other.center) <=
                self.radius + other.radius;
        }

        fn distanceTo(
            self: Self,
            point: vector.Vector(T, 3),
        ) T {
            const segment = self.b.sub(self.a);
            const length_squared = segment.lengthSquared();
            if (length_squared == 0) {
                return point.distance(self.a);
            }
            const offset = point.sub(self.a);
            const factor = @min(@max(
                offset.dot(segment) / length_squared,
                0,
            ), 1);
            const closest = self.a.add(
                segment.scale(factor),
            );
            return point.distance(closest);
        }
    };
}

test "capsule" {
    const Vec3f = vector.Vec3(f32);
    const capsule = Capsule(f32).fromSegment(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 2, 0 }),
        0.5,
    );
    try std.testing.expect(capsule.containsPoint(
        Vec3f.fromElements(.{ 0, 1, 0.4 }),
    ));
    try std.testing.expect(!capsule.containsPoint(
        Vec3f.fromElements(.{ 0, 1, 0.6 }),
    ));
    try std.testing.expect(capsule.intersectsSphere(
        sphere.Sphere(f32){
            .center = Vec3f.fromElements(.{ 0, 1, 1 }),
            .radius = 0.5,
        },
    ));
    try std.testing.expect(!capsule.intersectsSphere(
        sphere.Sphere(f32){
            .center = Vec3f.fromElements(.{ 0, 1, 2 }),
            .radius = 0.5,
        },
    ));
}
