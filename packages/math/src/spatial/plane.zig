//! Planes in Hessian normal form.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// A plane with a normal and a distance.
///
/// The plane satisfies `dot(normal, p) + distance == 0`.
///
/// # Args
/// - `T`: the component type.
pub fn Plane(comptime T: type) type {
    return struct {
        /// Normal of the plane.
        normal: vector.Vector(T, 3),
        /// Signed distance from the origin.
        distance: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a plane from a point and a normal.
        ///
        /// # Args
        /// - `point`: a point on the plane.
        /// - `normal`: the plane normal.
        ///
        /// # Return
        /// The plane through `point`.
        ///
        /// # Note
        /// The normal is used as given; pass a unit
        /// vector so distances stay metric.
        pub fn fromPointNormal(
            point: vector.Vector(T, 3),
            normal: vector.Vector(T, 3),
        ) Self {
            return .{
                .normal = normal,
                .distance = -normal.dot(point),
            };
        }

        /// Returns the signed distance to a point.
        ///
        /// # Args
        /// - `point`: the point to measure.
        ///
        /// # Return
        /// The plane equation value; positive values
        /// are in front.
        ///
        /// # Note
        /// The distance is metric only for a unit
        /// normal.
        pub fn signedDistance(
            self: Self,
            point: vector.Vector(T, 3),
        ) T {
            return self.normal.dot(point) + self.distance;
        }

        /// Classifies a point against the plane.
        ///
        /// # Args
        /// - `point`: the point to classify.
        /// - `epsilon_value`: distance within which the
        ///   point is on the plane.
        ///
        /// # Return
        /// `behind`, `on`, or `front`.
        pub fn classify(
            self: Self,
            point: vector.Vector(T, 3),
            epsilon_value: T,
        ) Classification {
            const distance = self.signedDistance(point);
            if (distance > epsilon_value) return .front;
            if (distance < -epsilon_value) return .behind;
            return .on;
        }

        /// Normalizes the plane.
        ///
        /// # Return
        /// A plane with a unit normal.
        ///
        /// # Note
        /// A zero normal produces non-finite values.
        pub fn normalized(self: Self) Self {
            const factor = 1 / self.normal.length();
            return .{
                .normal = self.normal.scale(factor),
                .distance = self.distance * factor,
            };
        }

        /// Flips the plane to face the other way.
        ///
        /// # Return
        /// The plane with all signs negated.
        pub fn flipped(self: Self) Self {
            return .{
                .normal = self.normal.negate(),
                .distance = -self.distance,
            };
        }

        /// Projects a point onto the plane.
        ///
        /// # Args
        /// - `point`: the point to project.
        ///
        /// # Return
        /// The closest point on the plane.
        ///
        /// # Note
        /// The normal should be a unit vector.
        pub fn project(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            const distance = self.signedDistance(point);
            return point.sub(self.normal.scale(distance));
        }
    };
}

/// Where a point lies relative to a plane.
pub const Classification = enum {
    /// Negative distance; behind the plane.
    behind,
    /// Distance within the caller tolerance.
    on,
    /// Positive distance; in front of the plane.
    front,
};

test "plane distance and classification" {
    const Vec3f = vector.Vec3(f32);
    const plane = Plane(f32).fromPointNormal(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        plane.signedDistance(Vec3f.fromElements(.{ 0, 0, 2 })),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        plane.signedDistance(Vec3f.fromElements(.{ 0, 0, -2 })),
        1e-6,
    );
    try std.testing.expectEqual(
        Classification.front,
        plane.classify(
            Vec3f.fromElements(.{ 0, 0, 1 }),
            1e-6,
        ),
    );
    try std.testing.expectEqual(
        Classification.behind,
        plane.classify(
            Vec3f.fromElements(.{ 0, 0, -1 }),
            1e-6,
        ),
    );
    try std.testing.expectEqual(
        Classification.on,
        plane.classify(Vec3f.fromElements(.{ 1, 1, 0 }), 1e-6),
    );
}

test "plane transforms" {
    const Vec3f = vector.Vec3(f32);
    const plane = Plane(f32){
        .normal = Vec3f.fromElements(.{ 0, 0, 2 }),
        .distance = -4,
    };
    const normalized = plane.normalized();
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        normalized.normal.at(2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        normalized.distance,
        1e-6,
    );
    const flipped = normalized.flipped();
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        flipped.signedDistance(Vec3f.fromElements(.{ 0, 0, 2 })),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1),
        flipped.signedDistance(Vec3f.fromElements(.{ 0, 0, 3 })),
        1e-6,
    );
    const projected = normalized.project(
        Vec3f.fromElements(.{ 1, 2, 3 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), projected.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), projected.at(2), 1e-6);
}
