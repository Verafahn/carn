//! Line segments and point queries.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// A segment between two endpoints.
///
/// # Args
/// - `T`: the component type.
pub fn LineSegment(comptime T: type) type {
    return struct {
        /// First endpoint.
        start: vector.Vector(T, 3),
        /// Second endpoint.
        end: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a segment from two points.
        ///
        /// # Args
        /// - `start`: first endpoint.
        /// - `end`: second endpoint.
        ///
        /// # Return
        /// The segment between them.
        pub fn fromPoints(
            start: vector.Vector(T, 3),
            end: vector.Vector(T, 3),
        ) Self {
            return .{ .start = start, .end = end };
        }

        /// Returns the closest point on the segment.
        ///
        /// # Args
        /// - `point`: the query point.
        ///
        /// # Return
        /// The nearest point on the segment.
        ///
        /// # Note
        /// A zero-length segment returns `start`.
        pub fn closestPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            const segment = self.end.sub(self.start);
            const length_squared = segment.lengthSquared();
            if (length_squared == 0) return self.start;
            const offset = point.sub(self.start);
            const factor = @min(@max(
                offset.dot(segment) / length_squared,
                0,
            ), 1);
            return self.start.add(segment.scale(factor));
        }

        /// Returns the distance from a point.
        ///
        /// # Args
        /// - `point`: the query point.
        ///
        /// # Return
        /// The distance to the nearest segment point.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn distance(
            self: Self,
            point: vector.Vector(T, 3),
        ) T {
            return point.distance(self.closestPoint(point));
        }
    };
}

test "segment queries" {
    const Vec3f = vector.Vec3(f32);
    const segment = LineSegment(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 2, 0 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        segment.distance(Vec3f.fromElements(.{ 1, 1, 0 })),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        segment.distance(Vec3f.fromElements(.{ 0, 3, 0 })),
        1e-6,
    );
    const closest = segment.closestPoint(
        Vec3f.fromElements(.{ 1, 1, 0 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0), closest.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), closest.at(1), 1e-6);

    const degenerate = LineSegment(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.zero(),
    );
    try std.testing.expectEqual(
        Vec3f.zero(),
        degenerate.closestPoint(Vec3f.splat(1)),
    );
}
