//! Axis-aligned bounding boxes.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const sphere = @import("sphere.zig");

/// An axis-aligned bounding box.
///
/// # Args
/// - `T`: the component type.
pub fn Aabb(comptime T: type) type {
    return struct {
        /// Corner with the smallest coordinates.
        min: vector.Vector(T, 3),
        /// Corner with the largest coordinates.
        max: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a box from a point slice.
        ///
        /// # Args
        /// - `points`: the points to enclose.
        ///
        /// # Return
        /// The smallest containing box, or `null`
        /// when the slice is empty.
        pub fn fromPoints(
            points: []const vector.Vector(T, 3),
        ) ?Self {
            if (points.len == 0) return null;
            var box_min = points[0];
            var box_max = points[0];
            for (points[1..]) |point| {
                box_min = .{ .elements = @min(
                    box_min.elements,
                    point.elements,
                ) };
                box_max = .{ .elements = @max(
                    box_max.elements,
                    point.elements,
                ) };
            }
            return .{ .min = box_min, .max = box_max };
        }

        /// Builds a box from a center and extents.
        ///
        /// # Args
        /// - `center_value`: midpoint of the box.
        /// - `half_extents`: half the size per axis.
        ///
        /// # Return
        /// The box spanning the extents.
        pub fn fromCenterHalfExtents(
            center_value: vector.Vector(T, 3),
            half_extents: vector.Vector(T, 3),
        ) Self {
            return .{
                .min = center_value.sub(half_extents),
                .max = center_value.add(half_extents),
            };
        }

        /// Computes the union of two boxes.
        ///
        /// # Args
        /// - `other`: the other box.
        ///
        /// # Return
        /// The smallest box containing both.
        pub fn merge(self: Self, other: Self) Self {
            return .{
                .min = .{ .elements = @min(
                    self.min.elements,
                    other.min.elements,
                ) },
                .max = .{ .elements = @max(
                    self.max.elements,
                    other.max.elements,
                ) },
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
            return @reduce(.And, point.elements >=
                self.min.elements) and
                @reduce(.And, point.elements <= self.max.elements);
        }

        /// Tests whether two boxes overlap.
        ///
        /// # Args
        /// - `other`: the other box.
        ///
        /// # Return
        /// `true` when the boxes touch or overlap.
        pub fn intersects(self: Self, other: Self) bool {
            return @reduce(.And, self.min.elements <=
                other.max.elements) and
                @reduce(.And, self.max.elements >=
                    other.min.elements);
        }

        /// Returns the center of the box.
        ///
        /// # Return
        /// The midpoint of the two corners.
        pub fn center(self: Self) vector.Vector(T, 3) {
            return self.min.add(self.max).scale(0.5);
        }

        /// Returns the size of the box.
        ///
        /// # Return
        /// The extent along each axis.
        pub fn size(self: Self) vector.Vector(T, 3) {
            return self.max.sub(self.min);
        }

        /// Computes the bounding sphere.
        ///
        /// # Return
        /// A sphere around the box; it is not the
        /// smallest possible sphere.
        pub fn toSphere(self: Self) sphere.Sphere(T) {
            return .{
                .center = self.center(),
                .radius = self.size().scale(0.5).length(),
            };
        }

        /// Returns the closest point inside the box.
        ///
        /// # Args
        /// - `point`: the point to clamp.
        ///
        /// # Return
        /// `point` when inside, otherwise the nearest
        /// surface point.
        pub fn closestPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            return .{ .elements = @min(
                @max(point.elements, self.min.elements),
                self.max.elements,
            ) };
        }
    };
}

test "aabb construction and merge" {
    const Vec3f = vector.Vec3(f32);
    const a = Aabb(f32){
        .min = Vec3f.splat(-1),
        .max = Vec3f.splat(1),
    };
    const b = Aabb(f32){
        .min = Vec3f.zero(),
        .max = Vec3f.splat(2),
    };
    try std.testing.expect(a.containsPoint(Vec3f.zero()));
    try std.testing.expect(!a.containsPoint(
        Vec3f.fromElements(.{ 2, 0, 0 }),
    ));
    try std.testing.expect(a.intersects(b));
    try std.testing.expect(!a.intersects(Aabb(f32){
        .min = Vec3f.splat(3),
        .max = Vec3f.splat(4),
    }));

    const merged = a.merge(b);
    try std.testing.expectEqual(@as(f32, -1), merged.min.at(0));
    try std.testing.expectEqual(@as(f32, 2), merged.max.at(0));
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        merged.center().at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        merged.size().at(0),
        1e-6,
    );

    const points = [_]Vec3f{
        Vec3f.fromElements(.{ 1, 2, 3 }),
        Vec3f.fromElements(.{ -1, 0, 5 }),
    };
    const from_points = Aabb(f32).fromPoints(&points).?;
    try std.testing.expectEqual(@as(f32, -1), from_points.min.at(0));
    try std.testing.expectEqual(@as(f32, 5), from_points.max.at(2));
    try std.testing.expect(
        Aabb(f32).fromPoints(&[_]Vec3f{}) == null,
    );

    const closest = a.closestPoint(
        Vec3f.fromElements(.{ 3, 0, 0 }),
    );
    try std.testing.expectEqual(@as(f32, 1), closest.at(0));
}

test "aabb sphere and center extents" {
    const Vec3f = vector.Vec3(f32);
    const box = Aabb(f32){
        .min = Vec3f.splat(-1),
        .max = Vec3f.splat(1),
    };
    const enclosing = box.toSphere();
    try std.testing.expectApproxEqAbs(
        @as(f32, @sqrt(3)),
        enclosing.radius,
        1e-5,
    );
    try std.testing.expect(enclosing.containsPoint(
        Vec3f.fromElements(.{ 1, 1, 1 }),
    ));

    const centered = Aabb(f32).fromCenterHalfExtents(
        Vec3f.splat(1),
        Vec3f.splat(2),
    );
    try std.testing.expectEqual(@as(f32, -1), centered.min.at(0));
    try std.testing.expectEqual(@as(f32, 3), centered.max.at(0));
}
