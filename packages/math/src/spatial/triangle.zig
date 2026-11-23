//! Triangles and point queries.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// A triangle with three vertices.
///
/// # Args
/// - `T`: the component type.
pub fn Triangle(comptime T: type) type {
    return struct {
        /// First vertex.
        a: vector.Vector(T, 3),
        /// Second vertex.
        b: vector.Vector(T, 3),
        /// Third vertex.
        c: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a triangle from three vertices.
        ///
        /// # Args
        /// - `a`: first vertex.
        /// - `b`: second vertex.
        /// - `c`: third vertex.
        ///
        /// # Return
        /// The triangle through the vertices.
        pub fn fromPoints(
            a: vector.Vector(T, 3),
            b: vector.Vector(T, 3),
            c: vector.Vector(T, 3),
        ) Self {
            return .{ .a = a, .b = b, .c = c };
        }

        /// Computes the unnormalized normal.
        ///
        /// # Return
        /// `(b - a) x (c - a)`; its length is twice
        /// the area.
        pub fn normal(self: Self) vector.Vector(T, 3) {
            return self.b.sub(self.a).cross(self.c.sub(self.a));
        }

        /// Returns the closest point on the triangle.
        ///
        /// # Args
        /// - `point`: the query point.
        ///
        /// # Return
        /// The nearest point on the triangle.
        ///
        /// # Note
        /// Edges are closed and the triangle is
        /// double-sided. A degenerate triangle produces
        /// non-finite values.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn closestPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            const ab = self.b.sub(self.a);
            const ac = self.c.sub(self.a);
            const ap = point.sub(self.a);
            const d1 = ab.dot(ap);
            const d2 = ac.dot(ap);
            if (d1 <= 0 and d2 <= 0) return self.a;

            const bp = point.sub(self.b);
            const d3 = ab.dot(bp);
            const d4 = ac.dot(bp);
            if (d3 >= 0 and d4 <= d3) return self.b;

            const vc = d1 * d4 - d3 * d2;
            if (vc <= 0 and d1 >= 0 and d3 <= 0) {
                const factor = d1 / (d1 - d3);
                return self.a.add(ab.scale(factor));
            }

            const cp = point.sub(self.c);
            const d5 = ab.dot(cp);
            const d6 = ac.dot(cp);
            if (d6 >= 0 and d5 <= d6) return self.c;

            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 and d2 >= 0 and d6 <= 0) {
                const factor = d2 / (d2 - d6);
                return self.a.add(ac.scale(factor));
            }

            const va = d3 * d6 - d5 * d4;
            if (va <= 0 and (d4 - d3) >= 0 and
                (d5 - d6) >= 0)
            {
                const factor = (d4 - d3) /
                    ((d4 - d3) + (d5 - d6));
                return self.b.add(
                    self.c.sub(self.b).scale(factor),
                );
            }

            const denominator = 1 / (va + vb + vc);
            const v = vb * denominator;
            const w = vc * denominator;
            return self.a.add(ab.scale(v)).add(ac.scale(w));
        }

        /// Returns the distance from a point.
        ///
        /// # Args
        /// - `point`: the query point.
        ///
        /// # Return
        /// The distance to the nearest triangle point.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn distance(
            self: Self,
            point: vector.Vector(T, 3),
        ) T {
            return point.distance(self.closestPoint(point));
        }

        /// Computes barycentric coordinates.
        ///
        /// # Args
        /// - `point`: the query point; it is projected
        ///   onto the triangle plane.
        ///
        /// # Return
        /// Weights `.{ u, v, w }` for the three
        /// vertices; they sum to one.
        ///
        /// # Note
        /// A degenerate triangle produces non-finite
        /// weights.
        pub fn barycentric(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            const v0 = self.b.sub(self.a);
            const v1 = self.c.sub(self.a);
            const v2 = point.sub(self.a);
            const d00 = v0.dot(v0);
            const d01 = v0.dot(v1);
            const d11 = v1.dot(v1);
            const d20 = v2.dot(v0);
            const d21 = v2.dot(v1);
            const denominator = d00 * d11 - d01 * d01;
            const v = (d11 * d20 - d01 * d21) /
                denominator;
            const w = (d00 * d21 - d01 * d20) /
                denominator;
            return vector.Vector(T, 3).fromElements(.{
                1 - v - w,
                v,
                w,
            });
        }
    };
}

/// A ray-triangle intersection result.
///
/// # Args
/// - `T`: the component type.
pub fn TriangleHit(comptime T: type) type {
    return struct {
        /// Ray parameter of the intersection.
        t: T,
        /// Barycentric weight of the second vertex.
        u: T,
        /// Barycentric weight of the third vertex.
        v: T,

        /// The component type.
        pub const Component = T;
    };
}

test "triangle queries" {
    const Vec3f = vector.Vec3(f32);
    const triangle = Triangle(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 0, 1, 0 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        triangle.distance(Vec3f.fromElements(.{ 0.25, 0.25, -1 })),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        triangle.distance(Vec3f.fromElements(.{ 2, 0, 0 })),
        1e-6,
    );
    const closest = triangle.closestPoint(
        Vec3f.fromElements(.{ 0.25, 0.25, 1 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0), closest.at(2), 1e-6);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        closest.at(0),
        1e-6,
    );

    const weights = triangle.barycentric(
        Vec3f.fromElements(.{ 0.25, 0.25, 0 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        weights.at(1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        weights.at(2),
        1e-6,
    );

    const normal = triangle.normal();
    try std.testing.expectApproxEqAbs(@as(f32, 0), normal.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), normal.at(2), 1e-6);
}
