//! Rays and intersection tests.
//!
//! Ray parameters are expressed in units of the stored
//! direction, so a non-unit direction changes the
//! meaning of `t`. Parallel tests take their tolerance
//! from the caller.

const std = @import("std");
const base = @import("../base/root.zig");
const vector = @import("../linear/vector.zig");
const plane = @import("plane.zig");
const aabb = @import("aabb.zig");
const sphere = @import("sphere.zig");
const triangle = @import("triangle.zig");

/// A half-line with an origin and a direction.
///
/// # Args
/// - `T`: the component type.
pub fn Ray(comptime T: type) type {
    return struct {
        /// Start of the ray.
        origin: vector.Vector(T, 3),
        /// Direction of travel; need not be unit.
        direction: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a ray from an origin and direction.
        ///
        /// # Args
        /// - `origin`: start of the ray.
        /// - `direction`: travel direction.
        ///
        /// # Return
        /// The ray through the origin.
        pub fn fromPoints(
            origin: vector.Vector(T, 3),
            direction: vector.Vector(T, 3),
        ) Self {
            return .{ .origin = origin, .direction = direction };
        }

        /// Intersects the ray with a plane.
        ///
        /// # Args
        /// - `target`: the plane to hit.
        /// - `epsilon_value`: largest considered
        ///   parallel denominator.
        ///
        /// # Return
        /// The ray parameter, or `null` when the ray is
        /// parallel or the hit lies behind.
        ///
        /// # Note
        /// The plane normal should be unit for `t` to
        /// be a distance.
        pub fn intersectPlane(
            self: Self,
            target: plane.Plane(T),
            epsilon_value: T,
        ) ?T {
            const denominator = self.direction.dot(
                target.normal,
            );
            if (base.compare.isZero(
                T,
                denominator,
                epsilon_value,
            )) {
                return null;
            }
            const t = -(self.origin.dot(target.normal) +
                target.distance) / denominator;
            if (t < 0) return null;
            return t;
        }

        /// Intersects the ray with a triangle.
        ///
        /// # Args
        /// - `target`: the triangle to hit.
        /// - `epsilon_value`: largest considered
        ///   parallel determinant.
        ///
        /// # Return
        /// The hit parameters, or `null` when the ray
        /// misses or lies behind.
        ///
        /// # Note
        /// The triangle is double-sided. The weights
        /// `u` and `v` belong to the second and third
        /// vertices.
        pub fn intersectTriangle(
            self: Self,
            target: triangle.Triangle(T),
            epsilon_value: T,
        ) ?triangle.TriangleHit(T) {
            const edge_ab = target.b.sub(target.a);
            const edge_ac = target.c.sub(target.a);
            const p = self.direction.cross(edge_ac);
            const determinant = edge_ab.dot(p);
            if (base.compare.isZero(
                T,
                determinant,
                epsilon_value,
            )) {
                return null;
            }
            const inverse = 1 / determinant;
            const s = self.origin.sub(target.a);
            const u = inverse * s.dot(p);
            if (u < 0 or u > 1) return null;

            const q = s.cross(edge_ab);
            const v = inverse * self.direction.dot(q);
            if (v < 0 or u + v > 1) return null;

            const t = inverse * edge_ac.dot(q);
            if (t < 0) return null;
            return .{ .t = t, .u = u, .v = v };
        }

        /// Intersects the ray with an aligned box.
        ///
        /// # Args
        /// - `box`: the box to hit.
        /// - `epsilon_value`: largest considered zero
        ///   direction component.
        ///
        /// # Return
        /// The first non-negative ray parameter, or
        /// `null` when the ray misses. An origin inside
        /// the box returns zero.
        pub fn intersectAabb(
            self: Self,
            box: aabb.Aabb(T),
            epsilon_value: T,
        ) ?T {
            var near: T = -std.math.inf(T);
            var far: T = std.math.inf(T);
            inline for (0..3) |axis| {
                const direction = self.direction.elements[axis];
                const origin = self.origin.elements[axis];
                if (base.compare.isZero(
                    T,
                    direction,
                    epsilon_value,
                )) {
                    if (origin < box.min.elements[axis] or
                        origin > box.max.elements[axis])
                    {
                        return null;
                    }
                } else {
                    const inverse = 1 / direction;
                    var first = (box.min.elements[axis] -
                        origin) * inverse;
                    var second = (box.max.elements[axis] -
                        origin) * inverse;
                    if (first > second) {
                        const swap = first;
                        first = second;
                        second = swap;
                    }
                    near = @max(near, first);
                    far = @min(far, second);
                    if (near > far) return null;
                }
            }
            if (far < 0) return null;
            return @max(near, 0);
        }

        /// Intersects the ray with a sphere.
        ///
        /// # Args
        /// - `target`: the sphere to hit.
        /// - `epsilon_value`: largest considered zero
        ///   squared direction length.
        ///
        /// # Return
        /// Both crossing parameters, or `null` when the
        /// ray misses or the direction is too short.
        ///
        /// # Note
        /// `near` is negative when the origin is
        /// inside the sphere.
        pub fn intersectSphere(
            self: Self,
            target: sphere.Sphere(T),
            epsilon_value: T,
        ) ?SphereHit(T) {
            const offset = self.origin.sub(target.center);
            const a = self.direction.lengthSquared();
            if (base.compare.isZero(T, a, epsilon_value)) {
                return null;
            }
            const b = 2 * offset.dot(self.direction);
            const c = offset.lengthSquared() -
                target.radius * target.radius;
            const discriminant = b * b - 4 * a * c;
            if (discriminant < 0) return null;

            const root = @sqrt(discriminant);
            const near = (-b - root) / (2 * a);
            const far = (-b + root) / (2 * a);
            if (far < 0) return null;
            return .{ .near = near, .far = far };
        }
    };
}

/// A ray-sphere intersection result.
///
/// # Args
/// - `T`: the component type.
pub fn SphereHit(comptime T: type) type {
    return struct {
        /// Parameter of the first crossing.
        near: T,
        /// Parameter of the second crossing.
        far: T,

        /// The component type.
        pub const Component = T;
    };
}

test "ray plane" {
    const Vec3f = vector.Vec3(f32);
    const ray = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    const target = plane.Plane(f32).fromPointNormal(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 5),
        ray.intersectPlane(target, 1e-6).?,
        1e-6,
    );

    const behind = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, -5 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    try std.testing.expect(
        behind.intersectPlane(target, 1e-6) == null,
    );

    const parallel = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 1, 0, 0 }),
    );
    try std.testing.expect(
        parallel.intersectPlane(target, 1e-6) == null,
    );
}

test "ray triangle" {
    const Vec3f = vector.Vec3(f32);
    const target = triangle.Triangle(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 0, 1, 0 }),
    );
    const ray = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0.25, 0.25, 1 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    const hit = ray.intersectTriangle(target, 1e-6).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1), hit.t, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), hit.u, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), hit.v, 1e-6);

    const miss = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 2, 2, 1 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    try std.testing.expect(miss.intersectTriangle(target, 1e-6) == null);

    const from_back = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0.25, 0.25, -1 }),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    try std.testing.expect(
        from_back.intersectTriangle(target, 1e-6) != null,
    );
}

test "ray aabb" {
    const Vec3f = vector.Vec3(f32);
    const box = aabb.Aabb(f32){
        .min = Vec3f.splat(-1),
        .max = Vec3f.splat(1),
    };
    const outside = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 4),
        outside.intersectAabb(box, 1e-6).?,
        1e-6,
    );

    const inside = Ray(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        inside.intersectAabb(box, 1e-6).?,
        1e-6,
    );

    const away = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    try std.testing.expect(away.intersectAabb(box, 1e-6) == null);

    const sideways = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 5, 0.5 }),
        Vec3f.fromElements(.{ 1, 0, 0 }),
    );
    try std.testing.expect(sideways.intersectAabb(box, 1e-6) == null);
}

test "ray sphere" {
    const Vec3f = vector.Vec3(f32);
    const target = sphere.Sphere(f32){
        .center = Vec3f.zero(),
        .radius = 1,
    };
    const ray = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    const hit = ray.intersectSphere(target, 1e-6).?;
    try std.testing.expectApproxEqAbs(@as(f32, 4), hit.near, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), hit.far, 1e-6);

    const inside = Ray(f32).fromPoints(
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    const inner_hit = inside.intersectSphere(target, 1e-6).?;
    try std.testing.expect(inner_hit.near < 0);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        inner_hit.far,
        1e-6,
    );

    const miss = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 5, 5 }),
        Vec3f.fromElements(.{ 0, 0, -1 }),
    );
    try std.testing.expect(miss.intersectSphere(target, 1e-6) == null);

    const away = Ray(f32).fromPoints(
        Vec3f.fromElements(.{ 0, 0, 5 }),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    );
    try std.testing.expect(away.intersectSphere(target, 1e-6) == null);
}
