//! Rays, segments, triangles, and queries.
//!
//! Ray parameters are expressed in units of the
//! stored direction, so a non-unit direction changes
//! the meaning of `t`. Parallel tests take `epsilon`
//! from the caller.
//!
//! The functions are generic: each argument must
//! expose the documented fields with matching
//! component types, or the call fails to compile.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const bounds = @import("bounds.zig");
const Vec3f = vec.Vec3(f32);

/// A half-line with an origin and a direction.
///
/// # Args
/// - `T`: the component type.
pub fn Ray(comptime T: type) type {
    return struct {
        /// Start of the ray.
        origin: @Vector(3, T),
        /// Direction of travel; need not be unit.
        direction: @Vector(3, T),

        /// The component type.
        pub const Component = T;
    };
}

/// A segment between two endpoints.
///
/// # Args
/// - `T`: the component type.
pub fn LineSegment(comptime T: type) type {
    return struct {
        /// First endpoint.
        start: @Vector(3, T),
        /// Second endpoint.
        end: @Vector(3, T),

        /// The component type.
        pub const Component = T;
    };
}

/// A triangle with three vertices.
///
/// # Args
/// - `T`: the component type.
pub fn Triangle(comptime T: type) type {
    return struct {
        /// First vertex.
        a: @Vector(3, T),
        /// Second vertex.
        b: @Vector(3, T),
        /// Third vertex.
        c: @Vector(3, T),

        /// The component type.
        pub const Component = T;
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

/// Intersects a ray with a plane.
///
/// # Args
/// - `ray`: the ray to trace.
/// - `plane`: the plane to hit.
/// - `epsilon`: largest considered parallel
///   denominator.
///
/// # Return
/// The ray parameter, or `null` when the ray is
/// parallel or the hit lies behind the origin.
///
/// # Note
/// The plane normal should be a unit vector for
/// `t` to be a distance.
pub fn rayPlane(
    ray: anytype,
    plane: anytype,
    epsilon: anytype,
) ?@TypeOf(plane.distance) {
    const denominator = vec.dot(ray.direction, plane.normal);
    if (scalar.isZero(denominator, epsilon)) return null;
    const numerator = vec.dot(ray.origin, plane.normal) +
        plane.distance;
    const t = -numerator / denominator;
    if (t < 0) return null;
    return t;
}

/// Intersects a ray with a triangle.
///
/// The triangle is double-sided, so both winding
/// orders are hit.
///
/// # Args
/// - `ray`: the ray to trace.
/// - `triangle`: the triangle to hit.
/// - `epsilon`: largest considered parallel
///   determinant.
///
/// # Return
/// The hit parameters, or `null` when the ray
/// misses or lies behind the origin.
///
/// # Note
/// The weights `u` and `v` belong to the second and
/// third vertices; the first vertex weight is
/// `1 - u - v`.
pub fn rayTriangle(
    ray: anytype,
    triangle: anytype,
    epsilon: anytype,
) ?TriangleHit(scalar.ScalarOf(@TypeOf(ray.origin))) {
    const T = scalar.ScalarOf(@TypeOf(ray.origin));
    const edge_ab = triangle.b - triangle.a;
    const edge_ac = triangle.c - triangle.a;
    const p = vec.cross(ray.direction, edge_ac);
    const determinant = vec.dot(edge_ab, p);
    if (scalar.isZero(determinant, epsilon)) return null;

    const inverse = 1 / determinant;
    const s = ray.origin - triangle.a;
    const u = inverse * vec.dot(s, p);
    if (u < 0 or u > 1) return null;

    const q = vec.cross(s, edge_ab);
    const v = inverse * vec.dot(ray.direction, q);
    if (v < 0 or u + v > 1) return null;

    const t = inverse * vec.dot(edge_ac, q);
    if (t < 0) return null;
    return TriangleHit(T){ .t = t, .u = u, .v = v };
}

/// Intersects a ray with an axis-aligned box.
///
/// # Args
/// - `ray`: the ray to trace.
/// - `box`: the box to hit.
/// - `epsilon`: largest considered zero direction
///   component.
///
/// # Return
/// The first non-negative ray parameter, or `null`
/// when the ray misses. An origin inside the box
/// returns zero.
pub fn rayAabb(
    ray: anytype,
    box: anytype,
    epsilon: anytype,
) ?scalar.ScalarOf(@TypeOf(ray.direction)) {
    const T = scalar.ScalarOf(@TypeOf(ray.direction));
    var near: T = -std.math.inf(T);
    var far: T = std.math.inf(T);
    inline for (0..3) |axis| {
        const direction = ray.direction[axis];
        const origin = ray.origin[axis];
        if (scalar.isZero(direction, epsilon)) {
            if (origin < box.min[axis] or origin > box.max[axis]) {
                return null;
            }
        } else {
            const inverse = 1 / direction;
            var first = (box.min[axis] - origin) * inverse;
            var second = (box.max[axis] - origin) * inverse;
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

/// Intersects a ray with a sphere.
///
/// # Args
/// - `ray`: the ray to trace.
/// - `sphere`: the sphere to hit.
/// - `epsilon`: largest considered zero squared
///   direction length.
///
/// # Return
/// Both crossing parameters, or `null` when the ray
/// misses or the direction is too short.
///
/// # Note
/// `near` is negative when the origin is inside the
/// sphere.
pub fn raySphere(
    ray: anytype,
    sphere: anytype,
    epsilon: anytype,
) ?SphereHit(scalar.ScalarOf(@TypeOf(ray.origin))) {
    const T = scalar.ScalarOf(@TypeOf(ray.origin));
    const offset = ray.origin - sphere.center;
    const a = vec.dot(ray.direction, ray.direction);
    if (scalar.isZero(a, epsilon)) return null;
    const b = 2 * vec.dot(offset, ray.direction);
    const c = vec.dot(offset, offset) -
        sphere.radius * sphere.radius;
    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return null;

    const root = @sqrt(discriminant);
    const near = (-b - root) / (2 * a);
    const far = (-b + root) / (2 * a);
    if (far < 0) return null;
    return SphereHit(T){ .near = near, .far = far };
}

/// Returns the closest point on a segment.
///
/// # Args
/// - `point`: the query point.
/// - `start`: first segment endpoint.
/// - `end`: second segment endpoint.
///
/// # Return
/// The point on the segment nearest to `point`.
///
/// # Note
/// A zero-length segment returns `start`.
pub fn closestPointOnSegment(
    point: anytype,
    start: anytype,
    end: anytype,
) @TypeOf(point) {
    const V = @TypeOf(point);
    const segment = end - start;
    const length_squared = vec.lengthSquared(segment);
    if (length_squared == 0) return start;
    const factor = @min(@max(
        vec.dot(point - start, segment) / length_squared,
        0,
    ), 1);
    const offset = start +
        segment * @as(V, @splat(factor));
    return offset;
}

/// Returns the distance from a point to a segment.
///
/// # Args
/// - `point`: the query point.
/// - `start`: first segment endpoint.
/// - `end`: second segment endpoint.
///
/// # Return
/// The distance to the nearest segment point.
pub fn pointSegmentDistance(
    point: anytype,
    start: anytype,
    end: anytype,
) @TypeOf(point[0]) {
    const closest = closestPointOnSegment(point, start, end);
    return vec.length(point - closest);
}

/// Returns the closest point on a triangle.
///
/// # Args
/// - `point`: the query point.
/// - `triangle`: the triangle to project onto.
///
/// # Return
/// The point on the triangle nearest to `point`.
///
/// # Note
/// The edges are treated as closed and the triangle
/// is double-sided. A degenerate triangle produces
/// non-finite values.
pub fn closestPointOnTriangle(
    point: anytype,
    triangle: anytype,
) @TypeOf(point) {
    const V = @TypeOf(point);
    const ab = triangle.b - triangle.a;
    const ac = triangle.c - triangle.a;
    const ap = point - triangle.a;
    const d1 = vec.dot(ab, ap);
    const d2 = vec.dot(ac, ap);
    if (d1 <= 0 and d2 <= 0) return triangle.a;

    const bp = point - triangle.b;
    const d3 = vec.dot(ab, bp);
    const d4 = vec.dot(ac, bp);
    if (d3 >= 0 and d4 <= d3) return triangle.b;

    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 and d1 >= 0 and d3 <= 0) {
        const factor = d1 / (d1 - d3);
        return triangle.a + ab * @as(V, @splat(factor));
    }

    const cp = point - triangle.c;
    const d5 = vec.dot(ab, cp);
    const d6 = vec.dot(ac, cp);
    if (d6 >= 0 and d5 <= d6) return triangle.c;

    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 and d2 >= 0 and d6 <= 0) {
        const factor = d2 / (d2 - d6);
        return triangle.a + ac * @as(V, @splat(factor));
    }

    const va = d3 * d6 - d5 * d4;
    if (va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0) {
        const factor = (d4 - d3) /
            ((d4 - d3) + (d5 - d6));
        return triangle.b +
            (triangle.c - triangle.b) * @as(V, @splat(factor));
    }

    const denominator = 1 / (va + vb + vc);
    const v = vb * denominator;
    const w = vc * denominator;
    return triangle.a +
        ab * @as(V, @splat(v)) +
        ac * @as(V, @splat(w));
}

/// Returns the distance from a point to a triangle.
///
/// # Args
/// - `point`: the query point.
/// - `triangle`: the triangle to measure against.
///
/// # Return
/// The distance to the nearest triangle point.
pub fn pointTriangleDistance(
    point: anytype,
    triangle: anytype,
) @TypeOf(point[0]) {
    const closest = closestPointOnTriangle(point, triangle);
    return vec.length(point - closest);
}

/// Computes barycentric coordinates.
///
/// # Args
/// - `triangle`: the reference triangle.
/// - `point`: the query point; it is projected onto
///   the triangle plane.
///
/// # Return
/// Weights `.{ u, v, w }` for the three vertices;
/// they sum to one.
///
/// # Note
/// A degenerate triangle produces non-finite
/// weights.
pub fn barycentric(
    triangle: anytype,
    point: anytype,
) @Vector(3, @TypeOf(point[0])) {
    const v0 = triangle.b - triangle.a;
    const v1 = triangle.c - triangle.a;
    const v2 = point - triangle.a;
    const d00 = vec.dot(v0, v0);
    const d01 = vec.dot(v0, v1);
    const d11 = vec.dot(v1, v1);
    const d20 = vec.dot(v2, v0);
    const d21 = vec.dot(v2, v1);
    const denominator = d00 * d11 - d01 * d01;
    const v = (d11 * d20 - d01 * d21) / denominator;
    const w = (d00 * d21 - d01 * d20) / denominator;
    const u = 1 - v - w;
    return .{ u, v, w };
}

/// Computes the unnormalized triangle normal.
///
/// # Args
/// - `triangle`: the triangle to measure.
///
/// # Return
/// `(b - a) x (c - a)`; its length is twice the
/// triangle area.
pub fn triangleNormal(triangle: anytype) @TypeOf(triangle.a) {
    return vec.cross(
        triangle.b - triangle.a,
        triangle.c - triangle.a,
    );
}

test "ray plane" {
    const ray = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 0, 0, -1 },
    };
    const plane = bounds.fromPointNormal(
        Vec3f{ 0, 0, 0 },
        Vec3f{ 0, 0, 1 },
    );
    const t = rayPlane(ray, plane, 1e-6).?;
    try std.testing.expectApproxEqAbs(@as(f32, 5), t, 1e-6);

    const behind = Ray(f32){
        .origin = .{ 0, 0, -5 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expect(rayPlane(behind, plane, 1e-6) == null);

    const parallel = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 1, 0, 0 },
    };
    try std.testing.expect(rayPlane(parallel, plane, 1e-6) == null);
}

test "ray triangle" {
    const triangle = Triangle(f32){
        .a = .{ 0, 0, 0 },
        .b = .{ 1, 0, 0 },
        .c = .{ 0, 1, 0 },
    };
    const ray = Ray(f32){
        .origin = .{ 0.25, 0.25, 1 },
        .direction = .{ 0, 0, -1 },
    };
    const hit = rayTriangle(ray, triangle, 1e-6).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1), hit.t, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), hit.u, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), hit.v, 1e-6);

    const centroid = TriangleHit(f32){
        .t = 1,
        .u = 1.0 / 3.0,
        .v = 1.0 / 3.0,
    };
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 3.0),
        centroid.u,
        1e-6,
    );

    const miss = Ray(f32){
        .origin = .{ 2, 2, 1 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expect(rayTriangle(miss, triangle, 1e-6) == null);

    const below = Ray(f32){
        .origin = .{ 0.25, 0.25, -1 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expect(rayTriangle(below, triangle, 1e-6) == null);

    // The triangle is double-sided.
    const from_back = Ray(f32){
        .origin = .{ 0.25, 0.25, -1 },
        .direction = .{ 0, 0, 1 },
    };
    try std.testing.expect(rayTriangle(from_back, triangle, 1e-6) != null);
}

test "ray aabb" {
    const box = bounds.Aabb(f32){
        .min = .{ -1, -1, -1 },
        .max = .{ 1, 1, 1 },
    };
    const outside = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expectApproxEqAbs(
        @as(f32, 4),
        rayAabb(outside, box, 1e-6).?,
        1e-6,
    );

    const inside = Ray(f32){
        .origin = .{ 0, 0, 0 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        rayAabb(inside, box, 1e-6).?,
        1e-6,
    );

    const away = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 0, 0, 1 },
    };
    try std.testing.expect(rayAabb(away, box, 1e-6) == null);

    const sideways = Ray(f32){
        .origin = .{ 0, 5, 0.5 },
        .direction = .{ 1, 0, 0 },
    };
    try std.testing.expect(rayAabb(sideways, box, 1e-6) == null);
}

test "ray sphere" {
    const sphere = bounds.Sphere(f32){
        .center = .{ 0, 0, 0 },
        .radius = 1,
    };
    const ray = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 0, 0, -1 },
    };
    const hit = raySphere(ray, sphere, 1e-6).?;
    try std.testing.expectApproxEqAbs(@as(f32, 4), hit.near, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), hit.far, 1e-6);

    const inside = Ray(f32){
        .origin = .{ 0, 0, 0 },
        .direction = .{ 0, 0, 1 },
    };
    const inner_hit = raySphere(inside, sphere, 1e-6).?;
    try std.testing.expect(inner_hit.near < 0);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        inner_hit.far,
        1e-6,
    );

    const miss = Ray(f32){
        .origin = .{ 0, 5, 5 },
        .direction = .{ 0, 0, -1 },
    };
    try std.testing.expect(raySphere(miss, sphere, 1e-6) == null);

    const away = Ray(f32){
        .origin = .{ 0, 0, 5 },
        .direction = .{ 0, 0, 1 },
    };
    try std.testing.expect(raySphere(away, sphere, 1e-6) == null);
}

test "point segment distance" {
    const start: Vec3f = .{ 0, 0, 0 };
    const end: Vec3f = .{ 0, 2, 0 };
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        pointSegmentDistance(Vec3f{ 1, 1, 0 }, start, end),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        pointSegmentDistance(Vec3f{ 0, 3, 0 }, start, end),
        1e-6,
    );
    const closest = closestPointOnSegment(
        Vec3f{ 1, 1, 0 },
        start,
        end,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0), closest[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), closest[1], 1e-6);
    const degenerate = closestPointOnSegment(
        Vec3f{ 1, 1, 0 },
        start,
        start,
    );
    try std.testing.expectEqual(start, degenerate);
}

test "point triangle distance" {
    const triangle = Triangle(f32){
        .a = .{ 0, 0, 0 },
        .b = .{ 1, 0, 0 },
        .c = .{ 0, 1, 0 },
    };
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        pointTriangleDistance(Vec3f{ 0.25, 0.25, -1 }, triangle),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        pointTriangleDistance(Vec3f{ 2, 0, 0 }, triangle),
        1e-6,
    );
    const closest = closestPointOnTriangle(
        Vec3f{ 0.25, 0.25, 1 },
        triangle,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0), closest[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), closest[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), closest[1], 1e-6);
}

test "barycentric and normal" {
    const triangle = Triangle(f32){
        .a = .{ 0, 0, 0 },
        .b = .{ 1, 0, 0 },
        .c = .{ 0, 1, 0 },
    };
    const weights = barycentric(triangle, Vec3f{ 0.25, 0.25, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), weights[2], 1e-6);

    const normal = triangleNormal(triangle);
    try std.testing.expectApproxEqAbs(@as(f32, 0), normal[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), normal[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), normal[2], 1e-6);
}
