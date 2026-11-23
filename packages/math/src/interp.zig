//! Interpolation, Bezier curves, and splines.
//!
//! Every function is generic over floats and
//! vectors: the arithmetic runs componentwise for
//! vectors, and scalar arguments are broadcast.
//!
//! Parameters are not clamped; pass a normalized
//! `t` when the curve should stay inside its
//! control points.

const std = @import("std");

/// Linearly interpolates between two values.
///
/// # Args
/// - `a`: the value at `t = 0`.
/// - `b`: the value at `t = 1`.
/// - `t`: interpolation factor.
///
/// # Return
/// The point between `a` and `b`.
pub fn lerp(a: anytype, b: anytype, t: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    const other: T = b;
    return a + (other - a) * toLike(T, t);
}

/// Returns zero below an edge and one at or above it.
///
/// # Args
/// - `edge`: the step position.
/// - `x`: the value to test.
///
/// # Return
/// `0` when `x < edge`, otherwise `1`.
pub fn step(edge: anytype, x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const edge_value = toLike(T, edge);
    return switch (@typeInfo(T)) {
        .vector => |info| @select(
            info.child,
            x < edge_value,
            @as(T, @splat(0)),
            @as(T, @splat(1)),
        ),
        else => if (x < edge_value) 0 else 1,
    };
}

/// Smoothly interpolates between two edges.
///
/// # Args
/// - `edge0`: value where the result is zero.
/// - `edge1`: value where the result is one.
/// - `x`: the value to map.
///
/// # Return
/// A cubic Hermite step clamped to `0..1`.
///
/// # Note
/// `edge0` and `edge1` must differ.
pub fn smoothStep(
    edge0: anytype,
    edge1: anytype,
    x: anytype,
) @TypeOf(x) {
    const T = @TypeOf(x);
    const one = broadcast(T, 1);
    const t = clamp01(
        T,
        (x - toLike(T, edge0)) /
            (toLike(T, edge1) - toLike(T, edge0)),
    );
    return t * t * (one * broadcast(T, 3) -
        broadcast(T, 2) * t);
}

/// Smoothly interpolates with a zero second
/// derivative at both edges.
///
/// # Args
/// - `edge0`: value where the result is zero.
/// - `edge1`: value where the result is one.
/// - `x`: the value to map.
///
/// # Return
/// A quintic step clamped to `0..1`.
///
/// # Note
/// `edge0` and `edge1` must differ.
pub fn smootherStep(
    edge0: anytype,
    edge1: anytype,
    x: anytype,
) @TypeOf(x) {
    const T = @TypeOf(x);
    const t = clamp01(
        T,
        (x - toLike(T, edge0)) /
            (toLike(T, edge1) - toLike(T, edge0)),
    );
    return t * t * t *
        (t * (t * broadcast(T, 6) - broadcast(T, 15)) +
            broadcast(T, 10));
}

/// Evaluates a quadratic Bezier curve.
///
/// # Args
/// - `p0`: start point.
/// - `p1`: control point.
/// - `p2`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the curve.
pub fn quadraticBezier(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const u = broadcast(T, 1) - toLike(T, t);
    const v = toLike(T, t);
    return a * (u * u) +
        b * (u * v * broadcast(T, 2)) +
        c * (v * v);
}

/// Evaluates the tangent of a quadratic Bezier
/// curve.
///
/// # Args
/// - `p0`: start point.
/// - `p1`: control point.
/// - `p2`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn quadraticBezierTangent(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const u = broadcast(T, 1) - toLike(T, t);
    const v = toLike(T, t);
    return (b - a) * (u * broadcast(T, 2)) +
        (c - b) * (v * broadcast(T, 2));
}

/// Evaluates a cubic Bezier curve.
///
/// # Args
/// - `p0`: start point.
/// - `p1`: first control point.
/// - `p2`: second control point.
/// - `p3`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the curve.
pub fn cubicBezier(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    p3: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const d: T = p3;
    const u = broadcast(T, 1) - toLike(T, t);
    const v = toLike(T, t);
    return a * (u * u * u) +
        b * (u * u * v * broadcast(T, 3)) +
        c * (u * v * v * broadcast(T, 3)) +
        d * (v * v * v);
}

/// Evaluates the tangent of a cubic Bezier curve.
///
/// # Args
/// - `p0`: start point.
/// - `p1`: first control point.
/// - `p2`: second control point.
/// - `p3`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn cubicBezierTangent(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    p3: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const d: T = p3;
    const u = broadcast(T, 1) - toLike(T, t);
    const v = toLike(T, t);
    return (b - a) * (u * u * broadcast(T, 3)) +
        (c - b) * (u * v * broadcast(T, 6)) +
        (d - c) * (v * v * broadcast(T, 3));
}

/// Evaluates a Catmull-Rom spline segment.
///
/// # Args
/// - `p0`: point before the segment.
/// - `p1`: segment start.
/// - `p2`: segment end.
/// - `p3`: point after the segment.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the segment from `p1` to `p2`.
pub fn catmullRom(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    p3: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const d: T = p3;
    const v = toLike(T, t);
    const v2 = v * v;
    const v3 = v2 * v;
    const part0 = b * broadcast(T, 2);
    const part1 = (c - a) * v;
    const part2 = (a * broadcast(T, 2) - b * broadcast(T, 5) +
        c * broadcast(T, 4) - d) * v2;
    const part3 = (b * broadcast(T, 3) -
        a - c * broadcast(T, 3) + d) * v3;
    return (part0 + part1 + part2 + part3) *
        broadcast(T, 0.5);
}

/// Evaluates the tangent of a Catmull-Rom segment.
///
/// # Args
/// - `p0`: point before the segment.
/// - `p1`: segment start.
/// - `p2`: segment end.
/// - `p3`: point after the segment.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn catmullRomTangent(
    p0: anytype,
    p1: anytype,
    p2: anytype,
    p3: anytype,
    t: anytype,
) @TypeOf(p0) {
    const T = @TypeOf(p0);
    const a = p0;
    const b: T = p1;
    const c: T = p2;
    const d: T = p3;
    const v = toLike(T, t);
    const part0 = c - a;
    const part1 = (a * broadcast(T, 2) - b * broadcast(T, 5) +
        c * broadcast(T, 4) - d) * (v * broadcast(T, 2));
    const part2 = (b * broadcast(T, 3) -
        a - c * broadcast(T, 3) + d) *
        (v * v * broadcast(T, 3));
    return (part0 + part1 + part2) * broadcast(T, 0.5);
}

fn broadcast(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => @splat(value),
        else => value,
    };
}

fn toLike(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => switch (@typeInfo(@TypeOf(value))) {
            .vector => value,
            else => @splat(value),
        },
        else => value,
    };
}

fn clamp01(comptime T: type, value: T) T {
    return @min(
        @max(value, broadcast(T, 0)),
        broadcast(T, 1),
    );
}

test lerp {
    try std.testing.expectApproxEqAbs(
        @as(f32, 5),
        lerp(@as(f32, 0), @as(f32, 10), 0.5),
        1e-6,
    );
    const a: @Vector(2, f32) = .{ 0, 10 };
    const b: @Vector(2, f32) = .{ 10, 20 };
    const middle = lerp(a, b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), middle[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 15), middle[1], 1e-6);
}

test "step and smooth step" {
    try std.testing.expectEqual(@as(f32, 0), step(
        @as(f32, 1),
        @as(f32, 0.5),
    ));
    try std.testing.expectEqual(@as(f32, 1), step(
        @as(f32, 1),
        @as(f32, 1),
    ));
    const vector = step(@as(f32, 1), @Vector(3, f32){ 0, 1, 2 });
    try std.testing.expectEqual(@as(f32, 0), vector[0]);
    try std.testing.expectEqual(@as(f32, 1), vector[1]);

    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        smoothStep(@as(f32, 0), @as(f32, 1), 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        smoothStep(@as(f32, 0), @as(f32, 1), 1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        smoothStep(@as(f32, 0), @as(f32, 1), 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        smootherStep(@as(f32, 0), @as(f32, 1), 0.5),
        1e-6,
    );
    const clamped = smoothStep(
        @as(f32, 0),
        @as(f32, 1),
        @as(f32, -2),
    );
    try std.testing.expectEqual(@as(f32, 0), clamped);
}

test "quadratic bezier" {
    const start: @Vector(2, f32) = .{ 0, 0 };
    const control: @Vector(2, f32) = .{ 1, 2 };
    const end: @Vector(2, f32) = .{ 2, 0 };
    const mid = quadraticBezier(start, control, end, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), mid[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), mid[1], 1e-6);
    const at_start = quadraticBezier(start, control, end, 0);
    try std.testing.expectEqual(start, at_start);
    const at_end = quadraticBezier(start, control, end, 1);
    try std.testing.expectEqual(end, at_end);

    const tangent = quadraticBezierTangent(
        start,
        control,
        end,
        0,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 2), tangent[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), tangent[1], 1e-6);
}

test "cubic bezier" {
    const a: f32 = 0;
    const b: f32 = 1;
    const c: f32 = 2;
    const d: f32 = 3;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        cubicBezier(a, b, c, d, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        cubicBezier(a, b, c, d, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        cubicBezier(a, b, c, d, 1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        cubicBezierTangent(a, b, c, d, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        cubicBezierTangent(a, b, c, d, 1),
        1e-6,
    );
}

test "catmull rom" {
    const a: f32 = 0;
    const b: f32 = 1;
    const c: f32 = 2;
    const d: f32 = 3;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        catmullRom(a, b, c, d, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        catmullRom(a, b, c, d, 1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        catmullRom(a, b, c, d, 0.5),
        1e-6,
    );

    const points = [_]@Vector(2, f32){
        .{ 0, 0 },
        .{ 1, 1 },
        .{ 2, 0 },
        .{ 3, 1 },
    };
    const midpoint = catmullRom(
        points[0],
        points[1],
        points[2],
        points[3],
        0.5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), midpoint[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), midpoint[1], 1e-6);

    const tangent = catmullRomTangent(
        points[0],
        points[1],
        points[2],
        points[3],
        0.5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), tangent[0], 1e-6);
}
