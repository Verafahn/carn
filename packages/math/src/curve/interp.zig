//! Interpolation, Bezier curves, and splines.
//!
//! Curves accept floats and vectors: the value type
//! determines the scalar factor type. Parameters are
//! not clamped, so pass a normalized `t` when the
//! curve should stay inside its control points.

const std = @import("std");

/// Returns the factor type for a value type.
///
/// # Args
/// - `T`: a float or vector type.
///
/// # Return
/// The float type used for scalar factors.
///
/// # Compile Error
/// - `T` is not a float or a supported vector.
pub fn FactorOf(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .float => T,
        .@"struct" => T.Component,
        else => @compileError(
            "expected a float or vector type, got `" ++
                @typeName(T) ++ "`",
        ),
    };
}

/// Linearly interpolates between two values.
///
/// # Args
/// - `T`: the value type; a float or vector.
/// - `a`: the value at `t = 0`.
/// - `b`: the value at `t = 1`.
/// - `t`: the interpolation factor.
///
/// # Return
/// The point between `a` and `b`.
pub fn lerp(
    comptime T: type,
    a: T,
    b: T,
    t: FactorOf(T),
) T {
    return add(T, a, scale(T, sub(T, b, a), t));
}

/// Returns zero below an edge and one at or above it.
///
/// # Args
/// - `T`: a float type.
/// - `edge`: the step position.
/// - `x`: the value to test.
///
/// # Return
/// `0` when `x < edge`, otherwise `1`.
pub fn step(comptime T: type, edge: T, x: T) T {
    assertFloat(T);
    return if (x < edge) 0 else 1;
}

/// Smoothly interpolates between two edges.
///
/// # Args
/// - `T`: a float type.
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
    comptime T: type,
    edge0: T,
    edge1: T,
    x: T,
) T {
    assertFloat(T);
    const t = clamp01(T, (x - edge0) / (edge1 - edge0));
    return t * t * (3 - 2 * t);
}

/// Smoothly interpolates with a zero second
/// derivative at both edges.
///
/// # Args
/// - `T`: a float type.
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
    comptime T: type,
    edge0: T,
    edge1: T,
    x: T,
) T {
    assertFloat(T);
    const t = clamp01(T, (x - edge0) / (edge1 - edge0));
    return t * t * t * (t * (t * 6 - 15) + 10);
}

/// Evaluates a quadratic Bezier curve.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: start point.
/// - `p1`: control point.
/// - `p2`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the curve.
pub fn quadraticBezier(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    t: FactorOf(T),
) T {
    const u = 1 - t;
    return add(T, add(
        T,
        scale(T, p0, u * u),
        scale(T, p1, 2 * u * t),
    ), scale(T, p2, t * t));
}

/// Evaluates the tangent of a quadratic Bezier
/// curve.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: start point.
/// - `p1`: control point.
/// - `p2`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn quadraticBezierTangent(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    t: FactorOf(T),
) T {
    const u = 1 - t;
    return add(
        T,
        scale(T, sub(T, p1, p0), 2 * u),
        scale(T, sub(T, p2, p1), 2 * t),
    );
}

/// Evaluates a cubic Bezier curve.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: start point.
/// - `p1`: first control point.
/// - `p2`: second control point.
/// - `p3`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the curve.
pub fn cubicBezier(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    p3: T,
    t: FactorOf(T),
) T {
    const u = 1 - t;
    return add(T, add(
        T,
        add(
            T,
            scale(T, p0, u * u * u),
            scale(T, p1, 3 * u * u * t),
        ),
        scale(T, p2, 3 * u * t * t),
    ), scale(T, p3, t * t * t));
}

/// Evaluates the tangent of a cubic Bezier curve.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: start point.
/// - `p1`: first control point.
/// - `p2`: second control point.
/// - `p3`: end point.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn cubicBezierTangent(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    p3: T,
    t: FactorOf(T),
) T {
    const u = 1 - t;
    return add(T, add(
        T,
        scale(T, sub(T, p1, p0), 3 * u * u),
        scale(T, sub(T, p2, p1), 6 * u * t),
    ), scale(T, sub(T, p3, p2), 3 * t * t));
}

/// Evaluates a Catmull-Rom spline segment.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: point before the segment.
/// - `p1`: segment start.
/// - `p2`: segment end.
/// - `p3`: point after the segment.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The point on the segment from `p1` to `p2`.
pub fn catmullRom(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    p3: T,
    t: FactorOf(T),
) T {
    const t2 = t * t;
    const t3 = t2 * t;
    const part0 = scale(T, p1, 2);
    const part1 = scale(T, sub(T, p2, p0), t);
    const part2 = scale(
        T,
        sub(
            T,
            sub(T, scale(T, p0, 2), scale(T, p1, 5)),
            sub(T, scale(T, p3, 1), scale(T, p2, 4)),
        ),
        t2,
    );
    const part3 = scale(
        T,
        add(
            T,
            sub(T, scale(T, p1, 3), p0),
            sub(T, p3, scale(T, p2, 3)),
        ),
        t3,
    );
    return scale(
        T,
        add(T, add(T, part0, part1), add(T, part2, part3)),
        0.5,
    );
}

/// Evaluates the tangent of a Catmull-Rom segment.
///
/// # Args
/// - `T`: the point type; a float or vector.
/// - `p0`: point before the segment.
/// - `p1`: segment start.
/// - `p2`: segment end.
/// - `p3`: point after the segment.
/// - `t`: curve parameter in `0..1`.
///
/// # Return
/// The derivative with respect to `t`.
pub fn catmullRomTangent(
    comptime T: type,
    p0: T,
    p1: T,
    p2: T,
    p3: T,
    t: FactorOf(T),
) T {
    const part0 = sub(T, p2, p0);
    const part1 = scale(
        T,
        sub(
            T,
            sub(T, scale(T, p0, 2), scale(T, p1, 5)),
            sub(T, scale(T, p3, 1), scale(T, p2, 4)),
        ),
        2 * t,
    );
    const part2 = scale(
        T,
        add(
            T,
            sub(T, scale(T, p1, 3), p0),
            sub(T, p3, scale(T, p2, 3)),
        ),
        3 * t * t,
    );
    return scale(
        T,
        add(T, add(T, part0, part1), part2),
        0.5,
    );
}

fn add(comptime T: type, a: T, b: T) T {
    return switch (@typeInfo(T)) {
        .float => a + b,
        .@"struct" => a.add(b),
        else => @compileError(
            "expected a float or vector type, got `" ++
                @typeName(T) ++ "`",
        ),
    };
}

fn sub(comptime T: type, a: T, b: T) T {
    return switch (@typeInfo(T)) {
        .float => a - b,
        .@"struct" => a.sub(b),
        else => @compileError(
            "expected a float or vector type, got `" ++
                @typeName(T) ++ "`",
        ),
    };
}

fn scale(comptime T: type, value: T, factor: FactorOf(T)) T {
    return switch (@typeInfo(T)) {
        .float => value * factor,
        .@"struct" => value.scale(factor),
        else => @compileError(
            "expected a float or vector type, got `" ++
                @typeName(T) ++ "`",
        ),
    };
}

fn clamp01(comptime T: type, value: T) T {
    return @min(@max(value, 0), 1);
}

fn assertFloat(comptime T: type) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "expected a float type, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

test "lerp and steps" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 5),
        lerp(f32, 0, 10, 0.5),
        1e-6,
    );
    try std.testing.expectEqual(@as(f32, 0), step(f32, 1, 0.5));
    try std.testing.expectEqual(@as(f32, 1), step(f32, 1, 1));
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        smoothStep(f32, 0, 1, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        smootherStep(f32, 0, 1, 0.5),
        1e-6,
    );
    try std.testing.expectEqual(
        @as(f32, 0),
        smoothStep(f32, 0, 1, -2),
    );

    const Vec3f = @import("../linear/vector.zig").Vec3(f32);
    const a = Vec3f.fromElements(.{ 0, 0, 0 });
    const b = Vec3f.fromElements(.{ 10, 20, 30 });
    const middle = lerp(Vec3f, a, b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), middle.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10), middle.at(1), 1e-6);
}

test "quadratic bezier" {
    const Vec2f = @import("../linear/vector.zig").Vec2(f32);
    const start = Vec2f.fromElements(.{ 0, 0 });
    const control = Vec2f.fromElements(.{ 1, 2 });
    const end = Vec2f.fromElements(.{ 2, 0 });
    const middle = quadraticBezier(
        Vec2f,
        start,
        control,
        end,
        0.5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), middle.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), middle.at(1), 1e-6);
    try std.testing.expectEqual(
        start,
        quadraticBezier(Vec2f, start, control, end, 0),
    );
    try std.testing.expectEqual(
        end,
        quadraticBezier(Vec2f, start, control, end, 1),
    );

    const tangent = quadraticBezierTangent(
        Vec2f,
        start,
        control,
        end,
        0,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        tangent.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 4),
        tangent.at(1),
        1e-6,
    );
}

test "cubic bezier" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        cubicBezier(f32, 0, 1, 2, 3, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        cubicBezierTangent(f32, 0, 1, 2, 3, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        cubicBezierTangent(f32, 0, 1, 2, 3, 1),
        1e-6,
    );
}

test "catmull rom" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        catmullRom(f32, 0, 1, 2, 3, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        catmullRom(f32, 0, 1, 2, 3, 1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        catmullRom(f32, 0, 1, 2, 3, 0.5),
        1e-6,
    );

    const Vec2f = @import("../linear/vector.zig").Vec2(f32);
    const points = [_]Vec2f{
        Vec2f.fromElements(.{ 0, 0 }),
        Vec2f.fromElements(.{ 1, 1 }),
        Vec2f.fromElements(.{ 2, 0 }),
        Vec2f.fromElements(.{ 3, 1 }),
    };
    const midpoint = catmullRom(
        Vec2f,
        points[0],
        points[1],
        points[2],
        points[3],
        0.5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        midpoint.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        midpoint.at(1),
        1e-6,
    );
    const tangent = catmullRomTangent(
        Vec2f,
        points[0],
        points[1],
        points[2],
        points[3],
        0.5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        tangent.at(0),
        1e-6,
    );
}
