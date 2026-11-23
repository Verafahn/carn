//! Uniform sampling and low-discrepancy sequences.
//!
//! The samplers are pure functions of their input:
//! no generator is stored or advanced, so the caller
//! owns the random source. Inputs are expected in
//! `0..1`; out-of-range values are not clamped.

const std = @import("std");
const vec = @import("vec.zig");
const Vec2f = vec.Vec2(f32);

/// Samples the unit disk uniformly.
///
/// # Args
/// - `u`: two uniform values in `0..1`.
///
/// # Return
/// A point inside the unit disk.
///
/// # Note
/// The radius uses `sqrt(u[0])` so area density
/// stays uniform.
pub fn uniformDisk(u: anytype) @TypeOf(u) {
    const radius = @sqrt(u[0]);
    const angle = 2.0 * std.math.pi * u[1];
    const point: @TypeOf(u) = .{
        radius * @cos(angle),
        radius * @sin(angle),
    };
    return point;
}

/// Samples the unit sphere uniformly.
///
/// # Args
/// - `u`: two uniform values in `0..1`.
///
/// # Return
/// A point on the unit sphere.
pub fn uniformSphere(u: anytype) @Vector(3, @TypeOf(u[0])) {
    const T = @TypeOf(u[0]);
    const z = 1 - 2 * u[0];
    const radius = @sqrt(@max(0, 1 - z * z));
    const angle = 2.0 * std.math.pi * u[1];
    const point: @Vector(3, T) = .{
        radius * @cos(angle),
        radius * @sin(angle),
        z,
    };
    return point;
}

/// Samples the `+Z` unit hemisphere uniformly.
///
/// # Args
/// - `u`: two uniform values in `0..1`.
///
/// # Return
/// A point on the upper hemisphere.
pub fn uniformHemisphere(u: anytype) @Vector(3, @TypeOf(u[0])) {
    const T = @TypeOf(u[0]);
    const z = u[0];
    const radius = @sqrt(@max(0, 1 - z * z));
    const angle = 2.0 * std.math.pi * u[1];
    const point: @Vector(3, T) = .{
        radius * @cos(angle),
        radius * @sin(angle),
        z,
    };
    return point;
}

/// Samples the `+Z` hemisphere with a cosine
/// weight.
///
/// # Args
/// - `u`: two uniform values in `0..1`.
///
/// # Return
/// A unit vector whose density is proportional to
/// its `z` component.
pub fn cosineHemisphere(u: anytype) @Vector(3, @TypeOf(u[0])) {
    const T = @TypeOf(u[0]);
    const z = @sqrt(u[0]);
    const radius = @sqrt(@max(0, 1 - u[0]));
    const angle = 2.0 * std.math.pi * u[1];
    const point: @Vector(3, T) = .{
        radius * @cos(angle),
        radius * @sin(angle),
        z,
    };
    return point;
}

/// Computes the Halton radical inverse.
///
/// # Args
/// - `index`: sequence position; zero returns zero.
/// - `base`: number base; must be at least two.
///
/// # Return
/// The radical inverse in `0..1`.
///
/// # Note
/// Each call recomputes the value, so the sequence
/// is stateless.
///
/// # Panic
/// `base` is less than two.
pub fn halton(index: usize, base: u32) f32 {
    std.debug.assert(base >= 2);
    var result: f32 = 0;
    var fraction: f32 = 1;
    var remaining = index;
    while (remaining > 0) {
        fraction /= @as(f32, @floatFromInt(base));
        const digit = remaining % base;
        result += fraction * @as(f32, @floatFromInt(digit));
        remaining /= base;
    }
    return result;
}

/// Computes a two-dimensional Halton point.
///
/// # Args
/// - `index`: sequence position.
///
/// # Return
/// The bases 2 and 3 radical inverses.
pub fn halton2(index: usize) Vec2f {
    return .{ halton(index, 2), halton(index, 3) };
}

/// Computes a two-dimensional Sobol point.
///
/// # Args
/// - `index`: sequence position; must fit 32 bits.
///
/// # Return
/// The first two Sobol dimensions.
///
/// # Note
/// Direction numbers come from the standard
/// primitive polynomials for dimensions one and
/// two, and the point is recomputed on every call.
///
/// # Panic
/// `index` does not fit in 32 bits.
pub fn sobol2(index: usize) Vec2f {
    std.debug.assert(index < (1 << 32));
    const gray = index ^ (index >> 1);
    var x1: u32 = 0;
    var x2: u32 = 0;
    var direction1: u32 = 1 << 31;
    var direction2: u32 = 1 << 31;
    var bit: usize = 0;
    while (bit < 32) : (bit += 1) {
        const mask = @as(u32, 1) << @as(u5, @intCast(bit));
        if (gray & mask != 0) {
            x1 ^= direction1;
            x2 ^= direction2;
        }
        direction1 >>= 1;
        direction2 ^= direction2 >> 1;
    }
    return .{ toUnit(x1), toUnit(x2) };
}

fn toUnit(value: u32) f32 {
    return @as(f32, @floatFromInt(value)) /
        4294967296.0;
}

test "disk sampling" {
    const center = uniformDisk(@Vector(2, f32){ 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), center[0], 1e-6);
    const edge = uniformDisk(@Vector(2, f32){ 1, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), edge[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), edge[1], 1e-6);
    const quarter = uniformDisk(@Vector(2, f32){ 0.25, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), quarter[0], 1e-6);

    const sample = uniformDisk(@Vector(2, f32){ 0.6, 0.3 });
    try std.testing.expect(vec.length(sample) <= 1);
}

test "sphere sampling" {
    const inputs = [_]@Vector(2, f32){
        .{ 0, 0 },
        .{ 0.25, 0.5 },
        .{ 0.5, 0.25 },
        .{ 1, 1 },
    };
    for (inputs) |u| {
        const point = uniformSphere(u);
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            vec.length(point),
            1e-5,
        );
        const upper = uniformHemisphere(u);
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            vec.length(upper),
            1e-5,
        );
        try std.testing.expect(upper[2] >= 0);
        const weighted = cosineHemisphere(u);
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            vec.length(weighted),
            1e-5,
        );
        try std.testing.expect(weighted[2] >= 0);
    }
    const pole = uniformSphere(@Vector(2, f32){ 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), pole[2], 1e-6);
    const bottom = uniformSphere(@Vector(2, f32){ 1, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, -1), bottom[2], 1e-6);
}

test "halton" {
    try std.testing.expectEqual(@as(f32, 0), halton(0, 2));
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        halton(1, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        halton(2, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.75),
        halton(3, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 3.0),
        halton(1, 3),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 9.0),
        halton(3, 3),
        1e-6,
    );
    const point = halton2(1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), point[0], 1e-6);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 3.0),
        point[1],
        1e-6,
    );
}

test "sobol" {
    const expected = [_]@Vector(2, f32){
        .{ 0, 0 },
        .{ 0.5, 0.5 },
        .{ 0.75, 0.25 },
        .{ 0.25, 0.75 },
        .{ 0.375, 0.375 },
    };
    for (expected, 0..) |want, index| {
        const point = sobol2(index);
        try std.testing.expectApproxEqAbs(want[0], point[0], 1e-6);
        try std.testing.expectApproxEqAbs(want[1], point[1], 1e-6);
    }
}
