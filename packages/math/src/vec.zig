//! Vector types and geometric operations.
//!
//! A vector is a row matrix, represented as
//! `@Vector(N, T)`. Because the storage is a builtin
//! vector, addition, subtraction, negation, and
//! componentwise multiplication and division use the
//! normal Zig operators. This module adds the
//! geometric helpers the language does not provide.
//!
//! Multiplying a row vector by a matrix lives in
//! `mat.mulVec`, not here.

const std = @import("std");
const scalar = @import("scalar.zig");

/// Returns a vector type with `N` components of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
/// - `N`: the component count; must be at least 1.
///
/// # Return
/// `@Vector(N, T)`, the row matrix storage.
///
/// # Compile Error
/// - `N` is zero.
/// - `T` is not an integer or float type.
pub fn Vector(comptime T: type, comptime N: usize) type {
    if (N == 0) {
        @compileError("Vector: `N` must be at least 1");
    }
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError(
            "Vector: `T` must be an integer or float, got `" ++
                @typeName(T) ++ "`",
        ),
    }
    return @Vector(N, T);
}

/// Returns a two-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// `@Vector(2, T)`.
pub fn Vec2(comptime T: type) type {
    return Vector(T, 2);
}

/// Returns a three-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// `@Vector(3, T)`.
pub fn Vec3(comptime T: type) type {
    return Vector(T, 3);
}

/// Returns a four-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// `@Vector(4, T)`.
pub fn Vec4(comptime T: type) type {
    return Vector(T, 4);
}

/// Half-precision two-component vector.
pub const Half2 = Vec2(f16);
/// Half-precision three-component vector.
pub const Half3 = Vec3(f16);
/// Half-precision four-component vector.
pub const Half4 = Vec4(f16);
/// Signed 32-bit two-component vector.
pub const IVec2 = Vec2(i32);
/// Signed 32-bit three-component vector.
pub const IVec3 = Vec3(i32);
/// Signed 32-bit four-component vector.
pub const IVec4 = Vec4(i32);
/// Unsigned 32-bit two-component vector.
pub const UVec2 = Vec2(u32);
/// Unsigned 32-bit three-component vector.
pub const UVec3 = Vec3(u32);
/// Unsigned 32-bit four-component vector.
pub const UVec4 = Vec4(u32);

/// Builds a vector by replicating one scalar.
///
/// # Args
/// - `T`: the result vector type.
/// - `value`: the scalar copied into every
///   component.
///
/// # Return
/// A vector with `value` in every lane.
///
/// # Compile Error
/// - `T` is not a vector type.
pub fn splat(comptime T: type, value: anytype) T {
    assertVector(T);
    return @splat(value);
}

/// Computes the dot product of two vectors.
///
/// # Args
/// - `a`: first vector.
/// - `b`: second vector with the same type as `a`.
///
/// # Return
/// The sum of the componentwise products.
///
/// # Compile Error
/// - `a` is not a vector.
/// - `b` does not have the type of `a`.
pub fn dot(a: anytype, b: anytype) scalar.ScalarOf(@TypeOf(a)) {
    const T = @TypeOf(a);
    assertVector(T);
    const other: T = b;
    return @reduce(.Add, a * other);
}

/// Computes the three-dimensional cross product.
///
/// # Args
/// - `a`: first three-component vector.
/// - `b`: second three-component vector.
///
/// # Return
/// `a x b`, using a right-handed basis.
///
/// # Compile Error
/// - `a` is not a three-component vector.
/// - `b` does not have the type of `a`.
pub fn cross(a: anytype, b: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    switch (@typeInfo(T)) {
        .vector => |info| {
            if (info.len != 3) {
                @compileError(
                    "cross: operands must have 3 " ++
                        "components",
                );
            }
        },
        else => @compileError("cross: operands must be vectors"),
    }
    const other: T = b;
    return .{
        a[1] * other[2] - a[2] * other[1],
        a[2] * other[0] - a[0] * other[2],
        a[0] * other[1] - a[1] * other[0],
    };
}

/// Returns the squared length of a vector.
///
/// # Args
/// - `value`: an integer or float vector.
///
/// # Return
/// The sum of the squared components.
///
/// # Compile Error
/// - `value` is not a vector.
pub fn lengthSquared(value: anytype) scalar.ScalarOf(@TypeOf(value)) {
    assertVector(@TypeOf(value));
    return @reduce(.Add, value * value);
}

/// Returns the length of a vector.
///
/// # Args
/// - `value`: a float vector.
///
/// # Return
/// The Euclidean length.
///
/// # Compile Error
/// - `value` is not a vector of floats.
pub fn length(value: anytype) scalar.ScalarOf(@TypeOf(value)) {
    const T = @TypeOf(value);
    assertFloatVector(T);
    return @sqrt(lengthSquared(value));
}

/// Normalizes a vector in place of the input copy.
///
/// # Args
/// - `value`: a float vector.
///
/// # Return
/// A unit vector with the direction of `value`.
///
/// # Note
/// The result is not finite when `value` is zero or
/// infinite. Use `normalizeSafe` when the input may
/// be degenerate.
///
/// # Compile Error
/// - `value` is not a vector of floats.
pub fn normalize(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    assertFloatVector(T);
    const len = length(value);
    return value / @as(T, @splat(len));
}

/// Normalizes a vector unless it is too short.
///
/// # Args
/// - `value`: a float vector.
/// - `epsilon`: smallest length that is normalized.
///
/// # Return
/// A unit vector, or a zero vector when the length
/// is at most `epsilon`.
///
/// # Compile Error
/// - `value` is not a vector of floats.
pub fn normalizeSafe(value: anytype, epsilon: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    assertFloatVector(T);
    if (!scalar.isNormalizable(value, epsilon)) {
        return @as(T, @splat(0));
    }
    return normalize(value);
}

/// Returns the squared distance between two points.
///
/// # Args
/// - `a`: first point.
/// - `b`: second point with the type of `a`.
///
/// # Return
/// The squared length of `a - b`.
///
/// # Compile Error
/// - `a` is not a vector.
/// - `b` does not have the type of `a`.
pub fn distanceSquared(a: anytype, b: anytype) scalar.ScalarOf(@TypeOf(a)) {
    const T = @TypeOf(a);
    assertVector(T);
    const other: T = b;
    return lengthSquared(a - other);
}

/// Returns the distance between two points.
///
/// # Args
/// - `a`: first point; a float vector.
/// - `b`: second point with the type of `a`.
///
/// # Return
/// The Euclidean distance between `a` and `b`.
///
/// # Compile Error
/// - `a` is not a vector of floats.
pub fn distance(a: anytype, b: anytype) scalar.ScalarOf(@TypeOf(a)) {
    const T = @TypeOf(a);
    assertFloatVector(T);
    const other: T = b;
    return length(a - other);
}

/// Reflects an incident vector about a normal.
///
/// # Args
/// - `incident`: the incoming direction.
/// - `normal`: the unit surface normal.
///
/// # Return
/// `incident - 2 * dot(incident, normal) * normal`.
///
/// # Note
/// `normal` should be a unit vector; the result is
/// scaled by its length otherwise.
///
/// # Compile Error
/// - The operands are not vectors of the same type.
pub fn reflect(incident: anytype, normal: anytype) @TypeOf(incident) {
    const T = @TypeOf(incident);
    assertVector(T);
    const unit: T = normal;
    return incident -
        unit * @as(T, @splat(2 * dot(incident, unit)));
}

/// Refracts an incident vector through a surface.
///
/// # Args
/// - `incident`: the incoming unit direction.
/// - `normal`: the unit surface normal, opposed to
///   `incident`.
/// - `eta`: ratio of the incident and transmitted
///   indices of refraction.
///
/// # Return
/// The refracted direction, or `null` on total
/// internal reflection.
///
/// # Note
/// Both directions must be unit vectors.
///
/// # Compile Error
/// - The operands are not vectors of the same type.
pub fn refract(
    incident: anytype,
    normal: anytype,
    eta: anytype,
) ?@TypeOf(incident) {
    const T = @TypeOf(incident);
    assertVector(T);
    assertFloatVector(T);
    const unit: T = normal;
    const cos_i = dot(unit, incident);
    const k = 1 - eta * eta * (1 - cos_i * cos_i);
    if (k < 0) return null;
    const factor = eta * cos_i + @sqrt(k);
    return incident * @as(T, @splat(eta)) -
        unit * @as(T, @splat(factor));
}

/// Rearranges components by a compile-time mask.
///
/// # Args
/// - `value`: source vector.
/// - `mask`: component indices; an array or tuple
///   whose entries are in `0..N`.
///
/// # Return
/// A vector whose lane `i` is `value[mask[i]]`.
///
/// # Note
/// The mask length may differ from the source
/// length, so the result may have any component
/// count.
///
/// # Compile Error
/// - `mask` is not an array or tuple.
/// - A mask entry is negative or out of range.
pub fn swizzle(value: anytype, comptime mask: anytype) Vector(
    scalar.ScalarOf(@TypeOf(value)),
    componentCount(@TypeOf(mask)),
) {
    const T = @TypeOf(value);
    const C = scalar.ScalarOf(T);
    const count = comptime componentCount(@TypeOf(mask));
    const source_count = switch (@typeInfo(T)) {
        .vector => |info| info.len,
        else => @compileError("swizzle: `value` must be a vector"),
    };
    const indices = comptime blk: {
        var result: [count]i32 = undefined;
        var next: usize = 0;
        for (mask) |index| {
            if (index < 0 or index >= source_count) {
                @compileError(
                    "swizzle: mask entry out of range",
                );
            }
            result[next] = index;
            next += 1;
        }
        break :blk result;
    };
    return @shuffle(C, value, @as(T, undefined), indices);
}

/// Clamps every component into a range.
///
/// # Args
/// - `value`: a number or a vector.
/// - `lower`: minimum; a scalar or a vector.
/// - `upper`: maximum; a scalar or a vector.
///
/// # Return
/// The componentwise clamp of `value`.
///
/// # Note
/// Scalar bounds are broadcast to every component.
///
/// # Compile Error
/// - An operand cannot be coerced to `value`.
pub fn clamp(value: anytype, lower: anytype, upper: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    return @min(
        @max(value, coerceLike(T, lower)),
        coerceLike(T, upper),
    );
}

/// Clamps every component into `0..1`.
///
/// # Args
/// - `value`: a number or a vector.
///
/// # Return
/// The componentwise clamp of `value` to zero and
/// one.
pub fn saturate(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    return clamp(value, coerceLike(T, 0), coerceLike(T, 1));
}

/// Returns the component count of a vector or
/// sequence type.
fn componentCount(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .vector => |info| info.len,
        .array => |info| info.len,
        .@"struct" => |info| info.field_types.len,
        else => @compileError(
            "expected a vector, array, or tuple, got `" ++
                @typeName(T) ++ "`",
        ),
    };
}

fn coerceLike(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => switch (@typeInfo(@TypeOf(value))) {
            .vector => value,
            else => @splat(value),
        },
        else => value,
    };
}

fn assertVector(comptime T: type) void {
    switch (@typeInfo(T)) {
        .vector => {},
        else => @compileError(
            "expected a vector, got `" ++ @typeName(T) ++ "`",
        ),
    }
}

fn assertFloatVector(comptime T: type) void {
    switch (@typeInfo(T)) {
        .vector => |info| switch (@typeInfo(info.child)) {
            .float => {},
            else => @compileError(
                "expected a vector of floats, got `" ++
                    @typeName(T) ++ "`",
            ),
        },
        else => @compileError(
            "expected a vector of floats, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

test "vector aliases" {
    try std.testing.expectEqual(@Vector(2, f16), Half2);
    try std.testing.expectEqual(@Vector(4, i32), IVec4);
    try std.testing.expectEqual(@Vector(3, u32), UVec3);
    try std.testing.expectEqual(@Vector(2, f64), Vec2(f64));
}

test "splat" {
    const value = splat(@Vector(3, f32), 2.5);
    try std.testing.expectEqual(@as(f32, 2.5), value[0]);
    try std.testing.expectEqual(@as(f32, 2.5), value[2]);
}

test "dot and cross" {
    const a: Vec3(f32) = .{ 1, 2, 3 };
    const b: Vec3(f32) = .{ 4, 5, 6 };
    try std.testing.expectEqual(@as(f32, 32), dot(a, b));
    const unit_x: Vec3(f32) = .{ 1, 0, 0 };
    const unit_y: Vec3(f32) = .{ 0, 1, 0 };
    const result = cross(unit_x, unit_y);
    try std.testing.expectEqual(@as(f32, 0), result[0]);
    try std.testing.expectEqual(@as(f32, 0), result[1]);
    try std.testing.expectEqual(@as(f32, 1), result[2]);
    try std.testing.expectEqual(@as(f32, 0), dot(result, unit_y));
}

test "length and distance" {
    const a: Vec3(f32) = .{ 3, 4, 0 };
    try std.testing.expectEqual(@as(f32, 25), lengthSquared(a));
    try std.testing.expectEqual(@as(f32, 5), length(a));
    const b: Vec3(f32) = .{ 0, 0, 0 };
    try std.testing.expectEqual(@as(f32, 5), distance(a, b));
    try std.testing.expectEqual(@as(f32, 25), distanceSquared(a, b));
}

test "normalize" {
    const value: Vec3(f32) = .{ 0, 3, 4 };
    const unit = normalize(value);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(unit), 1e-6);
    const degenerate: Vec3(f32) = .{ 0, 0, 0 };
    const safe = normalizeSafe(degenerate, 1e-6);
    try std.testing.expectEqual(@as(f32, 0), safe[0]);
    const kept = normalizeSafe(value, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(kept), 1e-6);
}

test "reflect and refract" {
    const down: Vec3(f32) = .{ 1, -1, 0 };
    const up: Vec3(f32) = .{ 0, 1, 0 };
    const reflected = reflect(normalize(down), up);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.70710678),
        reflected[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.70710678),
        reflected[1],
        1e-6,
    );

    const incident: Vec3(f32) = .{ 0, -1, 0 };
    const refracted = refract(incident, up, 0.5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        length(refracted),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1),
        refracted[1],
        1e-6,
    );

    const slanted = normalize(Vec3(f32){ 1, -1, 0 });
    const bent = refract(slanted, up, 0.5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        length(bent),
        1e-6,
    );
    try std.testing.expect(bent[0] > 0);
    try std.testing.expect(bent[1] < 0);

    const total: Vec3(f32) = .{ 1, 0, 0 };
    try std.testing.expect(refract(total, up, 2) == null);
}

test "swizzle" {
    const value: Vec4(f32) = .{ 10, 20, 30, 40 };
    const xy = swizzle(value, .{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 10), xy[0]);
    try std.testing.expectEqual(@as(f32, 20), xy[1]);
    const zz = swizzle(value, [_]i32{ 2, 2, 2 });
    try std.testing.expectEqual(@as(f32, 30), zz[1]);
    const reversed = swizzle(value, .{ 3, 1 });
    try std.testing.expectEqual(@as(f32, 40), reversed[0]);
    try std.testing.expectEqual(@as(f32, 20), reversed[1]);
}

test "clampAndSaturate" {
    const value: Vec3(f32) = .{ -1, 0.5, 3 };
    const clamped = clamp(value, 0, 1);
    try std.testing.expectEqual(@as(f32, 0), clamped[0]);
    try std.testing.expectEqual(@as(f32, 0.5), clamped[1]);
    try std.testing.expectEqual(@as(f32, 1), clamped[2]);
    const saturated = saturate(value);
    try std.testing.expectEqual(@as(f32, 1), saturated[2]);
    try std.testing.expectEqual(@as(f32, 0), saturate(@as(f32, -2)));
    try std.testing.expectEqual(@as(f32, 1), saturate(@as(f32, 2)));
}
