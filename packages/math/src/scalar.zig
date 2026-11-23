//! Scalar comparisons and tolerance helpers.
//!
//! Nothing here assumes a tolerance: every function
//! that compares floating point values takes
//! `epsilon` from the caller. `Epsilon` only lists
//! suggested constants for callers that need one.
//!
//! The helpers accept scalars and vectors. A scalar
//! `epsilon` applies to every component.

const std = @import("std");
const fixed = @import("fixed.zig");

pub const Fixed = fixed.Fixed;

/// Suggested tolerances grouped by precision.
///
/// The constants are suggestions only. No function
/// in this library silently uses one; pass an
/// `epsilon` explicitly.
pub const Epsilon = struct {
    /// Coarse absolute tolerance for `f16`.
    pub const half_coarse: f16 = 1.0 / 1024.0;
    /// Fine absolute tolerance for `f16`.
    pub const half_fine: f16 = 1.0 / 32768.0;
    /// Coarse absolute tolerance for `f32`.
    pub const single_coarse: f32 = 1.0e-5;
    /// Fine absolute tolerance for `f32`.
    pub const single_fine: f32 = 1.0e-7;
    /// Coarse absolute tolerance for `f64`.
    pub const double_coarse: f64 = 1.0e-11;
    /// Fine absolute tolerance for `f64`.
    pub const double_fine: f64 = 1.0e-14;
};

/// Returns the component type of `T`.
///
/// For a vector this is the element type; for a
/// matrix or quaternion type it is the declared
/// `Component`. Other types are their own component
/// type.
///
/// # Args
/// - `T`: a scalar, vector, matrix, or other type.
///
/// # Return
/// The scalar stored by `T`.
pub fn ScalarOf(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .vector => |info| info.child,
        .@"struct" => if (@hasDecl(T, "Component")) T.Component else T,
        else => T,
    };
}

/// Compares two values by absolute difference.
///
/// Equal values compare equal, including signed
/// zeros and matching infinities. Any NaN makes the
/// result `false`.
///
/// # Args
/// - `a`: first value; a float or float vector.
/// - `b`: second value; must coerce to `@TypeOf(a)`.
/// - `epsilon`: non-negative scalar tolerance.
///
/// # Return
/// `true` when every component satisfies
/// `abs(a - b) <= epsilon`.
///
/// # Note
/// Prefer `approxEqRel` for values far from zero.
///
/// # Compile Error
/// - `a` is not a float or a vector of floats.
/// - `b` or `epsilon` does not coerce to `a`.
pub fn approxEqAbs(a: anytype, b: anytype, epsilon: anytype) bool {
    const T = @TypeOf(a);
    assertFloat(T);
    const other: T = b;
    if (allTrue(T, a == other)) return true;
    if (isNan(a) or isNan(other)) return false;
    return allTrue(T, @abs(a - other) <= broadcast(T, epsilon));
}

/// Compares two values by relative difference.
///
/// The tolerance is scaled by the larger magnitude,
/// so `epsilon` behaves like a number of significant
/// digits. Any NaN makes the result `false`.
///
/// # Args
/// - `a`: first value; a float or float vector.
/// - `b`: second value; must coerce to `@TypeOf(a)`.
/// - `epsilon`: positive scalar tolerance.
///
/// # Return
/// `true` when every component satisfies
/// `abs(a - b) <= epsilon * max(abs(a), abs(b))`.
///
/// # Note
/// Near zero the check is unreliable; combine it
/// with `approxEqAbs` through `approxEq`.
///
/// # Compile Error
/// - `a` is not a float or a vector of floats.
/// - `b` or `epsilon` does not coerce to `a`.
pub fn approxEqRel(a: anytype, b: anytype, epsilon: anytype) bool {
    const T = @TypeOf(a);
    assertFloat(T);
    const other: T = b;
    if (allTrue(T, a == other)) return true;
    if (isNan(a) or isNan(other)) return false;
    const scaled = @max(@abs(a), @abs(other)) * broadcast(T, epsilon);
    return allTrue(T, @abs(a - other) <= scaled);
}

/// Compares two values by absolute or relative
/// difference.
///
/// # Args
/// - `a`: first value; a float or float vector.
/// - `b`: second value; must coerce to `@TypeOf(a)`.
/// - `epsilon`: positive tolerance used for both
///   checks; a scalar.
///
/// # Return
/// `true` when `approxEqAbs` or `approxEqRel`
/// succeeds for every component.
///
/// # Note
/// A single `epsilon` serves both scales, so pass a
/// value that suits the magnitudes in use.
///
/// # Compile Error
/// - `a` is not a float or a vector of floats.
pub fn approxEq(a: anytype, b: anytype, epsilon: anytype) bool {
    return approxEqAbs(a, b, epsilon) or
        approxEqRel(a, b, epsilon);
}

/// Tests whether a value is close to zero.
///
/// # Args
/// - `value`: a number or a vector of numbers.
/// - `epsilon`: non-negative scalar tolerance.
///
/// # Return
/// `true` when every component satisfies
/// `abs(value) <= epsilon`.
///
/// # Note
/// A negative `epsilon` makes every test fail.
///
/// # Compile Error
/// - `value` is not a number or vector.
pub fn isZero(value: anytype, epsilon: anytype) bool {
    const T = @TypeOf(value);
    assertNumber(T);
    return allTrue(T, @abs(value) <= broadcast(T, epsilon));
}

/// Tests whether a value can be normalized.
///
/// # Args
/// - `value`: a float or float vector.
/// - `epsilon`: smallest accepted length, as a
///   scalar.
///
/// # Return
/// `true` when the length of `value` is greater
/// than `epsilon`.
///
/// # Note
/// `value` is kept as is; nothing is normalized.
///
/// # Compile Error
/// - `value` is not a float or a vector of floats.
pub fn isNormalizable(value: anytype, epsilon: anytype) bool {
    const T = @TypeOf(value);
    assertFloat(T);
    const squared = switch (@typeInfo(T)) {
        .vector => @reduce(.Add, value * value),
        else => value * value,
    };
    return @sqrt(squared) > epsilon;
}

/// Reports whether any component is NaN.
///
/// # Args
/// - `value`: a float or float vector.
///
/// # Return
/// `true` when at least one component is NaN.
///
/// # Compile Error
/// - `value` is not a float or a vector of floats.
pub fn isNan(value: anytype) bool {
    const T = @TypeOf(value);
    assertFloat(T);
    return anyTrue(T, value != value);
}

/// Reports whether any component is an infinity.
///
/// # Args
/// - `value`: a float or float vector.
///
/// # Return
/// `true` when at least one component is `+inf` or
/// `-inf`.
///
/// # Compile Error
/// - `value` is not a float or a vector of floats.
pub fn isInf(value: anytype) bool {
    const T = @TypeOf(value);
    assertFloat(T);
    const inf_value = std.math.inf(ScalarOf(T));
    return anyTrue(T, @abs(value) == broadcast(T, inf_value));
}

/// Reports whether every component is finite.
///
/// # Args
/// - `value`: a number or a vector of numbers.
///
/// # Return
/// `true` when no component is NaN or an infinity.
///
/// # Note
/// Integer values are always finite.
///
/// # Compile Error
/// - `value` is not a number or vector.
pub fn isFinite(value: anytype) bool {
    const T = @TypeOf(value);
    assertNumber(T);
    switch (@typeInfo(ScalarOf(T))) {
        .float => return !isNan(value) and !isInf(value),
        else => return true,
    }
}

/// Broadcasts `value` to `T`.
fn broadcast(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => @splat(value),
        else => value,
    };
}

/// Reduces a scalar or vector boolean to one flag.
fn allTrue(comptime T: type, value: anytype) bool {
    return switch (@typeInfo(T)) {
        .vector => @reduce(.And, value),
        else => value,
    };
}

/// Reports whether any scalar or vector lane is
/// true.
fn anyTrue(comptime T: type, value: anytype) bool {
    return switch (@typeInfo(T)) {
        .vector => @reduce(.Or, value),
        else => value,
    };
}

fn assertNumber(comptime T: type) void {
    switch (@typeInfo(ScalarOf(T))) {
        .int, .float => {},
        else => @compileError(
            "expected a number or a vector of numbers, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

fn assertFloat(comptime T: type) void {
    switch (@typeInfo(ScalarOf(T))) {
        .float => {},
        else => @compileError(
            "expected a float or a vector of floats, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

test "approxEq" {
    const single = Epsilon.single_coarse;
    try std.testing.expect(approxEqAbs(@as(f32, 1), 1 + single / 2, single));
    try std.testing.expect(!approxEqAbs(@as(f32, 1), 1.1, single));
    try std.testing.expect(approxEqRel(@as(f32, 1e6), 1e6 + 1, 1e-5));
    try std.testing.expect(approxEq(@as(f32, 0), 1e-8, 1e-6));
    try std.testing.expect(!approxEq(@as(f32, 0), 1e-3, 1e-6));

    const a: @Vector(2, f32) = .{ 1, 1 };
    const b: @Vector(2, f32) = .{ 1 + single / 2, 1 };
    try std.testing.expect(approxEq(a, b, single));
    try std.testing.expect(!approxEqRel(a, b, 1e-9));
}

test "isZero" {
    try std.testing.expect(isZero(@as(f32, 1e-9), 1e-6));
    try std.testing.expect(!isZero(@as(f32, 1e-3), 1e-6));
    try std.testing.expect(isZero(@Vector(3, f32){ 0, -0.0, 1e-9 }, 1e-6));
    try std.testing.expect(!isZero(@Vector(3, f32){ 0, 1, 0 }, 1e-6));
    try std.testing.expect(isZero(@as(i32, 0), 0));
}

test "isNormalizable" {
    try std.testing.expect(isNormalizable(@Vector(3, f32){ 1, 0, 0 }, 1e-6));
    try std.testing.expect(!isNormalizable(@Vector(3, f32){ 0, 0, 0 }, 1e-6));
    try std.testing.expect(isNormalizable(@as(f32, 0.5), 1e-6));
    try std.testing.expect(!isNormalizable(@as(f32, 0.0), 1e-6));
}

test "nanAndInf" {
    const nan_value = std.math.nan(f32);
    const inf_value = std.math.inf(f32);
    try std.testing.expect(isNan(nan_value));
    try std.testing.expect(!isNan(@as(f32, 1)));
    try std.testing.expect(isNan(@Vector(2, f32){ 0, nan_value }));
    try std.testing.expect(isInf(inf_value));
    try std.testing.expect(isInf(-inf_value));
    try std.testing.expect(!isInf(nan_value));
    try std.testing.expect(isInf(@Vector(2, f32){ 0, -inf_value }));
    try std.testing.expect(!isFinite(inf_value));
    try std.testing.expect(!isFinite(nan_value));
    try std.testing.expect(isFinite(@Vector(2, f32){ 1, 2 }));
    try std.testing.expect(isFinite(@as(i32, 7)));
}

test "ScalarOf" {
    try std.testing.expectEqual(f32, ScalarOf(@Vector(3, f32)));
    try std.testing.expectEqual(i64, ScalarOf(i64));
}

test "Fixed" {
    const fixed_value = Fixed(i32, 16).fromFloat(1.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), fixed_value.toFloat(), 1e-6);
    try std.testing.expectEqual(@as(i32, 1), fixed_value.toInt());
}
