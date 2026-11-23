//! Scalar comparisons and special-value checks.
//!
//! Each helper takes the component type as its first
//! parameter, so arguments and results are concrete.

const std = @import("std");

/// Compares two floats by absolute difference.
///
/// Equal values compare equal, including signed
/// zeros and matching infinities. Any NaN makes the
/// result `false`.
///
/// # Args
/// - `T`: a float type.
/// - `a`: first value.
/// - `b`: second value.
/// - `epsilon_value`: non-negative tolerance.
///
/// # Return
/// `true` when `abs(a - b) <= epsilon_value`.
///
/// # Note
/// Prefer `approxEqRel` for values far from zero.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn approxEqAbs(
    comptime T: type,
    a: T,
    b: T,
    epsilon_value: T,
) bool {
    assertFloat(T);
    if (a == b) return true;
    if (isNan(T, a) or isNan(T, b)) return false;
    return @abs(a - b) <= epsilon_value;
}

/// Compares two floats by relative difference.
///
/// The tolerance is scaled by the larger magnitude,
/// so `epsilon_value` behaves like a number of
/// significant digits.
///
/// # Args
/// - `T`: a float type.
/// - `a`: first value.
/// - `b`: second value.
/// - `epsilon_value`: positive tolerance.
///
/// # Return
/// `true` when
/// `abs(a - b) <= epsilon_value * max(abs(a), abs(b))`.
///
/// # Note
/// Near zero the result is unreliable; combine it
/// with `approxEqAbs` through `approxEq`.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn approxEqRel(
    comptime T: type,
    a: T,
    b: T,
    epsilon_value: T,
) bool {
    assertFloat(T);
    if (a == b) return true;
    if (isNan(T, a) or isNan(T, b)) return false;
    const scaled = @max(@abs(a), @abs(b)) * epsilon_value;
    return @abs(a - b) <= scaled;
}

/// Compares two floats by absolute or relative
/// difference.
///
/// # Args
/// - `T`: a float type.
/// - `a`: first value.
/// - `b`: second value.
/// - `epsilon_value`: positive tolerance for both
///   checks.
///
/// # Return
/// `true` when either `approxEqAbs` or `approxEqRel`
/// succeeds.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn approxEq(
    comptime T: type,
    a: T,
    b: T,
    epsilon_value: T,
) bool {
    return approxEqAbs(T, a, b, epsilon_value) or
        approxEqRel(T, a, b, epsilon_value);
}

/// Tests whether a value is close to zero.
///
/// # Args
/// - `T`: a number type.
/// - `value`: the value to test.
/// - `epsilon_value`: non-negative tolerance.
///
/// # Return
/// `true` when `abs(value) <= epsilon_value`.
///
/// # Compile Error
/// - `T` is not a number type.
pub fn isZero(
    comptime T: type,
    value: T,
    epsilon_value: T,
) bool {
    assertNumber(T);
    return @abs(value) <= epsilon_value;
}

/// Tests whether a float can be normalized.
///
/// # Args
/// - `T`: a float type.
/// - `value`: the value to test.
/// - `epsilon_value`: smallest accepted magnitude.
///
/// # Return
/// `true` when `abs(value)` is greater than
/// `epsilon_value`.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn isNormalizable(
    comptime T: type,
    value: T,
    epsilon_value: T,
) bool {
    assertFloat(T);
    return @abs(value) > epsilon_value;
}

/// Reports whether a value is NaN.
///
/// # Args
/// - `T`: a float type.
/// - `value`: the value to test.
///
/// # Return
/// `true` when the value is NaN.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn isNan(comptime T: type, value: T) bool {
    assertFloat(T);
    return value != value;
}

/// Reports whether a value is an infinity.
///
/// # Args
/// - `T`: a float type.
/// - `value`: the value to test.
///
/// # Return
/// `true` when the value is `+inf` or `-inf`.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn isInf(comptime T: type, value: T) bool {
    assertFloat(T);
    return @abs(value) == std.math.inf(T);
}

/// Reports whether a value is finite.
///
/// # Args
/// - `T`: a number type.
/// - `value`: the value to test.
///
/// # Return
/// `false` for NaN and infinities; integers are
/// always finite.
///
/// # Compile Error
/// - `T` is not a number type.
pub fn isFinite(comptime T: type, value: T) bool {
    assertNumber(T);
    switch (@typeInfo(T)) {
        .float => return !isNan(T, value) and !isInf(T, value),
        else => return true,
    }
}

fn assertNumber(comptime T: type) void {
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError(
            "expected a number type, got `" ++
                @typeName(T) ++ "`",
        ),
    }
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

test "approximate comparison" {
    const eps = 1e-5;
    try std.testing.expect(approxEqAbs(f32, 1, 1 + eps * 0.5, eps));
    try std.testing.expect(!approxEqAbs(f32, 1, 1.1, eps));
    try std.testing.expect(approxEqRel(f32, 1e6, 1e6 + 1, eps));
    try std.testing.expect(approxEq(f32, 0, 1e-8, 1e-6));
    try std.testing.expect(!approxEq(f32, 0, 1e-3, 1e-6));
}

test "zero and normalizable" {
    try std.testing.expect(isZero(f32, 1e-9, 1e-6));
    try std.testing.expect(!isZero(f32, 1e-3, 1e-6));
    try std.testing.expect(isZero(i32, 0, 0));
    try std.testing.expect(isNormalizable(f32, 0.5, 1e-6));
    try std.testing.expect(!isNormalizable(f32, 0, 1e-6));
}

test "special values" {
    try std.testing.expect(isNan(f32, std.math.nan(f32)));
    try std.testing.expect(!isNan(f32, 1));
    try std.testing.expect(isInf(f32, std.math.inf(f32)));
    try std.testing.expect(isInf(f32, -std.math.inf(f32)));
    try std.testing.expect(!isFinite(f32, std.math.nan(f32)));
    try std.testing.expect(!isFinite(f32, std.math.inf(f32)));
    try std.testing.expect(isFinite(f32, 1));
    try std.testing.expect(isFinite(i32, 7));
}
