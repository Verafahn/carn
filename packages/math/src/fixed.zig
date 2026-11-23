//! Fixed-point numbers stored in integers.
//!
//! `Fixed` keeps the fractional precision explicit in
//! the type, so no rounding rule is hidden in the
//! storage. Conversions and truncation are documented
//! on each operation.

const std = @import("std");

/// A binary fixed-point number with `fractional_bits`
/// fractional bits.
///
/// The raw integer is scaled by `2^fractional_bits`,
/// so `Fixed(i32, 16)` stores `1.5` as `98304`.
///
/// # Args
/// - `T`: the signed or unsigned integer storage.
/// - `fractional_bits`: number of fractional bits;
///   must be smaller than the width of `T`.
///
/// # Compile Error
/// - `T` is not an integer type.
/// - `fractional_bits` is not smaller than the
///   width of `T`.
pub fn Fixed(comptime T: type, comptime fractional_bits: u6) type {
    const info = switch (@typeInfo(T)) {
        .int => |int_info| int_info,
        else => @compileError(
            "Fixed: `T` must be an integer type, got `" ++
                @typeName(T) ++ "`",
        ),
    };
    if (fractional_bits >= info.bits) {
        @compileError(
            "Fixed: `fractional_bits` must be smaller than " ++
                "the width of `T`",
        );
    }

    return struct {
        /// Raw value scaled by `2^fractional_bits`.
        raw: T,

        const Self = @This();
        const Wide = @Int(
            info.signedness,
            info.bits * 2,
        );
        const scale: T = @as(T, 1) << fractional_bits;

        /// Wraps an already scaled integer.
        ///
        /// # Args
        /// - `raw`: the scaled storage value.
        ///
        /// # Return
        /// A fixed-point value backed by `raw`.
        pub fn fromRaw(raw: T) Self {
            return .{ .raw = raw };
        }

        /// Converts an integer to fixed point.
        ///
        /// # Args
        /// - `value`: an integer value.
        ///
        /// # Return
        /// `value` scaled by `2^fractional_bits`.
        ///
        /// # Panic
        /// `value` does not fit the storage type.
        pub fn fromInt(value: anytype) Self {
            return .{ .raw = @as(T, @intCast(value)) * scale };
        }

        /// Converts a float to fixed point.
        ///
        /// # Args
        /// - `value`: a float value.
        ///
        /// # Return
        /// `value` truncated toward zero.
        ///
        /// # Note
        /// Values outside the representable range are
        /// not clamped.
        pub fn fromFloat(value: anytype) Self {
            const scaled = value *
                @as(@TypeOf(value), @floatFromInt(scale));
            return .{ .raw = @intFromFloat(scaled) };
        }

        /// Converts the value to `f32`.
        ///
        /// # Return
        /// The stored value as a single precision
        /// float.
        pub fn toFloat(self: Self) f32 {
            return @as(f32, @floatFromInt(self.raw)) /
                @as(f32, @floatFromInt(scale));
        }

        /// Truncates the value toward zero.
        ///
        /// # Return
        /// The integer part of the value.
        pub fn toInt(self: Self) T {
            return @divTrunc(self.raw, scale);
        }

        /// Adds two fixed-point values.
        ///
        /// # Args
        /// - `a`: first operand.
        /// - `b`: second operand.
        ///
        /// # Return
        /// The sum, truncated to the storage width.
        pub fn add(a: Self, b: Self) Self {
            return .{ .raw = a.raw +% b.raw };
        }

        /// Subtracts `b` from `a`.
        ///
        /// # Args
        /// - `a`: first operand.
        /// - `b`: operand to subtract.
        ///
        /// # Return
        /// The difference, truncated to the storage
        /// width.
        pub fn subtract(a: Self, b: Self) Self {
            return .{ .raw = a.raw -% b.raw };
        }

        /// Negates a fixed-point value.
        ///
        /// # Args
        /// - `value`: the value to negate.
        ///
        /// # Return
        /// The negated value.
        pub fn negate(value: Self) Self {
            return .{ .raw = -%value.raw };
        }

        /// Multiplies two fixed-point values.
        ///
        /// # Args
        /// - `a`: first operand.
        /// - `b`: second operand.
        ///
        /// # Return
        /// The product, truncated toward zero.
        ///
        /// # Panic
        /// The result does not fit the storage type.
        pub fn multiply(a: Self, b: Self) Self {
            const product = @as(Wide, a.raw) * @as(Wide, b.raw);
            return .{
                .raw = @intCast(@divTrunc(product, @as(Wide, scale))),
            };
        }

        /// Divides `a` by `b`.
        ///
        /// # Args
        /// - `a`: dividend.
        /// - `b`: divisor.
        ///
        /// # Return
        /// The quotient, truncated toward zero.
        ///
        /// # Panic
        /// `b.raw` is zero.
        pub fn divide(a: Self, b: Self) Self {
            const numerator = @as(Wide, a.raw) * @as(Wide, scale);
            return .{
                .raw = @intCast(@divTrunc(numerator, @as(Wide, b.raw))),
            };
        }

        /// Compares two values by raw storage.
        ///
        /// # Args
        /// - `a`: first operand.
        /// - `b`: second operand.
        ///
        /// # Return
        /// `true` when both values are bit identical.
        pub fn eql(a: Self, b: Self) bool {
            return a.raw == b.raw;
        }
    };
}

test "Fixed" {
    const Fixed16 = Fixed(i32, 16);
    const one = Fixed16.fromInt(1);
    try std.testing.expectEqual(@as(i32, 65536), one.raw);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        Fixed16.fromFloat(1.5).toFloat(),
        1e-6,
    );
    try std.testing.expectEqual(
        @as(i32, 1),
        Fixed16.fromFloat(1.75).toInt(),
    );
}

test "Fixed arithmetic" {
    const Fixed16 = Fixed(i32, 16);
    const half = Fixed16.fromFloat(0.5);
    const quarter = Fixed16.fromFloat(0.25);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.75),
        Fixed16.add(half, quarter).toFloat(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        Fixed16.subtract(half, quarter).toFloat(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -0.5),
        Fixed16.negate(half).toFloat(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3),
        Fixed16.multiply(Fixed16.fromInt(2), Fixed16.fromFloat(1.5))
            .toFloat(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        Fixed16.divide(Fixed16.fromInt(3), Fixed16.fromFloat(1.5))
            .toFloat(),
        1e-6,
    );
    try std.testing.expect(Fixed16.eql(half, Fixed16.fromFloat(0.5)));
}

test "Fixed unsigned" {
    const UFixed = Fixed(u16, 8);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.5),
        UFixed.fromFloat(1.5).toFloat(),
        1e-3,
    );
}
