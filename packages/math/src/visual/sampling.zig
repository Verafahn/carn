//! Uniform sampling and low-discrepancy sequences.
//!
//! The samplers are pure functions: no generator is
//! stored or advanced, so the caller owns the random
//! source. Inputs are expected in `0..1`; out-of-range
//! values are not clamped.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// Sampling routines for `T`.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Sampling(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Sampling: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Samples the unit disk uniformly.
        ///
        /// # Args
        /// - `u`: two uniform values in `0..1`.
        ///
        /// # Return
        /// A point inside the unit disk.
        ///
        /// # Note
        /// The radius uses `sqrt(u[0])`, which keeps
        /// the area density uniform.
        pub fn uniformDisk(
            u: vector.Vector(T, 2),
        ) vector.Vector(T, 2) {
            const radius = @sqrt(u.at(0));
            const angle = 2 * std.math.pi * u.at(1);
            return vector.Vector(T, 2).fromElements(.{
                radius * @cos(angle),
                radius * @sin(angle),
            });
        }

        /// Samples the unit sphere uniformly.
        ///
        /// # Args
        /// - `u`: two uniform values in `0..1`.
        ///
        /// # Return
        /// A point on the unit sphere.
        pub fn uniformSphere(
            u: vector.Vector(T, 2),
        ) vector.Vector(T, 3) {
            const z = 1 - 2 * u.at(0);
            const radius = @sqrt(@max(0, 1 - z * z));
            const angle = 2 * std.math.pi * u.at(1);
            return vector.Vector(T, 3).fromElements(.{
                radius * @cos(angle),
                radius * @sin(angle),
                z,
            });
        }

        /// Samples the `+Z` hemisphere uniformly.
        ///
        /// # Args
        /// - `u`: two uniform values in `0..1`.
        ///
        /// # Return
        /// A point on the upper hemisphere.
        pub fn uniformHemisphere(
            u: vector.Vector(T, 2),
        ) vector.Vector(T, 3) {
            const z = u.at(0);
            const radius = @sqrt(@max(0, 1 - z * z));
            const angle = 2 * std.math.pi * u.at(1);
            return vector.Vector(T, 3).fromElements(.{
                radius * @cos(angle),
                radius * @sin(angle),
                z,
            });
        }

        /// Samples the `+Z` hemisphere with a cosine
        /// weight.
        ///
        /// # Args
        /// - `u`: two uniform values in `0..1`.
        ///
        /// # Return
        /// A unit vector whose density is
        /// proportional to its `z` component.
        pub fn cosineHemisphere(
            u: vector.Vector(T, 2),
        ) vector.Vector(T, 3) {
            const z = @sqrt(u.at(0));
            const radius = @sqrt(@max(0, 1 - u.at(0)));
            const angle = 2 * std.math.pi * u.at(1);
            return vector.Vector(T, 3).fromElements(.{
                radius * @cos(angle),
                radius * @sin(angle),
                z,
            });
        }

        /// Computes the Halton radical inverse.
        ///
        /// # Args
        /// - `index`: sequence position; zero yields
        ///   zero.
        /// - `base_value`: number base; at least two.
        ///
        /// # Return
        /// The radical inverse in `0..1`.
        ///
        /// # Note
        /// Each call recomputes the value, so the
        /// sequence is stateless.
        ///
        /// # Panic
        /// In safety-checked builds, `base_value` is
        /// less than two.
        pub fn halton(index: usize, base_value: u32) T {
            std.debug.assert(base_value >= 2);
            var result: T = 0;
            var fraction: T = 1;
            var remaining = index;
            while (remaining > 0) {
                fraction /= @as(T, @floatFromInt(base_value));
                const digit = remaining % base_value;
                result += fraction *
                    @as(T, @floatFromInt(digit));
                remaining /= base_value;
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
        pub fn halton2(index: usize) vector.Vector(T, 2) {
            return vector.Vector(T, 2).fromElements(.{
                halton(index, 2),
                halton(index, 3),
            });
        }

        /// Computes a two-dimensional Sobol point.
        ///
        /// # Args
        /// - `index`: sequence position; it must fit
        ///   in 32 bits.
        ///
        /// # Return
        /// The first two Sobol dimensions.
        ///
        /// # Note
        /// Direction numbers come from the standard
        /// primitive polynomials for dimensions one and
        /// two; the point is recomputed on every call.
        ///
        /// # Panic
        /// In safety-checked builds, `index` does not
        /// fit in 32 bits.
        pub fn sobol2(index: usize) vector.Vector(T, 2) {
            std.debug.assert(index < (1 << 32));
            const gray = index ^ (index >> 1);
            var x1: u32 = 0;
            var x2: u32 = 0;
            var direction1: u32 = 1 << 31;
            var direction2: u32 = 1 << 31;
            var bit: usize = 0;
            while (bit < 32) : (bit += 1) {
                const mask = @as(u32, 1) <<
                    @as(u5, @intCast(bit));
                if (gray & mask != 0) {
                    x1 ^= direction1;
                    x2 ^= direction2;
                }
                direction1 >>= 1;
                direction2 ^= direction2 >> 1;
            }
            return vector.Vector(T, 2).fromElements(.{
                toUnit(x1),
                toUnit(x2),
            });
        }

        fn toUnit(value: u32) T {
            return @as(T, @floatFromInt(value)) /
                4294967296.0;
        }
    };
}

test "disk sampling" {
    const Vec2f = vector.Vec2(f32);
    const center = Sampling(f32).uniformDisk(
        Vec2f.fromElements(.{ 0, 0 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0), center.at(0), 1e-6);
    const edge = Sampling(f32).uniformDisk(
        Vec2f.fromElements(.{ 1, 0 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), edge.at(0), 1e-6);
    const quarter = Sampling(f32).uniformDisk(
        Vec2f.fromElements(.{ 0.25, 0 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        quarter.at(0),
        1e-6,
    );
    const sample = Sampling(f32).uniformDisk(
        Vec2f.fromElements(.{ 0.6, 0.3 }),
    );
    try std.testing.expect(sample.length() <= 1);
}

test "sphere sampling" {
    const Vec2f = vector.Vec2(f32);
    const inputs = [_]Vec2f{
        Vec2f.fromElements(.{ 0, 0 }),
        Vec2f.fromElements(.{ 0.25, 0.5 }),
        Vec2f.fromElements(.{ 0.5, 0.25 }),
        Vec2f.fromElements(.{ 1, 1 }),
    };
    for (inputs) |u| {
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            Sampling(f32).uniformSphere(u).length(),
            1e-5,
        );
        const upper = Sampling(f32).uniformHemisphere(u);
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            upper.length(),
            1e-5,
        );
        try std.testing.expect(upper.at(2) >= 0);
        const weighted = Sampling(f32).cosineHemisphere(u);
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            weighted.length(),
            1e-5,
        );
        try std.testing.expect(weighted.at(2) >= 0);
    }
    const pole = Sampling(f32).uniformSphere(
        Vec2f.fromElements(.{ 0, 0 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), pole.at(2), 1e-6);
}

test "halton" {
    try std.testing.expectEqual(@as(f32, 0), Sampling(f32).halton(0, 2));
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        Sampling(f32).halton(1, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        Sampling(f32).halton(2, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.75),
        Sampling(f32).halton(3, 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 3.0),
        Sampling(f32).halton(1, 3),
        1e-6,
    );
    const point = Sampling(f32).halton2(1);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        point.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0 / 3.0),
        point.at(1),
        1e-6,
    );
}

test "sobol" {
    const Vec2f = vector.Vec2(f32);
    const expected = [_]Vec2f{
        Vec2f.fromElements(.{ 0, 0 }),
        Vec2f.fromElements(.{ 0.5, 0.5 }),
        Vec2f.fromElements(.{ 0.75, 0.25 }),
        Vec2f.fromElements(.{ 0.25, 0.75 }),
        Vec2f.fromElements(.{ 0.375, 0.375 }),
    };
    for (expected, 0..) |want, index| {
        const point = Sampling(f32).sobol2(index);
        try std.testing.expectApproxEqAbs(
            want.at(0),
            point.at(0),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            want.at(1),
            point.at(1),
            1e-6,
        );
    }
}
