//! Quaternions and their rotation helpers.
//!
//! A quaternion stores a rotation; multiplying two
//! values composes them. `multiply(a, b)` applies `b`
//! first and `a` second to a row vector, which matches
//! the matrix composition in `matrix`.
//!
//! Angle arguments are in radians.

const std = @import("std");
const base = @import("../base/root.zig");
const vector = @import("vector.zig");
const matrix = @import("matrix.zig");
const basis = @import("basis.zig");

/// Returns a quaternion type with components of `T`.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Return
/// A struct with a vector part and a scalar part.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Quat(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Quat: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Vector part (imaginary components).
        v: vector.Vector(T, 3),
        /// Scalar part (real component).
        w: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;
        /// The vector-part type.
        pub const VectorPart = vector.Vector(T, 3);

        /// Returns the identity rotation.
        ///
        /// # Return
        /// A unit quaternion with no rotation.
        pub fn identity() Self {
            return .{ .v = VectorPart.zero(), .w = 1 };
        }

        /// Builds a rotation from an axis and an angle.
        ///
        /// # Args
        /// - `axis`: unit rotation axis.
        /// - `angle`: rotation angle in radians.
        ///
        /// # Return
        /// The matching unit quaternion.
        ///
        /// # Note
        /// The axis is used as given; pass a unit
        /// vector.
        pub fn fromAxisAngle(
            axis: VectorPart,
            angle: T,
        ) Self {
            const half = angle * 0.5;
            return .{
                .v = axis.scale(@sin(half)),
                .w = @cos(half),
            };
        }

        /// Builds the shortest rotation between two
        /// vectors.
        ///
        /// # Args
        /// - `from`: starting direction; non-zero.
        /// - `to`: target direction; non-zero.
        /// - `epsilon_value`: smallest accepted length.
        ///
        /// # Return
        /// The shortest rotation, or `null` when either
        /// direction is too short.
        ///
        /// # Note
        /// Opposite directions have no unique shortest
        /// rotation; any perpendicular axis is used.
        pub fn fromVectors(
            from: VectorPart,
            to: VectorPart,
            epsilon_value: T,
        ) ?Self {
            if (!from.isNormalizable(epsilon_value) or
                !to.isNormalizable(epsilon_value))
            {
                return null;
            }
            const a = from.normalize();
            const b = to.normalize();
            const cos_theta = a.dot(b);
            const raw: Self = .{
                .v = a.cross(b),
                .w = 1 + cos_theta,
            };
            if (raw.v.isNormalizable(epsilon_value) or
                !base.compare.isZero(
                    T,
                    raw.w,
                    epsilon_value,
                ))
            {
                return raw.normalizeSafe(epsilon_value);
            }
            const frame = basis.Basis(T)
                .fromNormal(a, epsilon_value) orelse return null;
            return .{ .v = frame.tangent, .w = 0 };
        }

        /// Converts a rotation matrix to a quaternion.
        ///
        /// # Args
        /// - `m`: a `3x3` row-vector rotation matrix.
        ///
        /// # Return
        /// The matching unit quaternion.
        ///
        /// # Note
        /// The matrix must be a pure rotation; scaling
        /// or shear yields an approximate result.
        pub fn fromMat3(m: matrix.Mat3(T)) Self {
            const trace = m.at(0, 0) + m.at(1, 1) +
                m.at(2, 2);
            if (trace > 0) {
                const s = @sqrt(trace + 1) * 2;
                return .{
                    .v = VectorPart.fromElements(.{
                        (m.at(1, 2) - m.at(2, 1)) / s,
                        (m.at(2, 0) - m.at(0, 2)) / s,
                        (m.at(0, 1) - m.at(1, 0)) / s,
                    }),
                    .w = 0.25 * s,
                };
            }
            if (m.at(0, 0) > m.at(1, 1) and
                m.at(0, 0) > m.at(2, 2))
            {
                const s = @sqrt(1 + m.at(0, 0) -
                    m.at(1, 1) - m.at(2, 2)) * 2;
                return .{
                    .v = VectorPart.fromElements(.{
                        0.25 * s,
                        (m.at(0, 1) + m.at(1, 0)) / s,
                        (m.at(0, 2) + m.at(2, 0)) / s,
                    }),
                    .w = (m.at(1, 2) - m.at(2, 1)) / s,
                };
            }
            if (m.at(1, 1) > m.at(2, 2)) {
                const s = @sqrt(1 + m.at(1, 1) -
                    m.at(0, 0) - m.at(2, 2)) * 2;
                return .{
                    .v = VectorPart.fromElements(.{
                        (m.at(0, 1) + m.at(1, 0)) / s,
                        0.25 * s,
                        (m.at(1, 2) + m.at(2, 1)) / s,
                    }),
                    .w = (m.at(0, 2) - m.at(2, 0)) / s,
                };
            }
            const s = @sqrt(1 + m.at(2, 2) -
                m.at(0, 0) - m.at(1, 1)) * 2;
            return .{
                .v = VectorPart.fromElements(.{
                    (m.at(0, 2) + m.at(2, 0)) / s,
                    (m.at(1, 2) + m.at(2, 1)) / s,
                    0.25 * s,
                }),
                .w = (m.at(1, 0) - m.at(0, 1)) / s,
            };
        }

        /// Multiplies two quaternions.
        ///
        /// # Args
        /// - `other`: rotation applied first.
        ///
        /// # Return
        /// The Hamilton product.
        ///
        /// # Note
        /// The order matches row-vector matrices: the
        /// result applies `other` then `self`.
        pub fn multiply(self: Self, other: Self) Self {
            const cross = self.v.cross(other.v);
            return .{
                .v = self.v.scale(other.w)
                    .add(other.v.scale(self.w))
                    .add(cross),
                .w = self.w * other.w - self.v.dot(other.v),
            };
        }

        /// Returns the conjugate.
        ///
        /// # Return
        /// A quaternion with a negated vector part.
        ///
        /// # Note
        /// For unit quaternions this is the inverse.
        pub fn conjugate(self: Self) Self {
            return .{ .v = self.v.negate(), .w = self.w };
        }

        /// Returns the multiplicative inverse.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted squared
        ///   length.
        ///
        /// # Return
        /// The inverse, or `null` when the squared
        /// length is at most `epsilon_value`.
        ///
        /// # Note
        /// The result divides by the squared length,
        /// so non-unit inputs are supported.
        pub fn inverse(self: Self, epsilon_value: T) ?Self {
            const squared = self.lengthSquared();
            if (base.compare.isZero(T, squared, epsilon_value)) {
                return null;
            }
            const factor = 1 / squared;
            return .{
                .v = self.v.scale(-factor),
                .w = self.w * factor,
            };
        }

        /// Computes the quaternion dot product.
        ///
        /// # Args
        /// - `other`: the quaternion to multiply with.
        ///
        /// # Return
        /// The sum of the componentwise products.
        pub fn dot(self: Self, other: Self) T {
            return self.v.dot(other.v) + self.w * other.w;
        }

        /// Returns the squared magnitude.
        ///
        /// # Return
        /// `dot(self, self)`.
        pub fn lengthSquared(self: Self) T {
            return self.v.lengthSquared() + self.w * self.w;
        }

        /// Returns the magnitude.
        ///
        /// # Return
        /// The Euclidean length of the components.
        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        /// Normalizes the quaternion.
        ///
        /// # Return
        /// A unit quaternion with the same rotation.
        ///
        /// # Note
        /// The result is not finite for a zero input;
        /// use `normalizeSafe` when that is possible.
        pub fn normalize(self: Self) Self {
            const factor = 1 / self.length();
            return .{
                .v = self.v.scale(factor),
                .w = self.w * factor,
            };
        }

        /// Normalizes a quaternion unless it is too
        /// short.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted length.
        ///
        /// # Return
        /// A unit quaternion, or the identity when the
        /// length is at most `epsilon_value`.
        pub fn normalizeSafe(
            self: Self,
            epsilon_value: T,
        ) Self {
            if (self.length() <= epsilon_value) {
                return Self.identity();
            }
            return self.normalize();
        }

        /// Rotates a vector.
        ///
        /// # Args
        /// - `value`: the vector to rotate.
        ///
        /// # Return
        /// The rotated vector.
        ///
        /// # Note
        /// The formula assumes a unit quaternion.
        pub fn rotateVector(
            self: Self,
            value: VectorPart,
        ) VectorPart {
            const uv = self.v.cross(value);
            return value.add(
                uv.scale(self.w)
                    .add(self.v.cross(uv))
                    .scale(2),
            );
        }

        /// Converts the quaternion to a rotation matrix.
        ///
        /// # Return
        /// A `3x3` row-vector rotation matrix.
        ///
        /// # Note
        /// The formula assumes a unit quaternion.
        pub fn toMat3(self: Self) matrix.Mat3(T) {
            const x = self.v.at(0);
            const y = self.v.at(1);
            const z = self.v.at(2);
            const w = self.w;
            const xx = x * x;
            const yy = y * y;
            const zz = z * z;
            const xy = x * y;
            const xz = x * z;
            const yz = y * z;
            const xw = x * w;
            const yw = y * w;
            const zw = z * w;
            return matrix.Mat3(T).fromColumns(.{
                .{
                    1 - 2 * (yy + zz),
                    2 * (xy - zw),
                    2 * (xz + yw),
                },
                .{
                    2 * (xy + zw),
                    1 - 2 * (xx + zz),
                    2 * (yz - xw),
                },
                .{
                    2 * (xz - yw),
                    2 * (yz + xw),
                    1 - 2 * (xx + yy),
                },
            });
        }

        /// Interpolates spherically.
        ///
        /// # Args
        /// - `other`: end rotation; a unit quaternion.
        /// - `t`: interpolation factor.
        ///
        /// # Return
        /// The interpolated unit quaternion along the
        /// shortest arc.
        ///
        /// # Note
        /// `t` is not clamped. Parallel inputs return
        /// `self`; zero-length inputs are unsupported.
        /// The inverse cosine may not lower on SPIR-V.
        pub fn slerp(self: Self, other: Self, t: T) Self {
            var target = other;
            var cos_theta = self.dot(target);
            if (cos_theta < 0) {
                target = target.negate();
                cos_theta = -cos_theta;
            }
            if (cos_theta >= 1) return self;

            const angle = std.math.acos(@min(cos_theta, 1));
            const sin_angle = @sin(angle);
            if (sin_angle == 0) return self;

            const w0 = @sin((1 - t) * angle) / sin_angle;
            const w1 = @sin(t * angle) / sin_angle;
            return .{
                .v = self.v.scale(w0).add(target.v.scale(w1)),
                .w = self.w * w0 + target.w * w1,
            };
        }

        /// Interpolates linearly and normalizes.
        ///
        /// # Args
        /// - `other`: end rotation; a unit quaternion.
        /// - `t`: interpolation factor.
        ///
        /// # Return
        /// The normalized blend along the shortest arc.
        ///
        /// # Note
        /// Faster than `slerp` but not constant speed.
        /// `t` is not clamped.
        pub fn nlerp(self: Self, other: Self, t: T) Self {
            var target = other;
            if (self.dot(target) < 0) {
                target = target.negate();
            }
            const complement = 1 - t;
            const blended: Self = .{
                .v = self.v.scale(complement)
                    .add(target.v.scale(t)),
                .w = self.w * complement + target.w * t,
            };
            return blended.normalize();
        }

        /// Negates the quaternion.
        ///
        /// # Return
        /// The negated value, which describes the same
        /// rotation.
        pub fn negate(self: Self) Self {
            return .{ .v = self.v.negate(), .w = -self.w };
        }
    };
}

test "quat basics" {
    const Vec3f = vector.Vec3(f32);
    const identity = Quat(f32).identity();
    try std.testing.expectEqual(@as(f32, 1), identity.w);
    try std.testing.expectEqual(@as(f32, 0), identity.v.at(0));

    const rotation = Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 0, 0, 1 }),
        @as(f32, std.math.pi) * 0.5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        rotation.length(),
        1e-6,
    );
    const backward = rotation.inverse(1e-6).?;
    const product = rotation.multiply(backward);
    try std.testing.expectApproxEqAbs(@as(f32, 1), product.w, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), product.v.at(0), 1e-6);
    try std.testing.expect(
        (Quat(f32){ .v = Vec3f.zero(), .w = 0 })
            .inverse(1e-6) == null,
    );
}

test "quat rotate matches matrix" {
    const Vec3f = vector.Vec3(f32);
    const axis = Vec3f.fromElements(.{ 1, 2, 3 }).normalize();
    const rotation = Quat(f32).fromAxisAngle(axis, 0.7);
    const m = rotation.toMat3();
    const point = Vec3f.fromElements(.{ 1, -2, 0.5 });
    const by_quat = rotation.rotateVector(point);
    const by_matrix = point.mulMatrix(3, m);
    try std.testing.expectApproxEqAbs(
        by_matrix.at(0),
        by_quat.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        by_matrix.at(2),
        by_quat.at(2),
        1e-6,
    );
}

test "quat matrix round trip" {
    const Vec3f = vector.Vec3(f32);
    const cases = [_]Vec3f{
        Vec3f.zero(),
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 0, 1, 0 }),
        Vec3f.fromElements(.{ 0.3, -1.2, 0.7 }),
    };
    for (cases) |angles| {
        const rotation = Quat(f32)
            .fromAxisAngle(
                Vec3f.fromElements(.{ 0, 0, 1 }),
                angles.at(2),
            )
            .multiply(
            Quat(f32)
                .fromAxisAngle(
                    Vec3f.fromElements(.{ 0, 1, 0 }),
                    angles.at(1),
                )
                .multiply(
                Quat(f32).fromAxisAngle(
                    Vec3f.fromElements(.{ 1, 0, 0 }),
                    angles.at(0),
                ),
            ),
        );
        const first = rotation.toMat3();
        const second = Quat(f32).fromMat3(first).toMat3();
        inline for (0..3) |c| {
            inline for (0..3) |r| {
                try std.testing.expectApproxEqAbs(
                    first.at(r, c),
                    second.at(r, c),
                    1e-5,
                );
            }
        }
    }
}

test "quat interpolation" {
    const Vec3f = vector.Vec3(f32);
    const start = Quat(f32).identity();
    const end = Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 0, 0, 1 }),
        @as(f32, std.math.pi) * 0.5,
    );
    const middle = start.slerp(end, 0.5);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        middle.length(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @cos((@as(f32, std.math.pi) * 0.5) * 0.25),
        middle.w,
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        start.nlerp(end, 0.5).length(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        start.nlerp(start.negate(), 0.5).length(),
        1e-6,
    );
}

test "quat from vectors" {
    const Vec3f = vector.Vec3(f32);
    const from = Vec3f.fromElements(.{ 0, 0, 1 });
    const to = Vec3f.fromElements(.{ 1, 0, 0 });
    const rotation = Quat(f32).fromVectors(from, to, 1e-6).?;
    const rotated = rotation.rotateVector(from);
    try std.testing.expectApproxEqAbs(to.at(0), rotated.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(to.at(1), rotated.at(1), 1e-6);
    try std.testing.expectApproxEqAbs(to.at(2), rotated.at(2), 1e-6);

    const opposite = Quat(f32).fromVectors(
        from,
        from.negate(),
        1e-6,
    ).?;
    const flipped = opposite.rotateVector(from);
    try std.testing.expectApproxEqAbs(-1, flipped.at(2), 1e-6);

    try std.testing.expect(
        Quat(f32).fromVectors(Vec3f.zero(), to, 1e-6) == null,
    );
}
