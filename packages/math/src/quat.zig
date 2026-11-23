//! Quaternions and their rotation helpers.
//!
//! A quaternion stores rotations so that multiplying
//! two values composes them. The multiplication order
//! matches `mat` composition: `multiply(a, b)` applies
//! `b` first and `a` second to a row vector.
//!
//! Angle arguments are in radians.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const basis = @import("basis.zig");
const Vec3 = vec.Vec3;

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
        v: @Vector(3, T),
        /// Scalar part (real component).
        w: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;
        /// The vector-part type.
        pub const Vec = @Vector(3, T);

        /// Returns the identity rotation.
        ///
        /// # Return
        /// A unit quaternion with no rotation.
        pub fn identity() Self {
            return .{ .v = @splat(0), .w = 1 };
        }
    };
}

/// Multiplies two quaternions.
///
/// # Args
/// - `a`: rotation applied second.
/// - `b`: rotation applied first.
///
/// # Return
/// The Hamilton product `a * b`.
///
/// # Note
/// The order matches matrix multiplication for row
/// vectors: the result applies `b` then `a`.
///
/// # Compile Error
/// - The operands are not the same quaternion type.
pub fn multiply(a: anytype, b: anytype) @TypeOf(a) {
    const Q = @TypeOf(a);
    const other: Q = b;
    const av = a.v;
    const bv = other.v;
    return .{
        .v = av * @as(Q.Vec, @splat(other.w)) +
            bv * @as(Q.Vec, @splat(a.w)) +
            vec.cross(av, bv),
        .w = a.w * other.w - vec.dot(av, bv),
    };
}

/// Returns the conjugate of a quaternion.
///
/// # Args
/// - `q`: the quaternion to conjugate.
///
/// # Return
/// A quaternion with a negated vector part.
///
/// # Note
/// For unit quaternions the conjugate is the
/// inverse rotation.
pub fn conjugate(q: anytype) @TypeOf(q) {
    return .{ .v = -q.v, .w = q.w };
}

/// Returns the multiplicative inverse.
///
/// # Args
/// - `q`: the quaternion to invert.
/// - `epsilon`: smallest accepted squared length.
///
/// # Return
/// The inverse, or `null` when the squared length
/// is at most `epsilon`.
///
/// # Note
/// The result divides by the squared length, so it
/// is not restricted to unit quaternions.
pub fn inverse(q: anytype, epsilon: anytype) ?@TypeOf(q) {
    const Q = @TypeOf(q);
    const squared = lengthSquared(q);
    if (scalar.isZero(squared, epsilon)) return null;
    const factor = 1 / squared;
    return .{
        .v = -q.v * @as(Q.Vec, @splat(factor)),
        .w = q.w * factor,
    };
}

/// Computes the quaternion dot product.
///
/// # Args
/// - `a`: first quaternion.
/// - `b`: second quaternion of the same type.
///
/// # Return
/// The sum of the componentwise products.
pub fn dot(a: anytype, b: anytype) @TypeOf(a.v[0]) {
    const Q = @TypeOf(a);
    const other: Q = b;
    return vec.dot(a.v, other.v) + a.w * other.w;
}

/// Returns the squared magnitude.
///
/// # Args
/// - `q`: the quaternion to measure.
///
/// # Return
/// `dot(q, q)`.
pub fn lengthSquared(q: anytype) @TypeOf(q.v[0]) {
    return vec.lengthSquared(q.v) + q.w * q.w;
}

/// Returns the magnitude.
///
/// # Args
/// - `q`: the quaternion to measure.
///
/// # Return
/// The Euclidean length of the four components.
pub fn length(q: anytype) @TypeOf(q.v[0]) {
    return @sqrt(lengthSquared(q));
}

/// Normalizes a quaternion.
///
/// # Args
/// - `q`: the quaternion to normalize.
///
/// # Return
/// A unit quaternion with the same rotation.
///
/// # Note
/// The result is not finite for a zero input. Use
/// `normalizeSafe` when that is possible.
pub fn normalize(q: anytype) @TypeOf(q) {
    const Q = @TypeOf(q);
    const factor = 1 / length(q);
    return .{
        .v = q.v * @as(Q.Vec, @splat(factor)),
        .w = q.w * factor,
    };
}

/// Normalizes a quaternion unless it is too short.
///
/// # Args
/// - `q`: the quaternion to normalize.
/// - `epsilon`: smallest accepted length.
///
/// # Return
/// A unit quaternion, or the identity when the
/// length is at most `epsilon`.
pub fn normalizeSafe(q: anytype, epsilon: anytype) @TypeOf(q) {
    const Q = @TypeOf(q);
    if (length(q) <= epsilon) return Q.identity();
    return normalize(q);
}

/// Builds a rotation from an axis and an angle.
///
/// # Args
/// - `axis`: unit rotation axis.
/// - `angle`: rotation angle in radians.
///
/// # Return
/// A unit quaternion for the rotation.
///
/// # Note
/// The axis is used as given; pass a unit vector.
///
/// # Compile Error
/// - `axis` is not a three-component float
///   vector.
/// - `angle` does not coerce to the axis
///   component type.
pub fn fromAxisAngle(axis: anytype, angle: anytype) Quat(
    scalar.ScalarOf(@TypeOf(axis)),
) {
    const T = scalar.ScalarOf(@TypeOf(axis));
    const Q = Quat(T);
    const half = angle * 0.5;
    const sine = @sin(half);
    return .{
        .v = axis * @as(Q.Vec, @splat(sine)),
        .w = @cos(half),
    };
}

/// Builds the shortest rotation between two vectors.
///
/// # Args
/// - `from`: starting direction; non-zero.
/// - `to`: target direction; non-zero.
/// - `epsilon`: smallest accepted vector length.
///
/// # Return
/// The shortest rotation, or `null` when either
/// direction is too short.
///
/// # Note
/// Opposite directions have no unique shortest
/// rotation; any perpendicular axis is used.
///
/// # Compile Error
/// - The operands are not three-component float
///   vectors of the same type.
pub fn fromVectors(
    from: anytype,
    to: anytype,
    epsilon: anytype,
) ?Quat(scalar.ScalarOf(@TypeOf(from))) {
    const T = scalar.ScalarOf(@TypeOf(from));
    const Q = Quat(T);
    if (!scalar.isNormalizable(from, epsilon) or
        !scalar.isNormalizable(to, epsilon))
    {
        return null;
    }

    const a = vec.normalize(from);
    const b = vec.normalize(to);
    const cos_theta = vec.dot(a, b);
    const raw: Q = .{
        .v = vec.cross(a, b),
        .w = 1 + cos_theta,
    };
    if (scalar.isNormalizable(raw.v, epsilon) or
        !scalar.isZero(raw.w, epsilon))
    {
        return normalizeSafe(raw, epsilon);
    }

    const frame = basis.orthonormalBasis(a, epsilon) orelse
        return null;
    return .{ .v = frame[0], .w = 0 };
}

/// Rotates a vector around a unit quaternion.
///
/// # Args
/// - `q`: unit quaternion.
/// - `value`: vector to rotate.
///
/// # Return
/// The rotated vector.
///
/// # Note
/// The formula assumes `q` is normalized.
///
/// # Compile Error
/// - `value` is not a three-component vector.
/// - `value` does not share the quaternion
///   component type.
pub fn rotateVector(q: anytype, value: anytype) @TypeOf(value) {
    const V = @TypeOf(value);
    const u = q.v;
    const uv = vec.cross(u, value);
    return value +
        (uv * @as(V, @splat(q.w)) + vec.cross(u, uv)) *
            @as(V, @splat(2));
}

/// Converts a unit quaternion to a rotation matrix.
///
/// # Args
/// - `q`: unit quaternion.
///
/// # Return
/// A `3x3` row-vector rotation matrix.
///
/// # Note
/// The formula assumes `q` is normalized.
pub fn toMat3(q: anytype) mat.Mat3(@TypeOf(q.v[0])) {
    const x = q.v[0];
    const y = q.v[1];
    const z = q.v[2];
    const w = q.w;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const xw = x * w;
    const yw = y * w;
    const zw = z * w;
    return .{ .columns = .{
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
    } };
}

/// Converts a rotation matrix to a unit quaternion.
///
/// # Args
/// - `matrix`: a `3x3` row-vector rotation matrix.
///
/// # Return
/// The matching unit quaternion.
///
/// # Note
/// The matrix must be a pure rotation; scaling or
/// shear produces an approximate result.
///
/// # Compile Error
/// - `matrix` is not a square `3x3` matrix.
pub fn fromMat3(matrix: anytype) Quat(
    scalar.ScalarOf(@TypeOf(matrix)),
) {
    const M = @TypeOf(matrix);
    if (M.row_count != 3 or M.column_count != 3) {
        @compileError("fromMat3: `matrix` must be a 3x3 matrix");
    }
    const trace = matrix.columns[0][0] +
        matrix.columns[1][1] +
        matrix.columns[2][2];

    if (trace > 0) {
        const s = @sqrt(trace + 1) * 2;
        return .{
            .v = .{
                (matrix.columns[2][1] - matrix.columns[1][2]) / s,
                (matrix.columns[0][2] - matrix.columns[2][0]) / s,
                (matrix.columns[1][0] - matrix.columns[0][1]) / s,
            },
            .w = 0.25 * s,
        };
    }
    if (matrix.columns[0][0] > matrix.columns[1][1] and
        matrix.columns[0][0] > matrix.columns[2][2])
    {
        const s = @sqrt(1 + matrix.columns[0][0] -
            matrix.columns[1][1] - matrix.columns[2][2]) * 2;
        return .{
            .v = .{
                0.25 * s,
                (matrix.columns[0][1] + matrix.columns[1][0]) / s,
                (matrix.columns[0][2] + matrix.columns[2][0]) / s,
            },
            .w = (matrix.columns[2][1] -
                matrix.columns[1][2]) / s,
        };
    }
    if (matrix.columns[1][1] > matrix.columns[2][2]) {
        const s = @sqrt(1 + matrix.columns[1][1] -
            matrix.columns[0][0] - matrix.columns[2][2]) * 2;
        return .{
            .v = .{
                (matrix.columns[0][1] + matrix.columns[1][0]) / s,
                0.25 * s,
                (matrix.columns[1][2] + matrix.columns[2][1]) / s,
            },
            .w = (matrix.columns[0][2] -
                matrix.columns[2][0]) / s,
        };
    }
    const s = @sqrt(1 + matrix.columns[2][2] -
        matrix.columns[0][0] - matrix.columns[1][1]) * 2;
    return .{
        .v = .{
            (matrix.columns[0][2] + matrix.columns[2][0]) / s,
            (matrix.columns[1][2] + matrix.columns[2][1]) / s,
            0.25 * s,
        },
        .w = (matrix.columns[1][0] - matrix.columns[0][1]) / s,
    };
}

/// Interpolates two quaternions spherically.
///
/// # Args
/// - `a`: start rotation; a unit quaternion.
/// - `b`: end rotation; a unit quaternion.
/// - `t`: interpolation factor.
///
/// # Return
/// The interpolated unit quaternion along the
/// shortest arc.
///
/// # Note
/// `t` is not clamped. Parallel inputs return
/// `a`; zero-length inputs are not supported. The
/// inverse cosine is host-only in practice, so
/// prefer `nlerp` in shader code.
///
/// # Compile Error
/// - The operands are not the same quaternion
///   type.
pub fn slerp(a: anytype, b: anytype, t: anytype) @TypeOf(a) {
    const Q = @TypeOf(a);
    var other: Q = b;
    var cos_theta = dot(a, other);
    if (cos_theta < 0) {
        other = negate(other);
        cos_theta = -cos_theta;
    }
    if (cos_theta >= 1) return a;

    const angle = std.math.acos(@min(cos_theta, 1));
    const sin_angle = @sin(angle);
    if (sin_angle == 0) return a;

    const w0 = @sin((1 - t) * angle) / sin_angle;
    const w1 = @sin(t * angle) / sin_angle;
    return .{
        .v = a.v * @as(Q.Vec, @splat(w0)) +
            other.v * @as(Q.Vec, @splat(w1)),
        .w = a.w * w0 + other.w * w1,
    };
}

/// Interpolates two quaternions linearly and
/// normalizes the result.
///
/// # Args
/// - `a`: start rotation; a unit quaternion.
/// - `b`: end rotation; a unit quaternion.
/// - `t`: interpolation factor.
///
/// # Return
/// The normalized blend along the shortest arc.
///
/// # Note
/// Faster than `slerp` but not constant speed.
/// `t` is not clamped.
///
/// # Compile Error
/// - The operands are not the same quaternion
///   type.
pub fn nlerp(a: anytype, b: anytype, t: anytype) @TypeOf(a) {
    const Q = @TypeOf(a);
    var other: Q = b;
    if (dot(a, other) < 0) other = negate(other);

    const w0 = 1 - t;
    const blended: Q = .{
        .v = a.v * @as(Q.Vec, @splat(w0)) +
            other.v * @as(Q.Vec, @splat(t)),
        .w = a.w * w0 + other.w * t,
    };
    return normalize(blended);
}

/// Negates a quaternion.
///
/// # Args
/// - `q`: the quaternion to negate.
///
/// # Return
/// The negated quaternion, which describes the same
/// rotation.
pub fn negate(q: anytype) @TypeOf(q) {
    return .{ .v = -q.v, .w = -q.w };
}

test "quat basics" {
    const Q = Quat(f32);
    const identity = Q.identity();
    try std.testing.expectEqual(@as(f32, 1), identity.w);
    try std.testing.expectEqual(@as(f32, 0), identity.v[0]);

    const rotation = fromAxisAngle(Vec3(f32){ 0, 0, 1 }, @as(f32, std.math.pi) / 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(rotation), 1e-6);
    const inverse_rotation = inverse(rotation, 1e-6).?;
    const product = multiply(rotation, inverse_rotation);
    try std.testing.expectApproxEqAbs(@as(f32, 1), product.w, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), product.v[0], 1e-6);
    try std.testing.expect(inverse(Q.identity(), 1e-6) != null);
    try std.testing.expect(inverse(Q{ .v = @splat(0), .w = 0 }, 1e-6) == null);
}

test "quat rotate matches matrix" {
    const axis = vec.normalize(Vec3(f32){ 1, 2, 3 });
    const rotation = fromAxisAngle(axis, 0.7);
    const matrix = toMat3(rotation);
    const point: Vec3(f32) = .{ 1, -2, 0.5 };
    const by_quat = rotateVector(rotation, point);
    const by_matrix = mat.mulVec(matrix, point);
    try std.testing.expectApproxEqAbs(by_matrix[0], by_quat[0], 1e-6);
    try std.testing.expectApproxEqAbs(by_matrix[1], by_quat[1], 1e-6);
    try std.testing.expectApproxEqAbs(by_matrix[2], by_quat[2], 1e-6);
}

test "quat matrix round trip" {
    const cases = [_]Vec3(f32){
        .{ 0, 0, 0 },
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
        .{ 0.3, -1.2, 0.7 },
        .{ @as(f32, std.math.pi), 0.2, -0.4 },
    };
    for (cases) |angles| {
        const q = multiply(
            fromAxisAngle(Vec3(f32){ 0, 0, 1 }, angles[2]),
            multiply(
                fromAxisAngle(Vec3(f32){ 0, 1, 0 }, angles[1]),
                fromAxisAngle(Vec3(f32){ 1, 0, 0 }, angles[0]),
            ),
        );
        const back = fromMat3(toMat3(q));
        const a = toMat3(q);
        const b = toMat3(back);
        inline for (0..3) |c| {
            inline for (0..3) |r| {
                try std.testing.expectApproxEqAbs(
                    a.columns[c][r],
                    b.columns[c][r],
                    1e-5,
                );
            }
        }
    }
}

test "quat slerp" {
    const a = Quat(f32).identity();
    const b = fromAxisAngle(Vec3(f32){ 0, 0, 1 }, @as(f32, std.math.pi) / 2);
    const start = slerp(a, b, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), start.w, 1e-6);
    const end = slerp(a, b, 1);
    try std.testing.expectApproxEqAbs(b.w, end.w, 1e-6);
    const middle = slerp(a, b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(middle), 1e-6);
    const quarter = @as(f32, std.math.pi) / 4;
    try std.testing.expectApproxEqAbs(
        @cos(quarter * 0.5),
        middle.w,
        1e-5,
    );
}

test "quat nlerp" {
    const a = Quat(f32).identity();
    const b = fromAxisAngle(Vec3(f32){ 1, 0, 0 }, 1.0);
    const middle = nlerp(a, b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(middle), 1e-6);
    const same = nlerp(a, negate(a), 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), length(same), 1e-6);
}

test "quat from vectors" {
    const from: Vec3(f32) = .{ 0, 0, 1 };
    const to: Vec3(f32) = .{ 1, 0, 0 };
    const rotation = fromVectors(from, to, 1e-6).?;
    const rotated = rotateVector(rotation, from);
    try std.testing.expectApproxEqAbs(to[0], rotated[0], 1e-6);
    try std.testing.expectApproxEqAbs(to[1], rotated[1], 1e-6);
    try std.testing.expectApproxEqAbs(to[2], rotated[2], 1e-6);

    const opposite = fromVectors(from, -from, 1e-6).?;
    const flipped = rotateVector(opposite, from);
    try std.testing.expectApproxEqAbs(-from[0], flipped[0], 1e-6);
    try std.testing.expectApproxEqAbs(-from[1], flipped[1], 1e-6);
    try std.testing.expectApproxEqAbs(-from[2], flipped[2], 1e-6);

    try std.testing.expect(
        fromVectors(Vec3(f32){ 0, 0, 0 }, to, 1e-6) == null,
    );
}
