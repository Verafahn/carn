//! Affine transform decomposition.

const std = @import("std");
const vector = @import("vector.zig");
const matrix = @import("matrix.zig");
const quaternion = @import("quaternion.zig");

/// The parts of an affine transform.
///
/// # Args
/// - `T`: the float component type.
pub fn Decomposition(comptime T: type) type {
    return struct {
        /// Translation part.
        translation: vector.Vector(T, 3),
        /// Rotation part; a unit quaternion.
        rotation: quaternion.Quat(T),
        /// Scale along the local axes.
        scale: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Splits an affine `4x4` matrix.
        ///
        /// # Args
        /// - `m`: a `4x4` affine matrix with no shear;
        ///   it must equal `S * R * T`.
        /// - `epsilon_value`: smallest accepted row
        ///   length.
        ///
        /// # Return
        /// The parts, or `null` when a scale component
        /// is too small.
        ///
        /// # Note
        /// Shear is not represented, so a sheared
        /// matrix yields an approximate rotation. A
        /// negative determinant moves the sign onto
        /// `scale.x`.
        pub fn fromMatrix(
            m: matrix.Mat4(T),
            epsilon_value: T,
        ) ?Self {
            const translation = vector.Vector(T, 3)
                .fromElements(.{
                m.at(3, 0),
                m.at(3, 1),
                m.at(3, 2),
            });

            var rows: [3]vector.Vector(T, 3) = undefined;
            inline for (0..3) |r| {
                rows[r] = vector.Vector(T, 3)
                    .fromElements(.{
                    m.at(r, 0),
                    m.at(r, 1),
                    m.at(r, 2),
                });
            }

            var scales = vector.Vector(T, 3).zero();
            inline for (0..3) |r| {
                if (!rows[r].isNormalizable(epsilon_value)) {
                    return null;
                }
                const length = rows[r].length();
                scales.set(r, length);
                rows[r] = rows[r].scale(1 / length);
            }

            var rotation = matrix.Mat3(T).fromRows(rows);
            if (rotation.determinant() < 0) {
                scales.set(0, -scales.at(0));
                rows[0] = rows[0].negate();
                rotation = matrix.Mat3(T).fromRows(rows);
            }

            return .{
                .translation = translation,
                .rotation = quaternion.Quat(T)
                    .fromMat3(rotation),
                .scale = scales,
            };
        }
    };
}

test "decompose affine matrix" {
    const Vec3f = vector.Vec3(f32);
    const Vec4f = vector.Vec4(f32);
    const rotation = quaternion.Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 1, 2, 3 }).normalize(),
        0.6,
    );
    const r3 = rotation.toMat3();
    const r4 = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ r3.at(0, 0), r3.at(0, 1), r3.at(0, 2), 0 }),
        Vec4f.fromElements(.{ r3.at(1, 0), r3.at(1, 1), r3.at(1, 2), 0 }),
        Vec4f.fromElements(.{ r3.at(2, 0), r3.at(2, 1), r3.at(2, 2), 0 }),
        Vec4f.fromElements(.{ 0, 0, 0, 1 }),
    });
    const scale_matrix = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ 2, 0, 0, 0 }),
        Vec4f.fromElements(.{ 0, 3, 0, 0 }),
        Vec4f.fromElements(.{ 0, 0, 4, 0 }),
        Vec4f.fromElements(.{ 0, 0, 0, 1 }),
    });
    const translation = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ 1, 0, 0, 0 }),
        Vec4f.fromElements(.{ 0, 1, 0, 0 }),
        Vec4f.fromElements(.{ 0, 0, 1, 0 }),
        Vec4f.fromElements(.{ 5, 6, 7, 1 }),
    });
    const combined = scale_matrix.mul(r4).mul(translation);
    const parts = combined.decompose(1e-5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 5),
        parts.translation.at(0),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 7),
        parts.translation.at(2),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 2), parts.scale.at(0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3), parts.scale.at(1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4), parts.scale.at(2), 1e-5);

    const rebuilt = parts.rotation.toMat3();
    inline for (0..3) |r| {
        inline for (0..3) |c| {
            try std.testing.expectApproxEqAbs(
                r3.at(r, c),
                rebuilt.at(r, c),
                1e-5,
            );
        }
    }
}

test "decompose negative scale" {
    const Vec3f = vector.Vec3(f32);
    const Vec4f = vector.Vec4(f32);
    const rotation = quaternion.Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 0, 1, 0 }),
        0.4,
    );
    const r3 = rotation.toMat3();
    const r4 = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ r3.at(0, 0), r3.at(0, 1), r3.at(0, 2), 0 }),
        Vec4f.fromElements(.{ r3.at(1, 0), r3.at(1, 1), r3.at(1, 2), 0 }),
        Vec4f.fromElements(.{ r3.at(2, 0), r3.at(2, 1), r3.at(2, 2), 0 }),
        Vec4f.fromElements(.{ 0, 0, 0, 1 }),
    });
    const scale_matrix = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ -2, 0, 0, 0 }),
        Vec4f.fromElements(.{ 0, 3, 0, 0 }),
        Vec4f.fromElements(.{ 0, 0, 4, 0 }),
        Vec4f.fromElements(.{ 0, 0, 0, 1 }),
    });
    const parts = scale_matrix.mul(r4).decompose(1e-5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        parts.scale.at(0),
        1e-5,
    );
    const rebuilt = parts.rotation.toMat3();
    inline for (0..3) |r| {
        inline for (0..3) |c| {
            try std.testing.expectApproxEqAbs(
                r3.at(r, c),
                rebuilt.at(r, c),
                1e-5,
            );
        }
    }
}
