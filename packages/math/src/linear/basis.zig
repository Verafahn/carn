//! Orthonormal bases built with Gram-Schmidt.
//!
//! A basis stores three unit vectors. Constructors
//! return `null` instead of inventing a direction when
//! the input degenerates.

const std = @import("std");
const vector = @import("vector.zig");

/// An orthonormal right-handed basis.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Basis(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Basis: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// First basis vector.
        tangent: vector.Vector(T, 3),
        /// Second basis vector.
        bitangent: vector.Vector(T, 3),
        /// Third basis vector.
        normal: vector.Vector(T, 3),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a basis around a normal.
        ///
        /// A world axis that is least aligned with the
        /// normal seeds the tangent, so every direction
        /// yields a stable basis.
        ///
        /// # Args
        /// - `normal`: the desired third basis vector.
        /// - `epsilon_value`: smallest accepted length.
        ///
        /// # Return
        /// The basis, or `null` when `normal` is too
        /// short.
        ///
        /// # Note
        /// The tangent is discontinuous across the
        /// sphere; do not rely on it being smooth.
        pub fn fromNormal(
            normal: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?Self {
            if (!normal.isNormalizable(epsilon_value)) {
                return null;
            }
            const unit = normal.normalize();
            const abs_x = @abs(unit.at(0));
            const abs_y = @abs(unit.at(1));
            const abs_z = @abs(unit.at(2));
            var helper: vector.Vector(T, 3) = undefined;
            if (abs_x <= abs_y and abs_x <= abs_z) {
                helper = vector.Vector(T, 3)
                    .fromElements(.{ 1, 0, 0 });
            } else if (abs_y <= abs_z) {
                helper = vector.Vector(T, 3)
                    .fromElements(.{ 0, 1, 0 });
            } else {
                helper = vector.Vector(T, 3)
                    .fromElements(.{ 0, 0, 1 });
            }
            const tangent = helper.cross(unit).normalize();
            return .{
                .tangent = tangent,
                .bitangent = unit.cross(tangent),
                .normal = unit,
            };
        }

        /// Orthonormalizes three vectors in priority
        /// order.
        ///
        /// The first vector is kept, the second is made
        /// perpendicular to it, and the third is made
        /// perpendicular to both.
        ///
        /// # Args
        /// - `values`: three vectors of the same type.
        /// - `epsilon_value`: smallest accepted length
        ///   at each step.
        ///
        /// # Return
        /// The basis, or `null` when a step degenerates.
        ///
        /// # Note
        /// The result depends on the input order.
        pub fn fromVectors(
            values: [3]vector.Vector(T, 3),
            epsilon_value: T,
        ) ?Self {
            if (!values[0].isNormalizable(epsilon_value)) {
                return null;
            }
            const first = values[0].normalize();

            const second_raw = values[1].sub(
                first.scale(first.dot(values[1])),
            );
            if (!second_raw.isNormalizable(epsilon_value)) {
                return null;
            }
            const second = second_raw.normalize();

            const third_raw = values[2]
                .sub(first.scale(first.dot(values[2])))
                .sub(second.scale(second.dot(values[2])));
            if (!third_raw.isNormalizable(epsilon_value)) {
                return null;
            }
            const third = third_raw.normalize();

            return .{
                .tangent = first,
                .bitangent = second,
                .normal = third,
            };
        }
    };
}

test "basis from normal" {
    const Vec3f = vector.Vec3(f32);
    const normals = [_]Vec3f{
        Vec3f.fromElements(.{ 0, 0, 1 }),
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 0, 1, 0 }),
        Vec3f.fromElements(.{ 0.3, -0.4, 0.5 }),
    };
    for (normals) |normal| {
        const frame = Basis(f32).fromNormal(normal, 1e-6).?;
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            frame.tangent.length(),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            @as(f32, 0),
            frame.tangent.dot(frame.bitangent),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            @as(f32, 0),
            frame.tangent.dot(frame.normal),
            1e-6,
        );
        const expected = normal.normalize();
        try std.testing.expectApproxEqAbs(
            expected.at(0),
            frame.normal.at(0),
            1e-6,
        );
    }
    try std.testing.expect(
        Basis(f32).fromNormal(Vec3f.zero(), 1e-6) == null,
    );
}

test "basis gram schmidt" {
    const Vec3f = vector.Vec3(f32);
    const values = [3]Vec3f{
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 1, 1, 0 }),
        Vec3f.fromElements(.{ 0, 1, 1 }),
    };
    const frame = Basis(f32).fromVectors(values, 1e-6).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        frame.tangent.dot(frame.bitangent),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        frame.bitangent.dot(frame.normal),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        frame.normal.length(),
        1e-6,
    );

    const degenerate = [3]Vec3f{
        Vec3f.fromElements(.{ 1, 0, 0 }),
        Vec3f.fromElements(.{ 2, 0, 0 }),
        Vec3f.fromElements(.{ 0, 0, 1 }),
    };
    try std.testing.expect(
        Basis(f32).fromVectors(degenerate, 1e-6) == null,
    );
}
