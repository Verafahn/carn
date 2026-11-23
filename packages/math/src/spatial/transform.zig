//! Local and world space transform wrapper.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const matrix = @import("../linear/matrix.zig");

/// A local-to-world transform.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Transform(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Transform: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// The local-to-world matrix.
        local_to_world: matrix.Mat4(T),

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Returns the identity transform.
        ///
        /// # Return
        /// A transform that leaves points unchanged.
        pub fn identity() Self {
            return .{
                .local_to_world = matrix.Mat4(T).identity(),
            };
        }

        /// Wraps an existing matrix.
        ///
        /// # Args
        /// - `value`: the local-to-world matrix.
        ///
        /// # Return
        /// A transform using `value`.
        pub fn fromMatrix(value: matrix.Mat4(T)) Self {
            return .{ .local_to_world = value };
        }

        /// Returns the underlying matrix.
        ///
        /// # Return
        /// The local-to-world matrix.
        pub fn toMatrix(self: Self) matrix.Mat4(T) {
            return self.local_to_world;
        }

        /// Transforms a point into world space.
        ///
        /// # Args
        /// - `point`: a point in local space.
        ///
        /// # Return
        /// The point in world space.
        pub fn toWorld(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            return self.local_to_world.transformPoint(point);
        }

        /// Transforms a point into local space.
        ///
        /// # Args
        /// - `point`: a point in world space.
        /// - `epsilon_value`: smallest accepted
        ///   determinant.
        ///
        /// # Return
        /// The point in local space, or `null` when
        /// the matrix is not invertible.
        pub fn toLocal(
            self: Self,
            point: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?vector.Vector(T, 3) {
            return self.local_to_world.toLocal(
                point,
                epsilon_value,
            );
        }

        /// Transforms a direction into world space.
        ///
        /// # Args
        /// - `value`: a direction in local space.
        ///
        /// # Return
        /// The direction in world space.
        pub fn directionToWorld(
            self: Self,
            value: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            return self.local_to_world.transformDirection(value);
        }

        /// Transforms a normal into world space.
        ///
        /// # Args
        /// - `normal`: the normal to transform.
        /// - `epsilon_value`: smallest accepted
        ///   determinant.
        ///
        /// # Return
        /// The transformed normal, or `null` when the
        /// matrix is not invertible.
        pub fn normalToWorld(
            self: Self,
            normal: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?vector.Vector(T, 3) {
            return self.local_to_world.transformNormal(
                normal,
                epsilon_value,
            );
        }
    };
}

test "transform round trip" {
    const Vec3f = vector.Vec3(f32);
    const Vec4f = vector.Vec4(f32);
    const matrix_value = matrix.Mat4(f32).fromRows(.{
        Vec4f.fromElements(.{ 0, -1, 0, 0 }),
        Vec4f.fromElements(.{ 1, 0, 0, 0 }),
        Vec4f.fromElements(.{ 0, 0, 1, 0 }),
        Vec4f.fromElements(.{ 4, 5, 6, 1 }),
    });
    const transform = Transform(f32).fromMatrix(matrix_value);
    const local = Vec3f.fromElements(.{ 3, 2, 1 });
    const world = transform.toWorld(local);
    try std.testing.expectApproxEqAbs(@as(f32, 6), world.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), world.at(1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), world.at(2), 1e-6);

    const back = transform.toLocal(world, 1e-9).?;
    try std.testing.expectApproxEqAbs(local.at(0), back.at(0), 1e-5);
    try std.testing.expectApproxEqAbs(local.at(2), back.at(2), 1e-5);

    const direction = transform.directionToWorld(local);
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        direction.at(0),
        1e-6,
    );

    const identity = Transform(f32).identity();
    try std.testing.expectEqual(
        local,
        identity.toWorld(local),
    );
}
