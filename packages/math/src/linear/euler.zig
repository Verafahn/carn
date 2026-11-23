//! Euler angles and rotation orders.
//!
//! An `Euler` value stores one angle per ordered axis
//! plus the order itself. The three angles always map
//! to the ordered axes, not to X, Y, and Z, so proper
//! orders such as `.xyx` work as well as Tait-Bryan
//! orders such as `.xyz`.
//!
//! Angles are in radians and are applied in the listed
//! order: `.xyz` rotates about X, then Y, then Z.

const std = @import("std");
const vector = @import("vector.zig");
const quaternion = @import("quaternion.zig");

/// The order in which Euler angles are applied.
///
/// Members list axes in application order. Orders
/// whose first and third axes match, such as `xyx`,
/// are proper Euler orders.
pub const Order = enum {
    /// Rotate about X, then Y, then Z.
    xyz,
    /// Rotate about X, then Z, then Y.
    xzy,
    /// Rotate about Y, then X, then Z.
    yxz,
    /// Rotate about Y, then Z, then X.
    yzx,
    /// Rotate about Z, then X, then Y.
    zxy,
    /// Rotate about Z, then Y, then X.
    zyx,
    /// Rotate about X, then Y, then X.
    xyx,
    /// Rotate about X, then Z, then X.
    xzx,
    /// Rotate about Y, then X, then Y.
    yxy,
    /// Rotate about Y, then Z, then Y.
    yzy,
    /// Rotate about Z, then X, then Z.
    zxz,
    /// Rotate about Z, then Y, then Z.
    zyz,
};

/// Returns a Euler angle type with components of `T`.
///
/// # Args
/// - `T`: the angle component type; a float.
///
/// # Return
/// A struct holding three angles and their order.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Euler(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Euler: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Angles for the ordered axes, in radians.
        angles: vector.Vector(T, 3),
        /// The axis order used to apply the angles.
        order: Order,

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Converts the angles to a quaternion.
        ///
        /// # Return
        /// The equivalent unit quaternion.
        ///
        /// # Note
        /// The first angle is applied first.
        pub fn toQuat(self: Self) quaternion.Quat(T) {
            return switch (self.order) {
                inline else => |order| {
                    return toQuatOrder(T, order, self.angles);
                },
            };
        }

        /// Converts a quaternion to Euler angles.
        ///
        /// # Args
        /// - `rotation`: a unit quaternion.
        /// - `order`: the desired application order.
        ///
        /// # Return
        /// The angles and order.
        ///
        /// # Note
        /// At gimbal lock the middle angle is fixed and
        /// the first angle becomes zero; the result still
        /// describes the same rotation. Angles are not
        /// wrapped. The extraction uses inverse
        /// trigonometric functions, which may not lower
        /// on SPIR-V.
        pub fn fromQuat(
            rotation: quaternion.Quat(T),
            order: Order,
        ) Self {
            return switch (order) {
                inline else => |tag| {
                    return fromQuatOrder(T, tag, rotation);
                },
            };
        }

        /// Builds angles from degrees.
        ///
        /// # Args
        /// - `degrees`: one angle per ordered axis.
        /// - `order`: the application order.
        ///
        /// # Return
        /// The same rotation in radians.
        pub fn fromDegrees(
            degrees: vector.Vector(T, 3),
            order: Order,
        ) Self {
            var radians: vector.Vector(T, 3) = undefined;
            inline for (0..3) |i| {
                radians.set(i, Self.degreesToRadians(
                    degrees.at(i),
                ));
            }
            return .{ .angles = radians, .order = order };
        }

        /// Converts the stored angles to degrees.
        ///
        /// # Return
        /// One degree angle per ordered axis.
        pub fn toDegrees(self: Self) vector.Vector(T, 3) {
            var degrees: vector.Vector(T, 3) = undefined;
            inline for (0..3) |i| {
                degrees.set(i, Self.radiansToDegrees(
                    self.angles.at(i),
                ));
            }
            return degrees;
        }

        /// Converts degrees to radians.
        ///
        /// # Args
        /// - `degrees`: the angle in degrees.
        ///
        /// # Return
        /// The angle in radians.
        pub fn degreesToRadians(degrees: T) T {
            return degrees * (std.math.pi / 180.0);
        }

        /// Converts radians to degrees.
        ///
        /// # Args
        /// - `radians`: the angle in radians.
        ///
        /// # Return
        /// The angle in degrees.
        pub fn radiansToDegrees(radians: T) T {
            return radians * (180.0 / std.math.pi);
        }
    };
}

fn toQuatOrder(
    comptime T: type,
    comptime order: Order,
    angles: vector.Vector(T, 3),
) quaternion.Quat(T) {
    const Q = quaternion.Quat(T);
    const axes = comptime axesOf(order);
    var result = Q.identity();
    inline for (0..3) |i| {
        const axis = comptime axes[i];
        result = Q.fromAxisAngle(
            axisVector(axis, T),
            angles.at(i),
        ).multiply(result);
    }
    return result;
}

fn fromQuatOrder(
    comptime T: type,
    comptime order: Order,
    rotation: quaternion.Quat(T),
) Euler(T) {
    const Q = quaternion.Quat(T);
    const axes = comptime axesOf(order);
    const a0 = comptime axes[0];
    const a1 = comptime axes[1];
    const a2 = comptime axes[2];
    const sign: T = comptime if (nextAxis(a0) == a1) 1 else -1;
    const m = rotation.toMat3();

    var angles = vector.Vector(T, 3).zero();
    if (a0 == a2) {
        const b = a1;
        const c = comptime otherAxis(a0, a1);
        angles.set(1, std.math.acos(clampUnit(T, m.at(a0, a0))));
        angles.set(0, std.math.atan2(
            m.at(b, a0),
            sign * m.at(c, a0),
        ));
    } else {
        angles.set(1, std.math.asin(clampUnit(
            T,
            -sign * m.at(a0, a2),
        )));
        angles.set(0, std.math.atan2(
            sign * m.at(a1, a2),
            m.at(a2, a2),
        ));
    }

    const q0 = Q.fromAxisAngle(axisVector(a0, T), angles.at(0));
    const q1 = Q.fromAxisAngle(axisVector(a1, T), angles.at(1));
    const q2 = rotation.multiply(q0.conjugate())
        .multiply(q1.conjugate());
    angles.set(2, 2 * std.math.atan2(
        q2.v.dot(axisVector(a2, T)),
        q2.w,
    ));

    return .{ .angles = angles, .order = order };
}

fn clampUnit(comptime T: type, value: T) T {
    return @min(@max(value, -1), 1);
}

fn axesOf(comptime order: Order) [3]usize {
    return switch (order) {
        .xyz => .{ 0, 1, 2 },
        .xzy => .{ 0, 2, 1 },
        .yxz => .{ 1, 0, 2 },
        .yzx => .{ 1, 2, 0 },
        .zxy => .{ 2, 0, 1 },
        .zyx => .{ 2, 1, 0 },
        .xyx => .{ 0, 1, 0 },
        .xzx => .{ 0, 2, 0 },
        .yxy => .{ 1, 0, 1 },
        .yzy => .{ 1, 2, 1 },
        .zxz => .{ 2, 0, 2 },
        .zyz => .{ 2, 1, 2 },
    };
}

fn nextAxis(comptime axis: usize) usize {
    return (axis + 1) % 3;
}

fn otherAxis(comptime first: usize, comptime second: usize) usize {
    return 3 - first - second;
}

fn axisVector(
    comptime axis: usize,
    comptime T: type,
) vector.Vector(T, 3) {
    return switch (axis) {
        0 => vector.Vector(T, 3).fromElements(.{ 1, 0, 0 }),
        1 => vector.Vector(T, 3).fromElements(.{ 0, 1, 0 }),
        else => vector.Vector(T, 3)
            .fromElements(.{ 0, 0, 1 }),
    };
}

test "angle conversion" {
    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi),
        Euler(f32).degreesToRadians(180),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 180),
        Euler(f32).radiansToDegrees(@as(f32, std.math.pi)),
        1e-4,
    );
    const degrees = vector.Vec3(f32).fromElements(.{ 90, 180, 0 });
    const converted = Euler(f32).fromDegrees(degrees, .xyz);
    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi) * 0.5,
        converted.angles.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 90),
        converted.toDegrees().at(0),
        1e-4,
    );
}

test "euler to quat applies angles in order" {
    const Vec3f = vector.Vec3(f32);
    const angles = Vec3f.fromElements(.{ 0.3, -0.7, 1.1 });
    const rotation = (Euler(f32){
        .angles = angles,
        .order = .xyz,
    }).toQuat();

    const x = quaternion.Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 1, 0, 0 }),
        angles.at(0),
    );
    const y = quaternion.Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 0, 1, 0 }),
        angles.at(1),
    );
    const z = quaternion.Quat(f32).fromAxisAngle(
        Vec3f.fromElements(.{ 0, 0, 1 }),
        angles.at(2),
    );
    const expected = z.multiply(y).multiply(x);

    try std.testing.expectApproxEqAbs(expected.w, rotation.w, 1e-6);
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(
            expected.v.at(i),
            rotation.v.at(i),
            1e-6,
        );
    }

    const point = Vec3f.fromElements(.{ 1, 2, 3 });
    const by_euler = rotation.rotateVector(point);
    const by_steps = z.rotateVector(
        y.rotateVector(x.rotateVector(point)),
    );
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(
            by_steps.at(i),
            by_euler.at(i),
            1e-5,
        );
    }
}

test "euler round trips for every order" {
    const Vec3f = vector.Vec3(f32);
    const triples = [_]Vec3f{
        Vec3f.fromElements(.{ 0.3, -0.7, 1.1 }),
        Vec3f.fromElements(.{ -1.2, 0.4, 2.6 }),
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0.9, 0, -0.2 }),
        Vec3f.fromElements(.{ -0.4, 1.3, 0.05 }),
        Vec3f.fromElements(.{ 0, std.math.pi, 0 }),
    };
    inline for (std.enums.values(Order)) |order| {
        for (triples) |angles| {
            const rotation = (Euler(f32){
                .angles = angles,
                .order = order,
            }).toQuat();
            const rebuilt = Euler(f32)
                .fromQuat(rotation, order)
                .toQuat();
            const first = rotation.toMat3();
            const second = rebuilt.toMat3();
            inline for (0..3) |c| {
                inline for (0..3) |r| {
                    try std.testing.expectApproxEqAbs(
                        first.at(r, c),
                        second.at(r, c),
                        1e-4,
                    );
                }
            }
        }
    }
}

test "euler identity" {
    const rotation = (Euler(f32){
        .angles = vector.Vec3(f32).zero(),
        .order = .zyx,
    }).toQuat();
    try std.testing.expectApproxEqAbs(@as(f32, 1), rotation.w, 1e-6);
    const back = Euler(f32).fromQuat(rotation, .xyx);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        back.angles.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        back.angles.at(1),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        back.angles.at(2),
        1e-6,
    );
}
