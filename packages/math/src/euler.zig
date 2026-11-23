//! Euler angles and rotation orders.
//!
//! An `Euler` value stores one angle per ordered
//! axis plus the order itself. Angles are in radians
//! and are applied in the order listed: `.xyz` means
//! rotate about X, then Y, then Z.
//!
//! Both Tait-Bryan orders such as `.xyz` and proper
//! orders such as `.xyx` are supported. The three
//! angles always map to the three ordered axes, not
//! to the X, Y, and Z components.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const quat = @import("quat.zig");
const Vec3f = vec.Vec3(f32);

/// The order in which Euler angles are applied.
///
/// Members list axes in application order. Orders
/// such as `xyz` are Tait-Bryan; orders whose first
/// and third axes match, such as `xyx`, are proper
/// Euler orders.
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
        angles: @Vector(3, T),
        /// The axis order used to apply the angles.
        order: Order,

        /// The component type.
        pub const Component = T;
    };
}

/// Converts an Euler rotation to a quaternion.
///
/// # Args
/// - `rotation`: the angles and their order.
///
/// # Return
/// The equivalent unit quaternion.
///
/// # Note
/// The rotation applies the first angle first, so
/// `.xyz` matches three successive rotations.
///
/// # Compile Error
/// - `rotation` is not an `Euler` value.
pub fn toQuat(rotation: anytype) quat.Quat(
    scalar.ScalarOf(@TypeOf(rotation.angles)),
) {
    return switch (rotation.order) {
        inline else => |order| toQuatOrder(order, rotation.angles),
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
/// At gimbal lock the middle angle is fixed and the
/// first angle is set to zero; the result still
/// describes the same rotation. Angles are not
/// wrapped into a canonical range. The extraction
/// calls inverse trigonometric functions, which may
/// not lower on the SPIR-V backend.
///
/// # Compile Error
/// - `rotation` is not a quaternion.
pub fn fromQuat(rotation: anytype, order: Order) Euler(
    @TypeOf(rotation.v[0]),
) {
    return switch (order) {
        inline else => |tag| fromQuatOrder(tag, rotation),
    };
}

/// Converts degrees to radians.
///
/// # Args
/// - `value`: an angle in degrees; a float or
///   float vector.
///
/// # Return
/// The angle in radians.
pub fn degreesToRadians(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const factor = std.math.pi / 180.0;
    return value * switch (@typeInfo(T)) {
        .vector => @as(T, @splat(factor)),
        else => factor,
    };
}

/// Converts radians to degrees.
///
/// # Args
/// - `value`: an angle in radians; a float or
///   float vector.
///
/// # Return
/// The angle in degrees.
pub fn radiansToDegrees(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const factor = 180.0 / std.math.pi;
    return value * switch (@typeInfo(T)) {
        .vector => @as(T, @splat(factor)),
        else => factor,
    };
}

fn toQuatOrder(comptime order: Order, angles: anytype) quat.Quat(
    scalar.ScalarOf(@TypeOf(angles)),
) {
    const T = scalar.ScalarOf(@TypeOf(angles));
    const Q = quat.Quat(T);
    const axes = comptime axesOf(order);
    var result = Q.identity();
    inline for (0..3) |i| {
        const axis = comptime axes[i];
        result = quat.multiply(
            quat.fromAxisAngle(axisVector(axis, T), angles[i]),
            result,
        );
    }
    return result;
}

fn fromQuatOrder(comptime order: Order, rotation: anytype) Euler(
    @TypeOf(rotation.v[0]),
) {
    const T = @TypeOf(rotation.v[0]);
    const axes = comptime axesOf(order);
    const a0 = comptime axes[0];
    const a1 = comptime axes[1];
    const a2 = comptime axes[2];
    const sign: T = comptime if (nextAxis(a0) == a1) 1 else -1;
    const matrix = quat.toMat3(rotation);

    var angles: @Vector(3, T) = @splat(0);
    if (a0 == a2) {
        const b = a1;
        const c = comptime otherAxis(a0, a1);
        angles[1] = std.math.acos(clampUnit(matrix.at(a0, a0)));
        angles[0] = std.math.atan2(
            matrix.at(b, a0),
            sign * matrix.at(c, a0),
        );
    } else {
        angles[1] = std.math.asin(clampUnit(
            -sign * matrix.at(a0, a2),
        ));
        angles[0] = std.math.atan2(
            sign * matrix.at(a1, a2),
            matrix.at(a2, a2),
        );
    }

    const q0 = quat.fromAxisAngle(axisVector(a0, T), angles[0]);
    const q1 = quat.fromAxisAngle(axisVector(a1, T), angles[1]);
    const q2 = quat.multiply(
        quat.multiply(rotation, quat.conjugate(q0)),
        quat.conjugate(q1),
    );
    angles[2] = 2 * std.math.atan2(
        vec.dot(q2.v, axisVector(a2, T)),
        q2.w,
    );

    return .{ .angles = angles, .order = order };
}

fn clampUnit(value: anytype) @TypeOf(value) {
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

fn axisVector(comptime axis: usize, comptime T: type) @Vector(3, T) {
    return switch (axis) {
        0 => .{ 1, 0, 0 },
        1 => .{ 0, 1, 0 },
        else => .{ 0, 0, 1 },
    };
}

test "angle conversion" {
    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi),
        degreesToRadians(@as(f32, 180)),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 180),
        radiansToDegrees(@as(f32, std.math.pi)),
        1e-4,
    );
    const degrees: @Vector(2, f32) = .{ 90, 180 };
    const radians = degreesToRadians(degrees);
    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi) / 2,
        radians[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi),
        radians[1],
        1e-6,
    );
}

test "euler to quat applies angles in order" {
    const angles: @Vector(3, f32) = .{ 0.3, -0.7, 1.1 };
    const rotation = toQuat(Euler(f32){
        .angles = angles,
        .order = .xyz,
    });

    const x = quat.fromAxisAngle(Vec3f{ 1, 0, 0 }, angles[0]);
    const y = quat.fromAxisAngle(Vec3f{ 0, 1, 0 }, angles[1]);
    const z = quat.fromAxisAngle(Vec3f{ 0, 0, 1 }, angles[2]);
    const expected = quat.multiply(z, quat.multiply(y, x));

    try std.testing.expectApproxEqAbs(expected.w, rotation.w, 1e-6);
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(
            expected.v[i],
            rotation.v[i],
            1e-6,
        );
    }

    const point: Vec3f = .{ 1, 2, 3 };
    const by_euler = quat.rotateVector(rotation, point);
    const by_steps = quat.rotateVector(
        z,
        quat.rotateVector(y, quat.rotateVector(x, point)),
    );
    inline for (0..3) |i| {
        try std.testing.expectApproxEqAbs(
            by_steps[i],
            by_euler[i],
            1e-5,
        );
    }
}

test "euler round trips for every order" {
    const triples = [_]@Vector(3, f32){
        .{ 0.3, -0.7, 1.1 },
        .{ -1.2, 0.4, 2.6 },
        .{ 0.0, 0.0, 0.0 },
        .{ 0.9, 0.0, -0.2 },
        .{ -0.4, 1.3, 0.05 },
        .{ 0.0, std.math.pi, 0.0 },
    };
    inline for (std.enums.values(Order)) |order| {
        for (triples) |angles| {
            const rotation = toQuat(Euler(f32){
                .angles = angles,
                .order = order,
            });
            const back = fromQuat(rotation, order);
            const rebuilt = toQuat(back);
            const first = quat.toMat3(rotation);
            const second = quat.toMat3(rebuilt);
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
    const rotation = toQuat(Euler(f32){
        .angles = .{ 0, 0, 0 },
        .order = .zyx,
    });
    try std.testing.expectApproxEqAbs(@as(f32, 1), rotation.w, 1e-6);
    const back = fromQuat(rotation, .xyx);
    try std.testing.expectApproxEqAbs(@as(f32, 0), back.angles[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), back.angles[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), back.angles[2], 1e-6);
}
