//! Orthonormal bases built with Gram-Schmidt.
//!
//! Both helpers return three unit vectors in the
//! order `.{ tangent, bitangent, normal }`. They
//! return `null` instead of inventing a direction
//! when the input degenerates.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;

/// Orthonormalizes three vectors in priority order.
///
/// The first vector is kept, the second is made
/// perpendicular to it, and the third is made
/// perpendicular to both.
///
/// # Args
/// - `values`: three vectors of the same type.
/// - `epsilon`: smallest accepted length at each
///   step.
///
/// # Return
/// Three orthonormal vectors, or `null` when a step
/// degenerates.
///
/// # Note
/// Earlier vectors have priority, so the result
/// depends on the input order.
///
/// # Compile Error
/// - `values` is not an array of three vectors.
pub fn gramSchmidt(
    values: anytype,
    epsilon: anytype,
) ?[3]@TypeOf(values[0]) {
    const V = @TypeOf(values[0]);
    if (@TypeOf(values) != [3]V) {
        @compileError(
            "gramSchmidt: `values` must hold three vectors",
        );
    }
    if (!scalar.isNormalizable(values[0], epsilon)) return null;
    const first = vec.normalize(values[0]);

    const second_raw = values[1] -
        first * @as(V, @splat(vec.dot(first, values[1])));
    if (!scalar.isNormalizable(second_raw, epsilon)) return null;
    const second = vec.normalize(second_raw);

    const third_raw = values[2] -
        first * @as(V, @splat(vec.dot(first, values[2]))) -
        second * @as(V, @splat(vec.dot(second, values[2])));
    if (!scalar.isNormalizable(third_raw, epsilon)) return null;
    const third = vec.normalize(third_raw);

    return .{ first, second, third };
}

/// Builds an orthonormal basis around a normal.
///
/// A world axis that is least aligned with the normal
/// seeds the tangent, so the basis is stable for
/// every direction.
///
/// # Args
/// - `normal`: the desired third basis vector.
/// - `epsilon`: smallest accepted length.
///
/// # Return
/// `.{ tangent, bitangent, normal }` with a unit
/// normal, or `null` when `normal` is too short.
///
/// # Note
/// The tangent direction depends on the input
/// direction; do not rely on it being continuous
/// across the whole sphere.
///
/// # Compile Error
/// - `normal` is not a three-component float
///   vector.
pub fn orthonormalBasis(
    normal: anytype,
    epsilon: anytype,
) ?[3]@TypeOf(normal) {
    const V = @TypeOf(normal);
    switch (@typeInfo(V)) {
        .vector => |info| {
            if (info.len != 3) {
                @compileError(
                    "orthonormalBasis: `normal` must have " ++
                        "3 components",
                );
            }
            switch (@typeInfo(info.child)) {
                .float => {},
                else => @compileError(
                    "orthonormalBasis: `normal` must hold " ++
                        "floats",
                ),
            }
        },
        else => @compileError(
            "orthonormalBasis: `normal` must be a vector",
        ),
    }
    if (!scalar.isNormalizable(normal, epsilon)) return null;

    const unit = vec.normalize(normal);
    const abs_x = @abs(unit[0]);
    const abs_y = @abs(unit[1]);
    const abs_z = @abs(unit[2]);
    const helper: V = if (abs_x <= abs_y and abs_x <= abs_z)
        .{ 1, 0, 0 }
    else if (abs_y <= abs_z)
        .{ 0, 1, 0 }
    else
        .{ 0, 0, 1 };

    const tangent = vec.normalize(vec.cross(helper, unit));
    return .{ tangent, vec.cross(unit, tangent), unit };
}

test "orthonormal basis" {
    const normals = [_]Vec3(f32){
        .{ 0, 0, 1 },
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0.3, -0.4, 0.5 },
    };
    for (normals) |normal| {
        const frame = orthonormalBasis(normal, 1e-6).?;
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            vec.length(frame[0]),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            @as(f32, 0),
            vec.dot(frame[0], frame[1]),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            @as(f32, 0),
            vec.dot(frame[0], frame[2]),
            1e-6,
        );
        try std.testing.expectApproxEqAbs(
            @as(f32, 1),
            vec.length(frame[2]),
            1e-6,
        );
        const expected = vec.normalize(normal);
        try std.testing.expectApproxEqAbs(
            expected[0],
            frame[2][0],
            1e-6,
        );
    }
    try std.testing.expect(
        orthonormalBasis(Vec3(f32){ 0, 0, 0 }, 1e-6) == null,
    );
}

test "gram schmidt" {
    const values = [3]Vec3(f32){
        .{ 1, 0, 0 },
        .{ 1, 1, 0 },
        .{ 0, 1, 1 },
    };
    const frame = gramSchmidt(values, 1e-6).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        vec.length(frame[0]),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        vec.dot(frame[0], frame[1]),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        vec.dot(frame[1], frame[2]),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        vec.length(frame[2]),
        1e-6,
    );

    const degenerate = [3]Vec3(f32){
        .{ 1, 0, 0 },
        .{ 2, 0, 0 },
        .{ 0, 0, 1 },
    };
    try std.testing.expect(
        gramSchmidt(degenerate, 1e-6) == null,
    );
}
