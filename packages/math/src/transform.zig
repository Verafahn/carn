//! Coordinate-space transforms built on matrices.
//!
//! Points transform with a homogeneous `w` of one,
//! directions with zero, and normals with the inverse
//! transpose. The helpers accept `3x3` matrices,
//! `4x4` matrices, and the compact `4x3` affine form.

const std = @import("std");
const scalar = @import("scalar.zig");
const mat = @import("mat.zig");

/// Transforms a point by a matrix.
///
/// # Args
/// - `m`: a `3x3`, `4x4`, or `4x3` matrix.
/// - `point`: a three-component point.
///
/// # Return
/// The transformed point. A `4x4` matrix divides by
/// the homogeneous `w` when it is not zero.
///
/// # Note
/// A `4x3` matrix uses an implicit `w` of one, so it
/// applies the translation in its last row.
///
/// # Compile Error
/// - `m` is not a supported matrix size.
/// - `point` is not a three-component vector.
pub fn transformPoint(m: anytype, point: anytype) @TypeOf(point) {
    const M = @TypeOf(m);
    const T = scalar.ScalarOf(@TypeOf(point));
    if (M.row_count == 3 and M.column_count == 3) {
        return mat.mulVec(m, point);
    }
    if (M.row_count == 4 and M.column_count == 3) {
        return mat.mulVec(
            m,
            @Vector(4, T){ point[0], point[1], point[2], 1 },
        );
    }
    if (M.row_count == 4 and M.column_count == 4) {
        const clip = mat.mulVec(
            m,
            @Vector(4, T){ point[0], point[1], point[2], 1 },
        );
        if (clip[3] == 0) {
            return .{ clip[0], clip[1], clip[2] };
        }
        const inverse = 1 / clip[3];
        return .{
            clip[0] * inverse,
            clip[1] * inverse,
            clip[2] * inverse,
        };
    }
    @compileError(
        "transformPoint: expected a 3x3, 4x4, or 4x3 matrix",
    );
}

/// Transforms a direction by a matrix.
///
/// # Args
/// - `m`: a `3x3`, `4x4`, or `4x3` matrix.
/// - `value`: a three-component direction.
///
/// # Return
/// The transformed direction, without translation.
///
/// # Note
/// The `w` component is zero, so the last matrix row
/// is ignored. Translations therefore leave
/// directions unchanged.
///
/// # Compile Error
/// - `m` is not a supported matrix size.
/// - `value` is not a three-component vector.
pub fn transformDirection(m: anytype, value: anytype) @TypeOf(value) {
    const M = @TypeOf(m);
    if (M.row_count == 3 and M.column_count == 3) {
        return mat.mulVec(m, value);
    }
    if (M.row_count == 4 and
        (M.column_count == 3 or M.column_count == 4))
    {
        const T = scalar.ScalarOf(@TypeOf(value));
        const result = mat.mulVec(
            m,
            @Vector(4, T){ value[0], value[1], value[2], 0 },
        );
        return .{ result[0], result[1], result[2] };
    }
    @compileError(
        "transformDirection: expected a 3x3, 4x4, or 4x3 matrix",
    );
}

/// Transforms a normal by a matrix.
///
/// # Args
/// - `m`: a `3x3`, `4x4`, or `4x3` matrix.
/// - `normal`: the normal to transform.
/// - `epsilon`: smallest accepted determinant of
///   the linear part.
///
/// # Return
/// The transformed normal, or `null` when the
/// linear part is not invertible.
///
/// # Note
/// Normals use the inverse transpose so they stay
/// perpendicular under non-uniform scale. The result
/// is not normalized.
///
/// # Compile Error
/// - `m` is not a supported matrix size.
/// - `normal` is not a three-component float
///   vector.
pub fn transformNormal(
    m: anytype,
    normal: anytype,
    epsilon: anytype,
) ?@TypeOf(normal) {
    const linear = linearPart(m);
    const normal_matrix = mat.normalMatrix(linear, epsilon) orelse
        return null;
    return mat.mulVec(normal_matrix, normal);
}

/// Transforms a point from local to world space.
///
/// # Args
/// - `local_to_world`: the local-to-world matrix.
/// - `point`: a point in local space.
///
/// # Return
/// The point in world space.
///
/// # Note
/// This is the same operation as `transformPoint`.
pub fn localToWorld(m: anytype, point: anytype) @TypeOf(point) {
    return transformPoint(m, point);
}

/// Transforms a point from world to local space.
///
/// # Args
/// - `local_to_world`: the square local-to-world
///   matrix.
/// - `point`: a point in world space.
/// - `epsilon`: smallest accepted determinant.
///
/// # Return
/// The point in local space, or `null` when the
/// matrix is not invertible.
///
/// # Compile Error
/// - The matrix is not square.
pub fn worldToLocal(
    m: anytype,
    point: anytype,
    epsilon: anytype,
) ?@TypeOf(point) {
    const inverse_matrix = mat.inverse(m, epsilon) orelse
        return null;
    return transformPoint(inverse_matrix, point);
}

fn linearPart(m: anytype) mat.Mat3(
    scalar.ScalarOf(@TypeOf(m)),
) {
    const M = @TypeOf(m);
    const T = scalar.ScalarOf(M);
    if (M.row_count == 3 and M.column_count == 3) {
        return m;
    }
    if ((M.row_count == 4 and M.column_count == 3) or
        (M.row_count == 4 and M.column_count == 4))
    {
        return mat.Mat3(T).fromRows(.{
            .{ m.columns[0][0], m.columns[1][0], m.columns[2][0] },
            .{ m.columns[0][1], m.columns[1][1], m.columns[2][1] },
            .{ m.columns[0][2], m.columns[1][2], m.columns[2][2] },
        });
    }
    @compileError(
        "linearPart: expected a 3x3, 4x4, or 4x3 matrix",
    );
}

test "transform point" {
    const Vec3f = @Vector(3, f32);
    const matrix = mat.Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 5, 6, 7, 1 },
    });
    const result = transformPoint(matrix, Vec3f{ 1, 2, 3 });
    try std.testing.expectApproxEqAbs(@as(f32, 6), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10), result[2], 1e-6);

    const compact = mat.Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const compact_result = transformPoint(
        compact,
        Vec3f{ 1, 2, 3 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 6),
        compact_result[0],
        1e-6,
    );

    const linear = mat.Mat3(f32).fromRows(.{
        .{ 2, 0, 0 },
        .{ 0, 2, 0 },
        .{ 0, 0, 2 },
    });
    const scaled = transformPoint(linear, Vec3f{ 1, 1, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 2), scaled[0], 1e-6);
}

test "transform direction" {
    const Vec3f = @Vector(3, f32);
    const matrix = mat.Mat4(f32).fromRows(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 2, 0 },
        .{ 5, 6, 7, 1 },
    });
    const direction = transformDirection(matrix, Vec3f{ 1, 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 2), direction[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), direction[1], 1e-6);
}

test "transform normal" {
    const Vec3f = @Vector(3, f32);
    const matrix = mat.Mat4(f32).fromRows(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    });
    const normal = transformNormal(
        matrix,
        Vec3f{ 1, 0, 0 },
        1e-9,
    ).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), normal[0], 1e-6);

    // A stretched plane keeps its transformed normal
    // perpendicular to transformed in-plane edges.
    const edge_a = Vec3f{ 1, 0, 0 };
    const edge_b = Vec3f{ 0, 1, 0 };
    const plane_normal = Vec3f{ 0, 0, 1 };
    const transformed_normal = transformNormal(
        matrix,
        plane_normal,
        1e-9,
    ).?;
    const transformed_a = transformDirection(matrix, edge_a);
    const transformed_b = transformDirection(matrix, edge_b);
    const dot_a = transformed_normal[0] * transformed_a[0] +
        transformed_normal[1] * transformed_a[1] +
        transformed_normal[2] * transformed_a[2];
    const dot_b = transformed_normal[0] * transformed_b[0] +
        transformed_normal[1] * transformed_b[1] +
        transformed_normal[2] * transformed_b[2];
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot_a, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot_b, 1e-6);

    const singular = mat.Mat3(f32).fromRows(.{
        .{ 1, 0, 0 },
        .{ 0, 0, 0 },
        .{ 0, 0, 1 },
    });
    try std.testing.expect(
        transformNormal(singular, plane_normal, 1e-9) == null,
    );
}

test "local and world" {
    const Vec3f = @Vector(3, f32);
    const matrix = mat.Mat4(f32).fromRows(.{
        .{ 0, -1, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 4, 5, 6, 1 },
    });
    const local = Vec3f{ 3, 2, 1 };
    const world = localToWorld(matrix, local);
    try std.testing.expectApproxEqAbs(@as(f32, 6), world[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), world[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7), world[2], 1e-6);

    const back = worldToLocal(matrix, world, 1e-9).?;
    try std.testing.expectApproxEqAbs(local[0], back[0], 1e-5);
    try std.testing.expectApproxEqAbs(local[1], back[1], 1e-5);
    try std.testing.expectApproxEqAbs(local[2], back[2], 1e-5);
}

test "perspective divide" {
    const projection = mat.Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, -1 },
        .{ 0, 0, 2, 0 },
    });
    const result = transformPoint(
        projection,
        @Vector(3, f32){ 2, 0, -2 },
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), result[2], 1e-6);
}
