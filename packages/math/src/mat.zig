//! Matrix types built on column-major storage.
//!
//! A matrix stores one `@Vector(M, T)` per column.
//! Vectors are row matrices, so a transform applies
//! as `v * m` and is written `mat.mulVec(m, v)`.
//!
//! Affine transforms keep the translation in the last
//! row, which is what row vectors require.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;
const quat = @import("quat.zig");

/// Returns a column-major matrix type.
///
/// # Args
/// - `T`: an integer or float component type.
/// - `M`: the row count; at least 1.
/// - `N`: the column count; at least 1.
///
/// # Return
/// A struct whose `columns` field holds `N`
/// vectors of `M` components.
///
/// # Compile Error
/// - `M` or `N` is zero.
/// - `T` is not an integer or float type.
pub fn Matrix(comptime T: type, comptime M: usize, comptime N: usize) type {
    if (M == 0 or N == 0) {
        @compileError("Matrix: `M` and `N` must be at least 1");
    }
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError(
            "Matrix: `T` must be an integer or float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// One vector per column.
        columns: [N]@Vector(M, T),

        const Self = @This();

        /// Number of rows.
        pub const row_count: usize = M;
        /// Number of columns.
        pub const column_count: usize = N;
        /// Component type.
        pub const Component = T;
        /// One column vector.
        pub const Column = @Vector(M, T);
        /// One row vector.
        pub const Row = @Vector(N, T);

        /// Returns the multiplicative identity.
        ///
        /// # Return
        /// A square matrix with ones on the diagonal.
        ///
        /// # Compile Error
        /// - The matrix is not square.
        pub fn identity() Self {
            if (M != N) {
                @compileError(
                    "identity: the matrix must be square",
                );
            }
            var result: Self = undefined;
            inline for (0..N) |c| {
                var value_column: Column = @splat(0);
                value_column[c] = 1;
                result.columns[c] = value_column;
            }
            return result;
        }

        /// Builds a matrix from columns.
        ///
        /// # Args
        /// - `values`: exactly `N` columns.
        ///
        /// # Return
        /// A matrix holding the columns in order.
        pub fn fromColumns(values: [N]Column) Self {
            return .{ .columns = values };
        }

        /// Builds a matrix from rows.
        ///
        /// # Args
        /// - `values`: exactly `M` rows.
        ///
        /// # Return
        /// A matrix holding the rows in order.
        pub fn fromRows(values: [M]Row) Self {
            var result: Self = undefined;
            inline for (0..M) |r| {
                inline for (0..N) |c| {
                    result.columns[c][r] = values[r][c];
                }
            }
            return result;
        }

        /// Builds a matrix with one value everywhere.
        ///
        /// # Args
        /// - `value`: the value of every component.
        ///
        /// # Return
        /// A matrix filled with `value`.
        pub fn fromElement(value: T) Self {
            var result: Self = undefined;
            const value_column: Column = @splat(value);
            for (0..N) |c| result.columns[c] = value_column;
            return result;
        }

        /// Reads one component.
        ///
        /// # Args
        /// - `row`: the row index.
        /// - `column`: the column index.
        ///
        /// # Return
        /// The component at `(row, column)`.
        ///
        /// # Compile Error
        /// - An index is out of range.
        pub fn at(
            self: Self,
            comptime row_index: usize,
            comptime column_index: usize,
        ) T {
            if (row_index >= M or column_index >= N) {
                @compileError("at: index out of range");
            }
            return self.columns[column_index][row_index];
        }

        /// Returns one column.
        ///
        /// # Args
        /// - `index`: a valid column index.
        ///
        /// # Return
        /// The column vector.
        ///
        /// # Panic
        /// In safety-checked builds, `index` is at least
        /// `column_count`.
        pub fn column(self: Self, index: usize) Column {
            std.debug.assert(index < N);
            return self.columns[index];
        }

        /// Returns one row.
        ///
        /// # Args
        /// - `index`: the row index.
        ///
        /// # Return
        /// The row as a vector.
        ///
        /// # Compile Error
        /// - `index` is at least `row_count`.
        pub fn row(self: Self, comptime index: usize) Row {
            if (index >= M) {
                @compileError("row: index out of range");
            }
            var result: Row = undefined;
            inline for (0..N) |c| result[c] = self.columns[c][index];
            return result;
        }
    };
}

/// Returns a two-by-two matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// A `2x2` matrix.
pub fn Mat2(comptime T: type) type {
    return Matrix(T, 2, 2);
}

/// Returns a three-by-three matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// A `3x3` matrix.
pub fn Mat3(comptime T: type) type {
    return Matrix(T, 3, 3);
}

/// Returns a four-by-four matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// A `4x4` matrix.
pub fn Mat4(comptime T: type) type {
    return Matrix(T, 4, 4);
}

/// Returns a compact affine matrix of `T`.
///
/// The type holds four rows and three columns, which
/// is what a homogeneous row vector needs: the last
/// row stores the translation and the result drops
/// the constant `w` component.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// A `4x3` matrix.
pub fn Mat3x4(comptime T: type) type {
    return Matrix(T, 4, 3);
}

/// Adds two matrices.
///
/// # Args
/// - `a`: first matrix.
/// - `b`: matrix of the same type.
///
/// # Return
/// The componentwise sum.
pub fn add(a: anytype, b: anytype) @TypeOf(a) {
    const M = @TypeOf(a);
    const other: M = b;
    var result: M = undefined;
    for (0..M.column_count) |c| {
        result.columns[c] = a.columns[c] + other.columns[c];
    }
    return result;
}

/// Subtracts two matrices.
///
/// # Args
/// - `a`: first matrix.
/// - `b`: matrix of the same type.
///
/// # Return
/// The componentwise difference.
pub fn subtract(a: anytype, b: anytype) @TypeOf(a) {
    const M = @TypeOf(a);
    const other: M = b;
    var result: M = undefined;
    for (0..M.column_count) |c| {
        result.columns[c] = a.columns[c] - other.columns[c];
    }
    return result;
}

/// Negates every component.
///
/// # Args
/// - `m`: the matrix to negate.
///
/// # Return
/// The componentwise negation.
pub fn negate(m: anytype) @TypeOf(m) {
    const M = @TypeOf(m);
    var result: M = undefined;
    for (0..M.column_count) |c| {
        result.columns[c] = -m.columns[c];
    }
    return result;
}

/// Multiplies two matrices.
///
/// # Args
/// - `a`: left matrix.
/// - `b`: right matrix; its row count must match
///   the column count of `a`.
///
/// # Return
/// The matrix product `a * b`.
///
/// # Compile Error
/// - The dimensions do not line up.
/// - The component types differ.
pub fn multiply(a: anytype, b: anytype) MulType(
    @TypeOf(a),
    @TypeOf(b),
) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    const Result = MulType(A, B);
    const inner = A.column_count;
    var result: Result = undefined;
    inline for (0..Result.column_count) |c| {
        var column: Result.Column = @splat(0);
        inline for (0..inner) |k| {
            const weight = b.columns[c][k];
            column += a.columns[k] *
                @as(Result.Column, @splat(weight));
        }
        result.columns[c] = column;
    }
    return result;
}

/// Transposes a matrix.
///
/// # Args
/// - `m`: the matrix to transpose.
///
/// # Return
/// The transposed matrix with swapped dimensions.
pub fn transpose(m: anytype) Matrix(
    scalar.ScalarOf(@TypeOf(m)),
    columnCount(@TypeOf(m)),
    rowCount(@TypeOf(m)),
) {
    const M = @TypeOf(m);
    const Result = Matrix(
        scalar.ScalarOf(M),
        columnCount(M),
        rowCount(M),
    );
    var result: Result = undefined;
    inline for (0..M.row_count) |r| {
        inline for (0..M.column_count) |c| {
            result.columns[r][c] = m.columns[c][r];
        }
    }
    return result;
}

/// Computes the determinant of a square matrix.
///
/// # Args
/// - `m`: a square matrix of size 1 to 4.
///
/// # Return
/// The signed determinant.
///
/// # Compile Error
/// - The matrix is not square.
/// - The matrix is larger than `4x4`.
pub fn determinant(m: anytype) scalar.ScalarOf(@TypeOf(m)) {
    const M = @TypeOf(m);
    const T = scalar.ScalarOf(M);
    const n = M.row_count;
    if (n != M.column_count) {
        @compileError("determinant: matrix must be square");
    }
    if (n == 1) return m.columns[0][0];
    if (n == 2) {
        return m.columns[0][0] * m.columns[1][1] -
            m.columns[1][0] * m.columns[0][1];
    }
    if (n == 3) {
        return vec.dot(
            m.columns[0],
            vec.cross(m.columns[1], m.columns[2]),
        );
    }
    if (n == 4) {
        var sum: T = 0;
        inline for (0..4) |c| {
            const sign: T = comptime if (c % 2 == 0) 1 else -1;
            sum += sign * m.columns[c][0] *
                determinant(minorOf(m, 0, c));
        }
        return sum;
    }
    @compileError("determinant: supports 1x1 to 4x4 matrices");
}

/// Computes the adjugate of a square matrix.
///
/// The adjugate is the transposed cofactor matrix.
/// Multiplying a matrix by its adjugate yields the
/// determinant times the identity.
///
/// # Args
/// - `m`: a square matrix of size 1 to 4.
///
/// # Return
/// The adjugate matrix.
///
/// # Compile Error
/// - The matrix is not square.
/// - The matrix is larger than `4x4`.
pub fn adjugate(m: anytype) @TypeOf(m) {
    const M = @TypeOf(m);
    const T = scalar.ScalarOf(M);
    const n = M.row_count;
    if (n != M.column_count) {
        @compileError("adjugate: matrix must be square");
    }
    var result: M = undefined;
    if (n == 1) {
        result.columns[0][0] = 1;
        return result;
    }
    if (n > 4) {
        @compileError("adjugate: supports 1x1 to 4x4 matrices");
    }
    inline for (0..n) |r| {
        inline for (0..n) |c| {
            const sign: T = comptime if ((r + c) % 2 == 0) 1 else -1;
            result.columns[c][r] = sign *
                determinant(minorOf(m, c, r));
        }
    }
    return result;
}

/// Inverts a square matrix.
///
/// # Args
/// - `m`: a square float matrix of size 1 to 4.
/// - `epsilon`: smallest accepted `abs(det)`.
///
/// # Return
/// The inverse, or `null` when the determinant is
/// too close to zero.
///
/// # Note
/// The check uses the determinant, so badly
/// conditioned matrices may still produce large
/// errors.
///
/// # Compile Error
/// - The component type is not a float.
/// - The matrix is not square or larger than
///   `4x4`.
pub fn inverse(m: anytype, epsilon: anytype) ?@TypeOf(m) {
    const M = @TypeOf(m);
    assertFloatMatrix(M);
    const det = determinant(m);
    if (scalar.isZero(det, epsilon)) return null;

    const adj = adjugate(m);
    var result: M = undefined;
    for (0..M.column_count) |c| {
        result.columns[c] =
            adj.columns[c] / @as(M.Column, @splat(det));
    }
    return result;
}

/// Computes the normal transformation matrix.
///
/// # Args
/// - `m`: a square float matrix of size 1 to 4.
/// - `epsilon`: smallest accepted `abs(det)`.
///
/// # Return
/// `transpose(inverse(m))`, or `null` when the
/// matrix is not invertible.
///
/// # Note
/// Normals transform with the inverse transpose, so
/// this is the matrix to use instead of `m` itself.
///
/// # Compile Error
/// - The component type is not a float.
/// - The matrix is not square.
pub fn normalMatrix(m: anytype, epsilon: anytype) ?@TypeOf(m) {
    assertFloatMatrix(@TypeOf(m));
    return inverse(transpose(m), epsilon);
}

/// Multiplies a row vector by a matrix.
///
/// # Args
/// - `m`: the matrix.
/// - `v`: a row vector whose length matches the
///   matrix row count.
///
/// # Return
/// A row vector of length `m.column_count`.
///
/// # Compile Error
/// - `v` is not a vector of the right length.
/// - The component types differ.
pub fn mulVec(m: anytype, v: anytype) VectorFor(@TypeOf(m)) {
    const M = @TypeOf(m);
    const V = @TypeOf(v);
    switch (@typeInfo(V)) {
        .vector => |info| {
            if (info.len != M.row_count) {
                @compileError(
                    "mulVec: vector length must match " ++
                        "the matrix row count",
                );
            }
        },
        else => @compileError("mulVec: `v` must be a vector"),
    }
    var result: VectorFor(M) = undefined;
    inline for (0..M.column_count) |c| {
        result[c] = vec.dot(v, m.columns[c]);
    }
    return result;
}

/// Expands a compact affine matrix to `4x4`.
///
/// # Args
/// - `m`: a `4x3` affine matrix.
///
/// # Return
/// The same transform as a `4x4` matrix.
///
/// # Compile Error
/// - `m` is not `4x3`.
pub fn toMat4(m: anytype) Matrix(
    scalar.ScalarOf(@TypeOf(m)),
    4,
    4,
) {
    const M = @TypeOf(m);
    const T = scalar.ScalarOf(M);
    if (M.row_count != 4 or M.column_count != 3) {
        @compileError("toMat4: `m` must be a 4x3 matrix");
    }
    const translation: @Vector(4, T) = .{ 0, 0, 0, 1 };
    return .{ .columns = .{
        m.columns[0],
        m.columns[1],
        m.columns[2],
        translation,
    } };
}

/// Compacts an affine `4x4` matrix to `4x3`.
///
/// # Args
/// - `m`: a `4x4` affine matrix.
///
/// # Return
/// The first three columns of `m`.
///
/// # Note
/// The fourth column is dropped, so it should be
/// `(0, 0, 0, 1)` for an affine transform.
///
/// # Compile Error
/// - `m` is not `4x4`.
pub fn fromMat4(m: anytype) Matrix(
    scalar.ScalarOf(@TypeOf(m)),
    4,
    3,
) {
    const M = @TypeOf(m);
    if (M.row_count != 4 or M.column_count != 4) {
        @compileError("fromMat4: `m` must be a 4x4 matrix");
    }
    return .{ .columns = .{
        m.columns[0],
        m.columns[1],
        m.columns[2],
    } };
}

/// The parts of an affine transform.
///
/// # Args
/// - `T`: the float component type.
pub fn Decomposition(comptime T: type) type {
    return struct {
        /// Translation part.
        translation: @Vector(3, T),
        /// Rotation part; a unit quaternion.
        rotation: quat.Quat(T),
        /// Scale along the local axes.
        scale: @Vector(3, T),
    };
}

/// Splits a `4x4` affine matrix into its parts.
///
/// # Args
/// - `m`: a `4x4` affine float matrix with no
///   shear; it must equal `S * R * T`.
/// - `epsilon`: smallest accepted row length.
///
/// # Return
/// The translation, rotation, and scale, or `null`
/// when a scale component is too small.
///
/// # Note
/// Shear is not represented, so a sheared matrix
/// produces an approximate rotation. A negative
/// determinant moves the sign onto `scale.x`.
///
/// # Compile Error
/// - `m` is not `4x4` or not a float matrix.
pub fn decompose(
    m: anytype,
    epsilon: anytype,
) ?Decomposition(scalar.ScalarOf(@TypeOf(m))) {
    const M = @TypeOf(m);
    assertFloatMatrix(M);
    if (M.row_count != 4 or M.column_count != 4) {
        @compileError("decompose: `m` must be a 4x4 matrix");
    }
    const T = scalar.ScalarOf(M);
    const translation: @Vector(3, T) = .{
        m.columns[0][3],
        m.columns[1][3],
        m.columns[2][3],
    };

    var rows: [3]@Vector(3, T) = undefined;
    inline for (0..3) |r| {
        rows[r] = .{
            m.columns[0][r],
            m.columns[1][r],
            m.columns[2][r],
        };
    }

    var scales: @Vector(3, T) = undefined;
    inline for (0..3) |r| scales[r] = vec.length(rows[r]);
    inline for (0..3) |r| {
        if (!scalar.isNormalizable(rows[r], epsilon)) {
            return null;
        }
        rows[r] = vec.normalize(rows[r]);
    }

    var rotation = Mat3(T).fromRows(rows);
    if (determinant(rotation) < 0) {
        scales[0] = -scales[0];
        rows[0] = -rows[0];
        rotation = Mat3(T).fromRows(rows);
    }

    return .{
        .translation = translation,
        .rotation = quat.fromMat3(rotation),
        .scale = scales,
    };
}

fn MulType(comptime A: type, comptime B: type) type {
    if (A.column_count != B.row_count) {
        @compileError(
            "multiply: left column count must match " ++
                "right row count",
        );
    }
    const T = scalar.ScalarOf(A);
    if (T != scalar.ScalarOf(B)) {
        @compileError(
            "multiply: matrices must share a component " ++
                "type",
        );
    }
    return Matrix(T, A.row_count, B.column_count);
}

fn MinorType(comptime M: type) type {
    return Matrix(
        scalar.ScalarOf(M),
        rowCount(M) - 1,
        columnCount(M) - 1,
    );
}

fn VectorFor(comptime M: type) type {
    return @Vector(
        columnCount(M),
        scalar.ScalarOf(M),
    );
}

fn rowCount(comptime M: type) usize {
    return M.row_count;
}

fn columnCount(comptime M: type) usize {
    return M.column_count;
}

fn minorOf(
    m: anytype,
    comptime skip_row: usize,
    comptime skip_column: usize,
) MinorType(@TypeOf(m)) {
    const M = @TypeOf(m);
    const Result = MinorType(M);
    var result: Result = undefined;
    inline for (0..Result.column_count) |c| {
        const source_column = comptime blk: {
            var source = c;
            if (source >= skip_column) source += 1;
            break :blk source;
        };
        var column: Result.Column = @splat(0);
        inline for (0..Result.row_count) |r| {
            const source_row = comptime blk: {
                var source = r;
                if (source >= skip_row) source += 1;
                break :blk source;
            };
            column[r] = m.columns[source_column][source_row];
        }
        result.columns[c] = column;
    }
    return result;
}

fn assertFloatMatrix(comptime M: type) void {
    switch (@typeInfo(scalar.ScalarOf(M))) {
        .float => {},
        else => @compileError(
            "expected a float matrix, got `" ++
                @typeName(M) ++ "`",
        ),
    }
}

test "matrix construction" {
    const M = Mat2(f32);
    const identity = M.identity();
    try std.testing.expectEqual(@as(f32, 1), identity.at(0, 0));
    try std.testing.expectEqual(@as(f32, 0), identity.at(0, 1));

    const from_rows = M.fromRows(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });
    try std.testing.expectEqual(@as(f32, 1), from_rows.at(0, 0));
    try std.testing.expectEqual(@as(f32, 2), from_rows.at(0, 1));
    try std.testing.expectEqual(@as(f32, 3), from_rows.at(1, 0));
    try std.testing.expectEqual(@as(f32, 4), from_rows.at(1, 1));
    try std.testing.expectEqual(@as(f32, 3), from_rows.column(0)[1]);
    const row = from_rows.row(1);
    try std.testing.expectEqual(@as(f32, 3), row[0]);
    try std.testing.expectEqual(@as(f32, 4), row[1]);

    const filled = Mat2(i32).fromElement(7);
    try std.testing.expectEqual(@as(i32, 7), filled.at(1, 1));
}

test "matrix arithmetic" {
    const M = Mat2(f32);
    const a = M.fromRows(.{ .{ 1, 2 }, .{ 3, 4 } });
    const b = M.fromRows(.{ .{ 5, 6 }, .{ 7, 8 } });
    const sum = add(a, b);
    try std.testing.expectEqual(@as(f32, 6), sum.at(0, 0));
    try std.testing.expectEqual(@as(f32, 12), sum.at(1, 1));
    const difference = subtract(b, a);
    try std.testing.expectEqual(@as(f32, 4), difference.at(0, 0));
    const negated = negate(a);
    try std.testing.expectEqual(@as(f32, -1), negated.at(0, 0));

    const product = multiply(a, b);
    try std.testing.expectEqual(@as(f32, 19), product.at(0, 0));
    try std.testing.expectEqual(@as(f32, 22), product.at(0, 1));
    try std.testing.expectEqual(@as(f32, 43), product.at(1, 0));
    try std.testing.expectEqual(@as(f32, 50), product.at(1, 1));
}

test "matrix multiply shapes" {
    const A = Matrix(f32, 2, 3);
    const B = Matrix(f32, 3, 2);
    const a = A.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const b = B.fromRows(.{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    });
    const product = multiply(a, b);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(product).row_count);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(product).column_count);
    try std.testing.expectEqual(@as(f32, 58), product.at(0, 0));
    try std.testing.expectEqual(@as(f32, 154), product.at(1, 1));
}

test "matrix transpose" {
    const A = Matrix(f32, 2, 3);
    const a = A.fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    });
    const transposed = transpose(a);
    try std.testing.expectEqual(@as(usize, 3), @TypeOf(transposed).row_count);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(transposed).column_count);
    try std.testing.expectEqual(@as(f32, 1), transposed.at(0, 0));
    try std.testing.expectEqual(@as(f32, 4), transposed.at(0, 1));
    try std.testing.expectEqual(@as(f32, 6), transposed.at(2, 1));
}

test "matrix determinant" {
    const a = Mat2(f32).fromRows(.{ .{ 1, 2 }, .{ 3, 4 } });
    try std.testing.expectApproxEqAbs(@as(f32, -2), determinant(a), 1e-6);

    const b = Mat3(f32).fromRows(.{
        .{ 6, 1, 1 },
        .{ 4, -2, 5 },
        .{ 2, 8, 7 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, -306), determinant(b), 1e-4);

    const c = Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 3, 0 },
        .{ 0, 0, 0, 4 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, 24), determinant(c), 1e-4);
}

test "matrix inverse" {
    const cases = [_]Mat3(f32){
        Mat3(f32).fromRows(.{
            .{ 1, 2, 3 },
            .{ 0, 1, 4 },
            .{ 5, 6, 0 },
        }),
        Mat3(f32).fromRows(.{
            .{ 2, 0, 0 },
            .{ 0, 4, 0 },
            .{ 0, 0, 8 },
        }),
        Mat3(f32).identity(),
    };
    for (cases) |matrix| {
        const inverted = inverse(matrix, 1e-6).?;
        const product = multiply(matrix, inverted);
        inline for (0..3) |r| {
            inline for (0..3) |c| {
                const expected: f32 = if (r == c) 1 else 0;
                try std.testing.expectApproxEqAbs(
                    expected,
                    product.at(r, c),
                    1e-5,
                );
            }
        }
    }

    const singular = Mat2(f32).fromRows(.{ .{ 1, 2 }, .{ 2, 4 } });
    try std.testing.expect(inverse(singular, 1e-6) == null);

    const four = Mat4(f32).fromRows(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 4, 0, 0 },
        .{ 0, 0, 8, 0 },
        .{ 1, 2, 3, 1 },
    });
    const inverted_four = inverse(four, 1e-6).?;
    const four_product = multiply(four, inverted_four);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        four_product.at(3, 3),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        four_product.at(3, 0),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        inverted_four.at(0, 0),
        1e-5,
    );
}

test "normal matrix" {
    const matrix = Mat3(f32).fromRows(.{
        .{ 2, 0, 0 },
        .{ 0, 4, 0 },
        .{ 0, 0, 8 },
    });
    const normals = normalMatrix(matrix, 1e-6).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        normals.at(0, 0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        normals.at(1, 1),
        1e-6,
    );

    const rotation = quat.toMat3(quat.fromAxisAngle(
        vec.normalize(Vec3(f32){ 1, 2, 3 }),
        0.8,
    ));
    const rotation_normals = normalMatrix(rotation, 1e-6).?;
    inline for (0..3) |r| {
        inline for (0..3) |c| {
            try std.testing.expectApproxEqAbs(
                rotation.at(r, c),
                rotation_normals.at(r, c),
                1e-5,
            );
        }
    }
}

test "matrix vector product" {
    const matrix = Mat3(f32).fromRows(.{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    });
    const result = mulVec(matrix, Vec3(f32){ 1, 1, 1 });
    try std.testing.expectEqual(@as(f32, 12), result[0]);
    try std.testing.expectEqual(@as(f32, 15), result[1]);
    try std.testing.expectEqual(@as(f32, 18), result[2]);

    const affine = Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const point = mulVec(affine, @Vector(4, f32){ 1, 2, 3, 1 });
    try std.testing.expectEqual(@as(f32, 6), point[0]);
    try std.testing.expectEqual(@as(f32, 8), point[1]);
    try std.testing.expectEqual(@as(f32, 10), point[2]);
}

test "matrix affine conversion" {
    const compact = Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const full = toMat4(compact);
    try std.testing.expectEqual(@as(f32, 5), full.at(3, 0));
    try std.testing.expectEqual(@as(f32, 1), full.at(3, 3));
    try std.testing.expectEqual(@as(f32, 0), full.at(0, 3));

    const back = fromMat4(full);
    inline for (0..4) |r| {
        inline for (0..3) |c| {
            try std.testing.expectEqual(
                compact.at(r, c),
                back.at(r, c),
            );
        }
    }
}

test "matrix decomposition" {
    const rotation = quat.fromAxisAngle(
        vec.normalize(Vec3(f32){ 1, 2, 3 }),
        0.6,
    );
    const r3 = quat.toMat3(rotation);
    const r4 = Mat4(f32).fromRows(.{
        .{ r3.at(0, 0), r3.at(0, 1), r3.at(0, 2), 0 },
        .{ r3.at(1, 0), r3.at(1, 1), r3.at(1, 2), 0 },
        .{ r3.at(2, 0), r3.at(2, 1), r3.at(2, 2), 0 },
        .{ 0, 0, 0, 1 },
    });
    const scale_matrix = Mat4(f32).fromRows(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 3, 0, 0 },
        .{ 0, 0, 4, 0 },
        .{ 0, 0, 0, 1 },
    });
    const translation = Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 5, 6, 7, 1 },
    });
    const combined = multiply(
        multiply(scale_matrix, r4),
        translation,
    );
    const parts = decompose(combined, 1e-5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 5),
        parts.translation[0],
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 6),
        parts.translation[1],
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 7),
        parts.translation[2],
        1e-5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 2), parts.scale[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3), parts.scale[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4), parts.scale[2], 1e-5);

    const rebuilt = quat.toMat3(parts.rotation);
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

test "matrix decomposition with negative scale" {
    const rotation = quat.fromAxisAngle(
        vec.normalize(Vec3(f32){ 0, 1, 0 }),
        0.4,
    );
    const r3 = quat.toMat3(rotation);
    const r4 = Mat4(f32).fromRows(.{
        .{ r3.at(0, 0), r3.at(0, 1), r3.at(0, 2), 0 },
        .{ r3.at(1, 0), r3.at(1, 1), r3.at(1, 2), 0 },
        .{ r3.at(2, 0), r3.at(2, 1), r3.at(2, 2), 0 },
        .{ 0, 0, 0, 1 },
    });
    const scale_matrix = Mat4(f32).fromRows(.{
        .{ -2, 0, 0, 0 },
        .{ 0, 3, 0, 0 },
        .{ 0, 0, 4, 0 },
        .{ 0, 0, 0, 1 },
    });
    const combined = multiply(scale_matrix, r4);
    const parts = decompose(combined, 1e-5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        parts.scale[0],
        1e-5,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 3), parts.scale[1], 1e-5);

    const rebuilt = quat.toMat3(parts.rotation);
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
