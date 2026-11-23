//! Column-major matrices.
//!
//! `Matrix(T, Rows, Cols)` stores one `@Vector` per
//! column. The one-row case is the `Vector` type, so
//! a vector is literally `Matrix(T, 1, N)`.
//!
//! Matrices are row-vector matrices: a row vector
//! multiplies from the left through `mulMatrix`, and
//! affine transforms keep their translation in the
//! last row. All operations are methods; operands
//! keep their exact types.

const std = @import("std");
const base = @import("../base/root.zig");
const vector = @import("vector.zig");
const decomposition = @import("decompose.zig");

/// Returns a column-major matrix type.
///
/// A matrix with one row is the `Vector` type; every
/// other shape uses the column-array representation.
///
/// # Args
/// - `T`: an integer or float component type.
/// - `Rows`: the row count; at least 1.
/// - `Cols`: the column count; at least 1.
///
/// # Return
/// `Vector(T, Cols)` when `Rows` is one, otherwise a
/// struct with a `columns` array.
///
/// # Note
/// `columns[c]` holds every row of column `c`.
///
/// # Compile Error
/// - `Rows` or `Cols` is zero.
/// - `T` is not an integer or float.
pub fn Matrix(
    comptime T: type,
    comptime Rows: usize,
    comptime Cols: usize,
) type {
    if (Rows == 0 or Cols == 0) {
        @compileError(
            "Matrix: `Rows` and `Cols` must be at least 1",
        );
    }
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError(
            "Matrix: `T` must be an integer or float, got `" ++
                @typeName(T) ++ "`",
        ),
    }
    if (Rows == 1) return vector.Vector(T, Cols);
    return Impl(T, Rows, Cols);
}

/// Returns a two-by-two matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `2x2` matrix type.
pub fn Mat2(comptime T: type) type {
    return Matrix(T, 2, 2);
}

/// Returns a three-by-three matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `3x3` matrix type.
pub fn Mat3(comptime T: type) type {
    return Matrix(T, 3, 3);
}

/// Returns a four-by-four matrix of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `4x4` matrix type.
pub fn Mat4(comptime T: type) type {
    return Matrix(T, 4, 4);
}

/// Returns a compact affine matrix of `T`.
///
/// The type holds four rows and three columns: a
/// homogeneous row vector multiplies into it and the
/// constant `w` component is dropped. The translation
/// lives in the last row.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `4x3` matrix type.
pub fn Mat3x4(comptime T: type) type {
    return Matrix(T, 4, 3);
}

fn Impl(
    comptime T: type,
    comptime Rows: usize,
    comptime Cols: usize,
) type {
    return struct {
        /// One vector per column.
        columns: [Cols]@Vector(Rows, T),

        const Self = @This();

        /// The component type.
        pub const Component = T;
        /// The number of rows.
        pub const row_count: usize = Rows;
        /// The number of columns.
        pub const column_count: usize = Cols;
        /// One raw column vector.
        pub const Column = @Vector(Rows, T);
        /// One typed row vector.
        pub const Row = vector.Vector(T, Cols);

        /// Returns the additive identity.
        ///
        /// # Return
        /// A matrix full of zeros.
        pub fn zero() Self {
            return Self.fromElement(0);
        }

        /// Returns the multiplicative identity.
        ///
        /// # Return
        /// A square matrix with ones on the diagonal.
        ///
        /// # Compile Error
        /// - The matrix is not square.
        pub fn identity() Self {
            if (Rows != Cols) {
                @compileError(
                    "identity: matrix must be square",
                );
            }
            var result: Self = undefined;
            inline for (0..Cols) |c| {
                var value_column: Column = @splat(0);
                value_column[c] = 1;
                result.columns[c] = value_column;
            }
            return result;
        }

        /// Builds a matrix from raw columns.
        ///
        /// # Args
        /// - `values`: exactly `Cols` columns.
        ///
        /// # Return
        /// A matrix holding the columns in order.
        pub fn fromColumns(values: [Cols]Column) Self {
            return .{ .columns = values };
        }

        /// Builds a matrix from typed rows.
        ///
        /// # Args
        /// - `values`: exactly `Rows` row vectors.
        ///
        /// # Return
        /// A matrix holding the rows in order.
        pub fn fromRows(values: [Rows]Row) Self {
            var result: Self = undefined;
            inline for (0..Rows) |r| {
                inline for (0..Cols) |c| {
                    result.columns[c][r] = values[r].elements[c];
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
            for (0..Cols) |c| result.columns[c] = value_column;
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
            if (row_index >= Rows or column_index >= Cols) {
                @compileError("at: index out of range");
            }
            return self.columns[column_index][row_index];
        }

        /// Writes one component.
        ///
        /// # Args
        /// - `row`: the row index.
        /// - `column`: the column index.
        /// - `value`: the new component value.
        ///
        /// # Compile Error
        /// - An index is out of range.
        pub fn set(
            self: *Self,
            comptime row_index: usize,
            comptime column_index: usize,
            value: T,
        ) void {
            if (row_index >= Rows or column_index >= Cols) {
                @compileError("set: index out of range");
            }
            self.columns[column_index][row_index] = value;
        }

        /// Returns one raw column.
        ///
        /// # Args
        /// - `index`: a valid column index.
        ///
        /// # Return
        /// The column storage.
        ///
        /// # Compile Error
        /// - `index` is at least `column_count`.
        pub fn column(self: Self, comptime index: usize) Column {
            if (index >= Cols) {
                @compileError("column: index out of range");
            }
            return self.columns[index];
        }

        /// Returns one row as a vector.
        ///
        /// # Args
        /// - `index`: a valid row index.
        ///
        /// # Return
        /// The row vector.
        ///
        /// # Compile Error
        /// - `index` is at least `row_count`.
        pub fn row(self: Self, comptime index: usize) Row {
            if (index >= Rows) {
                @compileError("row: index out of range");
            }
            var result: Row = undefined;
            inline for (0..Cols) |c| {
                result.elements[c] = self.columns[c][index];
            }
            return result;
        }

        /// Adds two matrices componentwise.
        ///
        /// # Args
        /// - `other`: the matrix to add.
        ///
        /// # Return
        /// The componentwise sum.
        pub fn add(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..Cols) |c| {
                result.columns[c] =
                    self.columns[c] + other.columns[c];
            }
            return result;
        }

        /// Subtracts two matrices componentwise.
        ///
        /// # Args
        /// - `other`: the matrix to subtract.
        ///
        /// # Return
        /// The componentwise difference.
        pub fn sub(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..Cols) |c| {
                result.columns[c] =
                    self.columns[c] - other.columns[c];
            }
            return result;
        }

        /// Negates every component.
        ///
        /// # Return
        /// The componentwise negation.
        pub fn negate(self: Self) Self {
            var result: Self = undefined;
            for (0..Cols) |c| {
                result.columns[c] = -self.columns[c];
            }
            return result;
        }

        /// Scales every component.
        ///
        /// # Args
        /// - `factor`: the multiplier.
        ///
        /// # Return
        /// The scaled matrix.
        pub fn scale(self: Self, factor: T) Self {
            var result: Self = undefined;
            const factor_column = @as(Column, @splat(factor));
            for (0..Cols) |c| {
                result.columns[c] =
                    self.columns[c] * factor_column;
            }
            return result;
        }

        /// Multiplies two square matrices of this type.
        ///
        /// # Args
        /// - `other`: the right operand.
        ///
        /// # Return
        /// The matrix product.
        ///
        /// # Note
        /// For a product with a different column count,
        /// use `mulTo`.
        ///
        /// # Compile Error
        /// - The matrix is not square.
        pub fn mul(self: Self, other: Self) Self {
            if (Rows != Cols) {
                @compileError("mul: matrix must be square");
            }
            return self.mulTo(Cols, other);
        }

        /// Multiplies by a matrix with `ResultCols`
        /// columns.
        ///
        /// # Args
        /// - `ResultCols`: the column count of the
        ///   right operand and of the result.
        /// - `other`: a matrix with `column_count` rows.
        ///
        /// # Return
        /// The matrix product with `ResultCols` columns.
        ///
        /// # Compile Error
        /// - `other` does not have `column_count` rows.
        pub fn mulTo(
            self: Self,
            comptime ResultCols: usize,
            other: Matrix(T, Cols, ResultCols),
        ) Matrix(T, Rows, ResultCols) {
            const Result = Matrix(T, Rows, ResultCols);
            var result: Result = undefined;
            if (Cols == 1) {
                inline for (0..ResultCols) |c| {
                    const factor = other.elements[c];
                    result.columns[c] =
                        self.columns[0] * @as(
                            Column,
                            @splat(factor),
                        );
                }
                return result;
            }
            inline for (0..ResultCols) |c| {
                var value_column: Column = @splat(0);
                inline for (0..Cols) |k| {
                    const weight = other.columns[c][k];
                    value_column += self.columns[k] * @as(
                        Column,
                        @splat(weight),
                    );
                }
                result.columns[c] = value_column;
            }
            return result;
        }

        /// Transposes the matrix.
        ///
        /// # Return
        /// The transposed matrix.
        ///
        /// # Note
        /// A `Cols` of one yields a row vector.
        pub fn transpose(self: Self) Matrix(T, Cols, Rows) {
            if (Cols == 1) {
                return vector.Vector(T, Rows)
                    .fromElements(self.columns[0]);
            }
            var result: Matrix(T, Cols, Rows) = undefined;
            inline for (0..Rows) |r| {
                inline for (0..Cols) |c| {
                    result.columns[r][c] = self.columns[c][r];
                }
            }
            return result;
        }

        /// Expands a compact affine matrix to `4x4`.
        ///
        /// # Return
        /// The same transform with a full last
        /// column.
        ///
        /// # Compile Error
        /// - The matrix is not `4x3`.
        pub fn toMat4(self: Self) Mat4(T) {
            if (Rows != 4 or Cols != 3) {
                @compileError("toMat4: matrix must be 4x3");
            }
            return Mat4(T).fromColumns(.{
                self.columns[0],
                self.columns[1],
                self.columns[2],
                .{ 0, 0, 0, 1 },
            });
        }

        /// Compacts an affine `4x4` matrix to `4x3`.
        ///
        /// # Return
        /// The first three columns.
        ///
        /// # Note
        /// The fourth column is dropped.
        ///
        /// # Compile Error
        /// - The matrix is not `4x4`.
        pub fn toMat3x4(self: Self) Mat3x4(T) {
            if (Rows != 4 or Cols != 4) {
                @compileError("toMat3x4: matrix must be 4x4");
            }
            return Mat3x4(T).fromColumns(.{
                self.columns[0],
                self.columns[1],
                self.columns[2],
            });
        }

        /// Computes the determinant.
        ///
        /// # Return
        /// The signed determinant.
        ///
        /// # Compile Error
        /// - The matrix is not square.
        /// - The matrix is larger than `4x4`.
        pub fn determinant(self: Self) T {
            if (Rows != Cols) {
                @compileError(
                    "determinant: matrix must be square",
                );
            }
            if (Rows == 2) {
                return self.columns[0][0] * self.columns[1][1] -
                    self.columns[1][0] * self.columns[0][1];
            }
            if (Rows == 3) {
                const a = vector.Vector(T, 3)
                    .fromElements(self.columns[0]);
                const b = vector.Vector(T, 3)
                    .fromElements(self.columns[1]);
                const c = vector.Vector(T, 3)
                    .fromElements(self.columns[2]);
                return a.dot(b.cross(c));
            }
            if (Rows == 4) {
                var sum: T = 0;
                inline for (0..4) |c| {
                    const sign: T = comptime if (c % 2 == 0) 1 else -1;
                    sum += sign * self.columns[c][0] *
                        self.minorOf(0, c).determinant();
                }
                return sum;
            }
            @compileError(
                "determinant: supports 2x2 to 4x4",
            );
        }

        /// Computes the adjugate matrix.
        ///
        /// # Return
        /// The transposed cofactor matrix.
        ///
        /// # Compile Error
        /// - The matrix is not square.
        /// - The matrix is larger than `4x4`.
        pub fn adjugate(self: Self) Self {
            if (Rows != Cols) {
                @compileError(
                    "adjugate: matrix must be square",
                );
            }
            var result: Self = undefined;
            if (Rows == 2) {
                result.columns[0] = .{
                    self.columns[1][1],
                    -self.columns[0][1],
                };
                result.columns[1] = .{
                    -self.columns[1][0],
                    self.columns[0][0],
                };
                return result;
            }
            if (Rows > 4) {
                @compileError(
                    "adjugate: supports 2x2 to 4x4",
                );
            }
            inline for (0..Rows) |r| {
                inline for (0..Rows) |c| {
                    const sign: T = comptime if ((r + c) % 2 == 0) 1 else -1;
                    result.columns[c][r] = sign *
                        self.minorOf(c, r).determinant();
                }
            }
            return result;
        }

        /// Inverts a square float matrix.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted
        ///   `abs(det)`.
        ///
        /// # Return
        /// The inverse, or `null` when the determinant
        /// is too close to zero.
        ///
        /// # Compile Error
        /// - The components are not floats.
        /// - The matrix is not square or larger than
        ///   `4x4`.
        pub fn inverse(
            self: Self,
            epsilon_value: T,
        ) ?Self {
            assertFloat(T);
            const det = self.determinant();
            if (base.compare.isZero(T, det, epsilon_value)) {
                return null;
            }
            return self.adjugate().scale(1 / det);
        }

        /// Computes the normal transformation matrix.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted
        ///   `abs(det)`.
        ///
        /// # Return
        /// The inverse transpose, or `null` when the
        /// matrix is not invertible.
        ///
        /// # Compile Error
        /// - The components are not floats.
        /// - The matrix is not square or larger than
        ///   `4x4`.
        pub fn normalMatrix(
            self: Self,
            epsilon_value: T,
        ) ?Self {
            assertFloat(T);
            if (Rows != Cols) {
                @compileError(
                    "normalMatrix: matrix must be square",
                );
            }
            return self.transpose().inverse(epsilon_value);
        }

        /// Splits the matrix into transform parts.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted row
        ///   length.
        ///
        /// # Return
        /// Translation, rotation, and scale, or `null`
        /// when a scale is too small.
        ///
        /// # Note
        /// The matrix must be affine and free of
        /// shear; see `Decomposition.fromMatrix`.
        ///
        /// # Compile Error
        /// - The matrix is not `4x4`.
        pub fn decompose(
            self: Self,
            epsilon_value: T,
        ) ?decomposition.Decomposition(T) {
            if (Rows != 4 or Cols != 4) {
                @compileError(
                    "decompose: matrix must be 4x4",
                );
            }
            return decomposition.Decomposition(T)
                .fromMatrix(self, epsilon_value);
        }

        /// Transforms a point.
        ///
        /// # Args
        /// - `point`: the point to transform.
        ///
        /// # Return
        /// The transformed point. A `4x4` matrix
        /// divides by the homogeneous `w`.
        ///
        /// # Note
        /// A `4x3` matrix uses an implicit `w` of one.
        ///
        /// # Compile Error
        /// - The matrix is not `3x3`, `4x4`, or `4x3`.
        pub fn transformPoint(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            if (Rows == 3 and Cols == 3) {
                return point.mulMatrix(3, self);
            }
            if (Rows == 4 and Cols == 3) {
                const homogeneous = vector.Vector(T, 4)
                    .fromElements(.{
                    point.elements[0],
                    point.elements[1],
                    point.elements[2],
                    1,
                });
                return homogeneous.mulMatrix(3, self);
            }
            if (Rows == 4 and Cols == 4) {
                const homogeneous = vector.Vector(T, 4)
                    .fromElements(.{
                    point.elements[0],
                    point.elements[1],
                    point.elements[2],
                    1,
                });
                const clip = homogeneous.mulMatrix(4, self);
                const w = clip.elements[3];
                if (w == 0) {
                    return vector.Vector(T, 3)
                        .fromElements(.{
                        clip.elements[0],
                        clip.elements[1],
                        clip.elements[2],
                    });
                }
                return vector.Vector(T, 3).fromElements(.{
                    clip.elements[0] / w,
                    clip.elements[1] / w,
                    clip.elements[2] / w,
                });
            }
            @compileError(
                "transformPoint: expected 3x3, 4x4, or 4x3",
            );
        }

        /// Transforms a direction.
        ///
        /// # Args
        /// - `value`: the direction to transform.
        ///
        /// # Return
        /// The transformed direction, ignoring
        /// translation.
        ///
        /// # Compile Error
        /// - The matrix is not `3x3`, `4x4`, or `4x3`.
        pub fn transformDirection(
            self: Self,
            value: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            if (Rows == 3 and Cols == 3) {
                return value.mulMatrix(3, self);
            }
            if (Rows == 4 and (Cols == 3 or Cols == 4)) {
                const homogeneous = vector.Vector(T, 4)
                    .fromElements(.{
                    value.elements[0],
                    value.elements[1],
                    value.elements[2],
                    0,
                });
                const result = homogeneous.mulMatrix(Cols, self);
                return vector.Vector(T, 3).fromElements(.{
                    result.elements[0],
                    result.elements[1],
                    result.elements[2],
                });
            }
            @compileError(
                "transformDirection: expected 3x3, 4x4, or 4x3",
            );
        }

        /// Transforms a normal.
        ///
        /// # Args
        /// - `normal`: the normal to transform.
        /// - `epsilon_value`: smallest accepted
        ///   determinant of the linear part.
        ///
        /// # Return
        /// The transformed normal, or `null` when the
        /// linear part is not invertible.
        ///
        /// # Note
        /// The inverse transpose keeps normals
        /// perpendicular under non-uniform scale. The
        /// result is not normalized.
        ///
        /// # Compile Error
        /// - The matrix is not `3x3`, `4x4`, or `4x3`.
        pub fn transformNormal(
            self: Self,
            normal: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?vector.Vector(T, 3) {
            const normals = self.linearPart()
                .normalMatrix(epsilon_value) orelse return null;
            return normal.mulMatrix(3, normals);
        }

        /// Transforms a world point into local space.
        ///
        /// # Args
        /// - `point`: the world-space point.
        /// - `epsilon_value`: smallest accepted
        ///   determinant.
        ///
        /// # Return
        /// The local-space point, or `null` when the
        /// matrix is not invertible.
        ///
        /// # Compile Error
        /// - The matrix is not a square float matrix.
        pub fn toLocal(
            self: Self,
            point: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?vector.Vector(T, 3) {
            if (Rows != Cols) {
                @compileError(
                    "toLocal: matrix must be square",
                );
            }
            const inverted = self.inverse(epsilon_value) orelse
                return null;
            return inverted.transformPoint(point);
        }

        /// Transforms a local point into world space.
        ///
        /// # Args
        /// - `point`: the local-space point.
        ///
        /// # Return
        /// The world-space point.
        ///
        /// # Compile Error
        /// - The matrix is not `3x3`, `4x4`, or `4x3`.
        pub fn toWorld(
            self: Self,
            point: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            return self.transformPoint(point);
        }

        fn linearPart(self: Self) Mat3(T) {
            if (Rows == 3 and Cols == 3) return self;
            if (Rows == 4 and (Cols == 3 or Cols == 4)) {
                return Mat3(T).fromColumns(.{
                    .{
                        self.columns[0][0],
                        self.columns[0][1],
                        self.columns[0][2],
                    },
                    .{
                        self.columns[1][0],
                        self.columns[1][1],
                        self.columns[1][2],
                    },
                    .{
                        self.columns[2][0],
                        self.columns[2][1],
                        self.columns[2][2],
                    },
                });
            }
            @compileError(
                "linearPart: expected 3x3, 4x4, or 4x3",
            );
        }

        fn minorOf(
            self: Self,
            comptime skip_row: usize,
            comptime skip_col: usize,
        ) Matrix(T, Rows - 1, Cols - 1) {
            const Result = Matrix(T, Rows - 1, Cols - 1);
            var result: Result = undefined;
            inline for (0..Cols - 1) |c| {
                const source_c = comptime blk: {
                    var source = c;
                    if (source >= skip_col) source += 1;
                    break :blk source;
                };
                var value_column: Result.Column = @splat(0);
                inline for (0..Rows - 1) |r| {
                    const source_r = comptime blk: {
                        var source = r;
                        if (source >= skip_row) source += 1;
                        break :blk source;
                    };
                    value_column[r] =
                        self.columns[source_c][source_r];
                }
                result.columns[c] = value_column;
            }
            return result;
        }
    };
}

fn assertFloat(comptime T: type) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "expected float components, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

test "matrix construction" {
    const Vec3f = vector.Vec3(f32);
    const identity = Mat3(f32).identity();
    try std.testing.expectEqual(@as(f32, 1), identity.at(0, 0));
    try std.testing.expectEqual(@as(f32, 0), identity.at(0, 1));

    const from_rows = Mat3(f32).fromRows(.{
        Vec3f.fromElements(.{ 1, 2, 3 }),
        Vec3f.fromElements(.{ 4, 5, 6 }),
        Vec3f.fromElements(.{ 7, 8, 9 }),
    });
    try std.testing.expectEqual(@as(f32, 4), from_rows.at(1, 0));
    try std.testing.expectEqual(@as(f32, 6), from_rows.at(1, 2));
    try std.testing.expectEqual(@as(f32, 4), from_rows.column(0)[1]);
    const row = from_rows.row(1);
    try std.testing.expectEqual(@as(f32, 4), row.at(0));
    try std.testing.expectEqual(@as(f32, 6), row.at(2));

    var mutable = from_rows;
    mutable.set(0, 0, 9);
    try std.testing.expectEqual(@as(f32, 9), mutable.at(0, 0));

    const filled = Mat2(i32).fromElement(7);
    try std.testing.expectEqual(@as(i32, 7), filled.at(1, 1));
}

test "matrix arithmetic" {
    const Vec2f = vector.Vec2(f32);
    const a = Mat2(f32).fromRows(.{
        Vec2f.fromElements(.{ 1, 2 }),
        Vec2f.fromElements(.{ 3, 4 }),
    });
    const b = Mat2(f32).fromRows(.{
        Vec2f.fromElements(.{ 5, 6 }),
        Vec2f.fromElements(.{ 7, 8 }),
    });
    const sum = a.add(b);
    try std.testing.expectEqual(@as(f32, 6), sum.at(0, 0));
    try std.testing.expectEqual(@as(f32, 12), sum.at(1, 1));
    const difference = b.sub(a);
    try std.testing.expectEqual(@as(f32, 4), difference.at(0, 0));
    try std.testing.expectEqual(@as(f32, -1), a.negate().at(0, 0));
    try std.testing.expectEqual(@as(f32, 2), a.scale(2).at(0, 0));

    const product = a.mul(b);
    try std.testing.expectEqual(@as(f32, 19), product.at(0, 0));
    try std.testing.expectEqual(@as(f32, 22), product.at(0, 1));
    try std.testing.expectEqual(@as(f32, 43), product.at(1, 0));
    try std.testing.expectEqual(@as(f32, 50), product.at(1, 1));
}

test "matrix multiply shapes" {
    const a = Matrix(f32, 2, 3).fromRows(.{
        vector.Vec3(f32).fromElements(.{ 1, 2, 3 }),
        vector.Vec3(f32).fromElements(.{ 4, 5, 6 }),
    });
    const b = Matrix(f32, 3, 2).fromRows(.{
        vector.Vec2(f32).fromElements(.{ 7, 8 }),
        vector.Vec2(f32).fromElements(.{ 9, 10 }),
        vector.Vec2(f32).fromElements(.{ 11, 12 }),
    });
    const product = a.mulTo(2, b);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(product).row_count);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(product).column_count);
    try std.testing.expectEqual(@as(f32, 58), product.at(0, 0));
    try std.testing.expectEqual(@as(f32, 154), product.at(1, 1));
}

test "matrix transpose" {
    const a = Matrix(f32, 2, 3).fromRows(.{
        vector.Vec3(f32).fromElements(.{ 1, 2, 3 }),
        vector.Vec3(f32).fromElements(.{ 4, 5, 6 }),
    });
    const transposed = a.transpose();
    try std.testing.expectEqual(@as(usize, 3), @TypeOf(transposed).row_count);
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(transposed).column_count);
    try std.testing.expectEqual(@as(f32, 4), transposed.at(0, 1));
    try std.testing.expectEqual(@as(f32, 6), transposed.at(2, 1));

    const ColumnMatrix = Matrix(f32, 2, 1);
    const tall = ColumnMatrix.fromColumns(.{.{ 3, 4 }});
    const row_vector = tall.transpose();
    try std.testing.expectEqual(@as(f32, 3), row_vector.at(0));
    try std.testing.expectEqual(@as(f32, 4), row_vector.at(1));
}

test "matrix determinant" {
    const Vec2f = vector.Vec2(f32);
    const a = Mat2(f32).fromRows(.{
        Vec2f.fromElements(.{ 1, 2 }),
        Vec2f.fromElements(.{ 3, 4 }),
    });
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        a.determinant(),
        1e-6,
    );

    const b = Mat3(f32).fromRows(.{
        vector.Vec3(f32).fromElements(.{ 6, 1, 1 }),
        vector.Vec3(f32).fromElements(.{ 4, -2, 5 }),
        vector.Vec3(f32).fromElements(.{ 2, 8, 7 }),
    });
    try std.testing.expectApproxEqAbs(
        @as(f32, -306),
        b.determinant(),
        1e-4,
    );

    const c = Mat4(f32).fromColumns(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 3, 0 },
        .{ 0, 0, 0, 4 },
    });
    try std.testing.expectApproxEqAbs(
        @as(f32, 24),
        c.determinant(),
        1e-4,
    );
}

test "matrix inverse" {
    const cases = [_]Mat3(f32){
        Mat3(f32).fromRows(.{
            vector.Vec3(f32).fromElements(.{ 1, 2, 3 }),
            vector.Vec3(f32).fromElements(.{ 0, 1, 4 }),
            vector.Vec3(f32).fromElements(.{ 5, 6, 0 }),
        }),
        Mat3(f32).fromColumns(.{
            .{ 2, 0, 0 },
            .{ 0, 4, 0 },
            .{ 0, 0, 8 },
        }),
        Mat3(f32).identity(),
    };
    for (cases) |matrix| {
        const inverted = matrix.inverse(1e-6).?;
        const product = matrix.mul(inverted);
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

    const singular = Mat2(f32).fromRows(.{
        vector.Vec2(f32).fromElements(.{ 1, 2 }),
        vector.Vec2(f32).fromElements(.{ 2, 4 }),
    });
    try std.testing.expect(singular.inverse(1e-6) == null);

    const four = Mat4(f32).fromColumns(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 4, 0, 0 },
        .{ 0, 0, 8, 0 },
        .{ 1, 2, 3, 1 },
    });
    const inverted_four = four.inverse(1e-6).?;
    const product_four = four.mul(inverted_four);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        product_four.at(3, 3),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        product_four.at(3, 0),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        inverted_four.at(0, 0),
        1e-5,
    );
}

test "normal matrix" {
    const matrix = Mat3(f32).fromColumns(.{
        .{ 2, 0, 0 },
        .{ 0, 4, 0 },
        .{ 0, 0, 8 },
    });
    const normals = matrix.normalMatrix(1e-6).?;
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
}

test "matrix vector product" {
    const matrix = Mat3(f32).fromRows(.{
        vector.Vec3(f32).fromElements(.{ 1, 2, 3 }),
        vector.Vec3(f32).fromElements(.{ 4, 5, 6 }),
        vector.Vec3(f32).fromElements(.{ 7, 8, 9 }),
    });
    const point = vector.Vec3(f32).fromElements(.{ 1, 1, 1 });
    const result = point.mulMatrix(3, matrix);
    try std.testing.expectEqual(@as(f32, 12), result.at(0));
    try std.testing.expectEqual(@as(f32, 15), result.at(1));
    try std.testing.expectEqual(@as(f32, 18), result.at(2));

    const affine = Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const homogeneous = vector.Vec4(f32)
        .fromElements(.{ 1, 2, 3, 1 });
    const transformed = homogeneous.mulMatrix(3, affine);
    try std.testing.expectEqual(@as(f32, 6), transformed.at(0));
    try std.testing.expectEqual(@as(f32, 8), transformed.at(1));
    try std.testing.expectEqual(@as(f32, 10), transformed.at(2));
}

test "transform helpers" {
    const matrix = Mat4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
        .{ 0, 0, 0, 1 },
    });
    const point = vector.Vec3(f32).fromElements(.{ 1, 2, 3 });
    const world = matrix.transformPoint(point);
    try std.testing.expectEqual(@as(f32, 6), world.at(0));
    try std.testing.expectEqual(@as(f32, 8), world.at(1));
    try std.testing.expectEqual(@as(f32, 10), world.at(2));

    const direction = matrix.transformDirection(point);
    try std.testing.expectEqual(@as(f32, 1), direction.at(0));
    try std.testing.expectEqual(@as(f32, 3), direction.at(2));

    const back = matrix.toLocal(world, 1e-9).?;
    try std.testing.expectApproxEqAbs(point.at(0), back.at(0), 1e-5);
    try std.testing.expectApproxEqAbs(point.at(2), back.at(2), 1e-5);

    const scaled = Mat4(f32).fromColumns(.{
        .{ 2, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    });
    const normal = scaled
        .transformNormal(
        vector.Vec3(f32).fromElements(.{ 1, 0, 0 }),
        1e-9,
    ).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), normal.at(0), 1e-6);

    const compact = Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const compact_world = compact.transformPoint(point);
    try std.testing.expectEqual(@as(f32, 6), compact_world.at(0));
}

test "matrix row vector case" {
    const Row = Matrix(f32, 1, 3);
    try std.testing.expectEqual(vector.Vec3(f32), Row);
    try std.testing.expectEqual(vector.Vec2(f32), Matrix(f32, 1, 2));
    const value = Row.fromElements(.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(f32, 2), value.at(1));
    const column_matrix = value.transpose();
    try std.testing.expectEqual(@as(f32, 3), column_matrix.at(2, 0));
}

test "affine conversion" {
    const compact = Mat3x4(f32).fromColumns(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
    });
    const full = compact.toMat4();
    try std.testing.expectEqual(@as(f32, 5), full.at(3, 0));
    try std.testing.expectEqual(@as(f32, 1), full.at(3, 3));
    try std.testing.expectEqual(@as(f32, 0), full.at(0, 3));

    const back = full.toMat3x4();
    inline for (0..4) |r| {
        inline for (0..3) |c| {
            try std.testing.expectEqual(
                compact.at(r, c),
                back.at(r, c),
            );
        }
    }
}

test "degenerate matrix shapes" {
    const column = Matrix(f32, 2, 1).fromColumns(.{.{ 2, 3 }});
    const row_vector = vector.Vec2(f32)
        .fromElements(.{ 4, 5 });
    const product = column.mulTo(2, row_vector);
    try std.testing.expectEqual(@as(f32, 8), product.at(0, 0));
    try std.testing.expectEqual(@as(f32, 10), product.at(0, 1));
    try std.testing.expectEqual(@as(f32, 12), product.at(1, 0));
    try std.testing.expectEqual(@as(f32, 15), product.at(1, 1));

    const scalar = vector.Vector(f32, 1)
        .fromElements(.{2});
    const scaled = scalar.mulMatrix(3, vector.Vec3(f32)
        .fromElements(.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(f32, 2), scaled.at(0));
    try std.testing.expectEqual(@as(f32, 6), scaled.at(2));
}
