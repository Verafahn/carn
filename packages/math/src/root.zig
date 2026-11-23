const matrix = @import("matrix.zig");

pub const Matrix = matrix.Matrix;

pub fn Vector(comptime T: type, comptime N: usize) type {
    return Matrix(T, 1, N);
}