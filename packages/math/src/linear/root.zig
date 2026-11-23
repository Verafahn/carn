//! Linear algebra: vectors, matrices, quaternions,
//! Euler angles, and orthonormal bases.

pub const vector = @import("vector.zig");
pub const matrix = @import("matrix.zig");
pub const quaternion = @import("quaternion.zig");
pub const euler = @import("euler.zig");
pub const basis = @import("basis.zig");
pub const decompose = @import("decompose.zig");

test {
    _ = vector;
    _ = matrix;
    _ = quaternion;
    _ = euler;
    _ = basis;
    _ = decompose;
}
