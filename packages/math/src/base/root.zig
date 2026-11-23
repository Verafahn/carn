//! Scalar helpers shared by every math type.

pub const compare = @import("compare.zig");
pub const epsilon = @import("epsilon.zig");
pub const fixed = @import("fixed.zig");

pub const Epsilon = epsilon.Epsilon;
pub const Fixed = fixed.Fixed;

test {
    _ = compare;
    _ = epsilon;
    _ = fixed;
}
