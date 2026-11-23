//! Suggested tolerance constants.
//!
//! Nothing in the library uses these implicitly: every
//! comparison takes its tolerance from the caller.

/// Suggested tolerances grouped by precision.
pub const Epsilon = struct {
    /// Coarse absolute tolerance for `f16`.
    pub const half_coarse: f16 = 1.0 / 1024.0;
    /// Fine absolute tolerance for `f16`.
    pub const half_fine: f16 = 1.0 / 32768.0;
    /// Coarse absolute tolerance for `f32`.
    pub const single_coarse: f32 = 1.0e-5;
    /// Fine absolute tolerance for `f32`.
    pub const single_fine: f32 = 1.0e-7;
    /// Coarse absolute tolerance for `f64`.
    pub const double_coarse: f64 = 1.0e-11;
    /// Fine absolute tolerance for `f64`.
    pub const double_fine: f64 = 1.0e-14;
};

test "epsilon values are positive" {
    const std = @import("std");
    try std.testing.expect(Epsilon.half_coarse > 0);
    try std.testing.expect(Epsilon.single_coarse > 0);
    try std.testing.expect(Epsilon.double_coarse > 0);
    try std.testing.expect(Epsilon.single_fine < Epsilon.single_coarse);
}
