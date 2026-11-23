//! Easing curves for animation timing.
//!
//! Every function maps a normalized progress of
//! `0..1` to an eased value. Progress outside `0..1`
//! is passed through the formula, which is useful for
//! inspecting overshoot but not for playback.

const std = @import("std");

/// Quadratic ease in: `t * t`.
///
/// # Args
/// - `T`: a float type.
/// - `t`: normalized progress.
///
/// # Return
/// The eased value.
pub fn easeInQuad(comptime T: type, t: T) T {
    assertFloat(T);
    return t * t;
}

/// Quadratic ease out.
pub fn easeOutQuad(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 1 - t;
    return 1 - u * u;
}

/// Quadratic ease in and out.
pub fn easeInOutQuad(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 2 - 2 * t;
    if (t < 0.5) return 2 * t * t;
    return 1 - u * u / 2;
}

/// Cubic ease in: `t * t * t`.
pub fn easeInCubic(comptime T: type, t: T) T {
    assertFloat(T);
    return t * t * t;
}

/// Cubic ease out.
pub fn easeOutCubic(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 1 - t;
    return 1 - u * u * u;
}

/// Cubic ease in and out.
pub fn easeInOutCubic(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 2 - 2 * t;
    if (t < 0.5) return 4 * t * t * t;
    return 1 - u * u * u / 2;
}

/// Quintic ease in.
pub fn easeInQuint(comptime T: type, t: T) T {
    assertFloat(T);
    return t * t * t * t * t;
}

/// Quintic ease out.
pub fn easeOutQuint(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 1 - t;
    return 1 - u * u * u * u * u;
}

/// Quintic ease in and out.
pub fn easeInOutQuint(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 2 - 2 * t;
    if (t < 0.5) return 16 * t * t * t * t * t;
    return 1 - u * u * u * u * u / 2;
}

/// Sinusoidal ease in.
pub fn easeInSine(comptime T: type, t: T) T {
    assertFloat(T);
    return 1 - @cos(t * (std.math.pi * 0.5));
}

/// Sinusoidal ease out.
pub fn easeOutSine(comptime T: type, t: T) T {
    assertFloat(T);
    return @sin(t * (std.math.pi * 0.5));
}

/// Sinusoidal ease in and out.
pub fn easeInOutSine(comptime T: type, t: T) T {
    assertFloat(T);
    return (@cos(t * std.math.pi) - 1) * -0.5;
}

/// Exponential ease in.
pub fn easeInExpo(comptime T: type, t: T) T {
    assertFloat(T);
    if (t <= 0) return 0;
    return @exp2(t * 10 - 10);
}

/// Exponential ease out.
pub fn easeOutExpo(comptime T: type, t: T) T {
    assertFloat(T);
    if (t >= 1) return 1;
    return 1 - @exp2(t * -10);
}

/// Exponential ease in and out.
pub fn easeInOutExpo(comptime T: type, t: T) T {
    assertFloat(T);
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    if (t < 0.5) return @exp2(t * 20 - 10) * 0.5;
    return (2 - @exp2(t * -20 + 10)) * 0.5;
}

/// Elastic ease in.
pub fn easeInElastic(comptime T: type, t: T) T {
    assertFloat(T);
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const base = @exp2((t - 1) * 10);
    const angle = (t * 10 - 10.75) *
        (2.0 * std.math.pi / 3.0);
    return -base * @sin(angle);
}

/// Elastic ease out.
pub fn easeOutElastic(comptime T: type, t: T) T {
    assertFloat(T);
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const base = @exp2(t * -10);
    const angle = (t * 10 - 0.75) *
        (2.0 * std.math.pi / 3.0);
    return base * @sin(angle) + 1;
}

/// Elastic ease in and out.
pub fn easeInOutElastic(comptime T: type, t: T) T {
    assertFloat(T);
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const wave = 2.0 * std.math.pi / 4.5;
    const angle = (t * 20 - 11.125) * wave;
    if (t < 0.5) {
        return -(@exp2(t * 20 - 10) * @sin(angle)) * 0.5;
    }
    return @exp2(t * -20 + 10) * @sin(angle) * 0.5 + 1;
}

/// Back ease in: pulls back before moving forward.
pub fn easeInBack(comptime T: type, t: T) T {
    assertFloat(T);
    const c1 = 1.70158;
    const c3 = 2.70158;
    return c3 * t * t * t - c1 * t * t;
}

/// Back ease out: overshoots before settling.
pub fn easeOutBack(comptime T: type, t: T) T {
    assertFloat(T);
    const u = t - 1;
    const c1 = 1.70158;
    const c3 = 2.70158;
    return 1 + c3 * u * u * u + c1 * u * u;
}

/// Back ease in and out.
pub fn easeInOutBack(comptime T: type, t: T) T {
    assertFloat(T);
    const c2 = 2.5949095;
    const doubled = 2 * t;
    if (t < 0.5) {
        return doubled * doubled *
            ((c2 + 1) * doubled - c2) / 2;
    }
    const shifted = doubled - 2;
    return (shifted * shifted *
        ((c2 + 1) * shifted + c2) + 2) / 2;
}

/// Bounce ease out.
pub fn easeOutBounce(comptime T: type, t: T) T {
    assertFloat(T);
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1.0 / d1) return n1 * t * t;
    if (t < 2.0 / d1) {
        const shifted = t - 1.5 / d1;
        return n1 * shifted * shifted + 0.75;
    }
    if (t < 2.5 / d1) {
        const shifted = t - 2.25 / d1;
        return n1 * shifted * shifted + 0.9375;
    }
    const shifted = t - 2.625 / d1;
    return n1 * shifted * shifted + 0.984375;
}

/// Bounce ease in.
pub fn easeInBounce(comptime T: type, t: T) T {
    assertFloat(T);
    const u = 1 - t;
    return 1 - easeOutBounce(T, u);
}

/// Bounce ease in and out.
pub fn easeInOutBounce(comptime T: type, t: T) T {
    assertFloat(T);
    if (t < 0.5) {
        return (1 - easeOutBounce(T, 1 - 2 * t)) / 2;
    }
    return (1 + easeOutBounce(T, 2 * t - 1)) / 2;
}

fn assertFloat(comptime T: type) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "expected a float type, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

fn expectEnds(in_value: f32, out_value: f32) !void {
    try std.testing.expectApproxEqAbs(@as(f32, 0), in_value, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out_value, 1e-4);
}

test "easing endpoints" {
    try expectEnds(easeInQuad(f32, 0), easeInQuad(f32, 1));
    try expectEnds(easeOutQuad(f32, 0), easeOutQuad(f32, 1));
    try expectEnds(easeInOutQuad(f32, 0), easeInOutQuad(f32, 1));
    try expectEnds(easeInCubic(f32, 0), easeInCubic(f32, 1));
    try expectEnds(easeOutCubic(f32, 0), easeOutCubic(f32, 1));
    try expectEnds(easeInOutCubic(f32, 0), easeInOutCubic(f32, 1));
    try expectEnds(easeInQuint(f32, 0), easeInQuint(f32, 1));
    try expectEnds(easeOutQuint(f32, 0), easeOutQuint(f32, 1));
    try expectEnds(easeInOutQuint(f32, 0), easeInOutQuint(f32, 1));
    try expectEnds(easeInSine(f32, 0), easeInSine(f32, 1));
    try expectEnds(easeOutSine(f32, 0), easeOutSine(f32, 1));
    try expectEnds(easeInOutSine(f32, 0), easeInOutSine(f32, 1));
    try expectEnds(easeInExpo(f32, 0), easeInExpo(f32, 1));
    try expectEnds(easeOutExpo(f32, 0), easeOutExpo(f32, 1));
    try expectEnds(easeInOutExpo(f32, 0), easeInOutExpo(f32, 1));
    try expectEnds(easeInElastic(f32, 0), easeInElastic(f32, 1));
    try expectEnds(easeOutElastic(f32, 0), easeOutElastic(f32, 1));
    try expectEnds(easeInOutElastic(f32, 0), easeInOutElastic(f32, 1));
    try expectEnds(easeInBack(f32, 0), easeInBack(f32, 1));
    try expectEnds(easeOutBack(f32, 0), easeOutBack(f32, 1));
    try expectEnds(easeInOutBack(f32, 0), easeInOutBack(f32, 1));
    try expectEnds(easeInBounce(f32, 0), easeInBounce(f32, 1));
    try expectEnds(easeOutBounce(f32, 0), easeOutBounce(f32, 1));
    try expectEnds(
        easeInOutBounce(f32, 0),
        easeInOutBounce(f32, 1),
    );
}

test "easing shapes" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        easeInQuad(f32, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.75),
        easeOutQuad(f32, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        easeInOutQuad(f32, 0.5),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        easeInOutSine(f32, 0.5),
        1e-6,
    );
    try std.testing.expect(easeOutBack(f32, 0.8) > 1);
    try std.testing.expect(easeInBack(f32, 0.2) < 0);
    try std.testing.expect(easeOutBounce(f32, 0.7) >= 0);
    try std.testing.expect(easeOutBounce(f32, 0.7) <= 1);
}
