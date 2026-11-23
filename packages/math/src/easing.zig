//! Easing curves for animation timing.
//!
//! Every function maps a normalized progress of
//! `0..1` to an eased value, and accepts a float or
//! a float vector. Progress outside `0..1` is passed
//! through the formula, which is useful for overshoot
//! inspection but not for playback.

const std = @import("std");

/// Quadratic ease in: `t * t`.
pub fn easeInQuad(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return v * v;
}

/// Quadratic ease out.
pub fn easeOutQuad(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 1) - v;
    return broadcast(T, 1) - u * u;
}

/// Quadratic ease in and out.
pub fn easeInOutQuad(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 2) - broadcast(T, 2) * v;
    const in_part = broadcast(T, 2) * v * v;
    const out_part = broadcast(T, 1) - u * u / broadcast(T, 2);
    return selectLess(T, t, 0.5, in_part, out_part);
}

/// Cubic ease in: `t * t * t`.
pub fn easeInCubic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return v * v * v;
}

/// Cubic ease out.
pub fn easeOutCubic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 1) - v;
    return broadcast(T, 1) - u * u * u;
}

/// Cubic ease in and out.
pub fn easeInOutCubic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 2) - broadcast(T, 2) * v;
    const in_part = broadcast(T, 4) * v * v * v;
    const out_part = broadcast(T, 1) - u * u * u / broadcast(T, 2);
    return selectLess(T, t, 0.5, in_part, out_part);
}

/// Quintic ease in.
pub fn easeInQuint(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return v * v * v * v * v;
}

/// Quintic ease out.
pub fn easeOutQuint(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 1) - v;
    return broadcast(T, 1) - u * u * u * u * u;
}

/// Quintic ease in and out.
pub fn easeInOutQuint(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = broadcast(T, 2) - broadcast(T, 2) * v;
    const in_part = broadcast(T, 16) * v * v * v * v * v;
    const out_part = broadcast(T, 1) -
        u * u * u * u * u / broadcast(T, 2);
    return selectLess(T, t, 0.5, in_part, out_part);
}

/// Sinusoidal ease in.
pub fn easeInSine(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return broadcast(T, 1) -
        @cos(v * broadcast(T, std.math.pi * 0.5));
}

/// Sinusoidal ease out.
pub fn easeOutSine(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return @sin(v * broadcast(T, std.math.pi * 0.5));
}

/// Sinusoidal ease in and out.
pub fn easeInOutSine(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return (@cos(v * broadcast(T, std.math.pi)) -
        broadcast(T, 1)) * broadcast(T, -0.5);
}

/// Exponential ease in.
pub fn easeInExpo(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const value = @exp2(v * broadcast(T, 10) -
        broadcast(T, 10));
    return selectLte(T, t, 0, broadcast(T, 0), value);
}

/// Exponential ease out.
pub fn easeOutExpo(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const value = broadcast(T, 1) -
        @exp2(v * broadcast(T, -10));
    return selectGte(T, t, 1, broadcast(T, 1), value);
}

/// Exponential ease in and out.
pub fn easeInOutExpo(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const in_part = @exp2(v * broadcast(T, 20) -
        broadcast(T, 10)) * broadcast(T, 0.5);
    const out_part = (broadcast(T, 2) -
        @exp2(v * broadcast(T, -20) + broadcast(T, 10))) *
        broadcast(T, 0.5);
    const middle = selectLess(T, t, 0.5, in_part, out_part);
    const low = selectLte(T, t, 0, broadcast(T, 0), middle);
    return selectGte(T, t, 1, broadcast(T, 1), low);
}

/// Elastic ease in.
pub fn easeInElastic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const base = @exp2((v - broadcast(T, 1)) *
        broadcast(T, 10));
    const angle = (v * broadcast(T, 10) -
        broadcast(T, 10.75)) *
        broadcast(T, 2.0 * std.math.pi / 3.0);
    const value = -base * @sin(angle);
    const low = selectLte(T, t, 0, broadcast(T, 0), value);
    return selectGte(T, t, 1, broadcast(T, 1), low);
}

/// Elastic ease out.
pub fn easeOutElastic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const base = @exp2(v * broadcast(T, -10));
    const angle = (v * broadcast(T, 10) -
        broadcast(T, 0.75)) *
        broadcast(T, 2.0 * std.math.pi / 3.0);
    const value = base * @sin(angle) + broadcast(T, 1);
    const low = selectLte(T, t, 0, broadcast(T, 0), value);
    return selectGte(T, t, 1, broadcast(T, 1), low);
}

/// Elastic ease in and out.
pub fn easeInOutElastic(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const wave = broadcast(T, 2.0 * std.math.pi / 4.5);
    const in_part = -(@exp2(v * broadcast(T, 20) -
        broadcast(T, 10)) *
        @sin((v * broadcast(T, 20) -
            broadcast(T, 11.125)) * wave)) *
        broadcast(T, 0.5);
    const out_part = (@exp2(v * broadcast(T, -20) +
        broadcast(T, 10)) *
        @sin((v * broadcast(T, 20) -
            broadcast(T, 11.125)) * wave)) *
        broadcast(T, 0.5) + broadcast(T, 1);
    const middle = selectLess(T, t, 0.5, in_part, out_part);
    const low = selectLte(T, t, 0, broadcast(T, 0), middle);
    return selectGte(T, t, 1, broadcast(T, 1), low);
}

/// Back ease in: pulls back before moving forward.
pub fn easeInBack(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const c1 = broadcast(T, 1.70158);
    const c3 = broadcast(T, 2.70158);
    return c3 * v * v * v - c1 * v * v;
}

/// Back ease out: overshoots before settling.
pub fn easeOutBack(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const u = v - broadcast(T, 1);
    const c1 = broadcast(T, 1.70158);
    const c3 = broadcast(T, 2.70158);
    return broadcast(T, 1) +
        c3 * u * u * u + c1 * u * u;
}

/// Back ease in and out.
pub fn easeInOutBack(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const c2 = broadcast(T, 2.5949095);
    const doubled = broadcast(T, 2) * v;
    const u = doubled - broadcast(T, 2);
    const in_part = (doubled * doubled *
        ((c2 + broadcast(T, 1)) * doubled - c2)) *
        broadcast(T, 0.5);
    const out_part = (u * u *
        ((c2 + broadcast(T, 1)) * u + c2) +
        broadcast(T, 2)) * broadcast(T, 0.5);
    return selectLess(T, t, 0.5, in_part, out_part);
}

/// Bounce ease out.
pub fn easeOutBounce(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return bounceOut(T, v);
}

/// Bounce ease in.
pub fn easeInBounce(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    return broadcast(T, 1) - bounceOut(
        T,
        broadcast(T, 1) - v,
    );
}

/// Bounce ease in and out.
pub fn easeInOutBounce(t: anytype) @TypeOf(t) {
    const T = @TypeOf(t);
    const v = toLike(T, t);
    const scaled = broadcast(T, 2) * v;
    const in_part = (broadcast(T, 1) - bounceOut(
        T,
        broadcast(T, 1) - scaled,
    )) * broadcast(T, 0.5);
    const out_part = (broadcast(T, 1) + bounceOut(
        T,
        scaled - broadcast(T, 1),
    )) * broadcast(T, 0.5);
    return selectLess(T, t, 0.5, in_part, out_part);
}

fn bounceOut(comptime T: type, v: T) T {
    const n1 = broadcast(T, 7.5625);
    const d1 = broadcast(T, 2.75);
    const first = n1 * v * v;
    const second_shift = v - broadcast(T, 1.5) / d1;
    const second = n1 * second_shift * second_shift +
        broadcast(T, 0.75);
    const third_shift = v - broadcast(T, 2.25) / d1;
    const third = n1 * third_shift * third_shift +
        broadcast(T, 0.9375);
    const fourth_shift = v - broadcast(T, 2.625) / d1;
    const fourth = n1 * fourth_shift * fourth_shift +
        broadcast(T, 0.984375);
    const last = selectLess(
        T,
        v,
        broadcast(T, 2.5) / d1,
        third,
        fourth,
    );
    const middle = selectLess(
        T,
        v,
        broadcast(T, 2) / d1,
        second,
        last,
    );
    return selectLess(
        T,
        v,
        broadcast(T, 1) / d1,
        first,
        middle,
    );
}

fn broadcast(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => @splat(value),
        else => value,
    };
}

fn toLike(comptime T: type, value: anytype) T {
    return switch (@typeInfo(T)) {
        .vector => switch (@typeInfo(@TypeOf(value))) {
            .vector => value,
            else => @splat(value),
        },
        else => value,
    };
}

fn selectLess(
    comptime T: type,
    x: anytype,
    edge: anytype,
    less: T,
    greater: T,
) T {
    return switch (@typeInfo(T)) {
        .vector => |info| @select(
            info.child,
            x < toLike(T, edge),
            less,
            greater,
        ),
        else => if (x < edge) less else greater,
    };
}

fn selectLte(
    comptime T: type,
    x: anytype,
    edge: anytype,
    less: T,
    greater: T,
) T {
    return switch (@typeInfo(T)) {
        .vector => |info| @select(
            info.child,
            x <= toLike(T, edge),
            less,
            greater,
        ),
        else => if (x <= edge) less else greater,
    };
}

fn selectGte(
    comptime T: type,
    x: anytype,
    edge: anytype,
    greater: T,
    less: T,
) T {
    return switch (@typeInfo(T)) {
        .vector => |info| @select(
            info.child,
            x >= toLike(T, edge),
            greater,
            less,
        ),
        else => if (x >= edge) greater else less,
    };
}

fn checkEndpoints(comptime ease: anytype) !void {
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        ease(@as(f32, 0)),
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        ease(@as(f32, 1)),
        1e-4,
    );
}

test "easing endpoints" {
    try checkEndpoints(easeInQuad);
    try checkEndpoints(easeOutQuad);
    try checkEndpoints(easeInOutQuad);
    try checkEndpoints(easeInCubic);
    try checkEndpoints(easeOutCubic);
    try checkEndpoints(easeInOutCubic);
    try checkEndpoints(easeInQuint);
    try checkEndpoints(easeOutQuint);
    try checkEndpoints(easeInOutQuint);
    try checkEndpoints(easeInSine);
    try checkEndpoints(easeOutSine);
    try checkEndpoints(easeInOutSine);
    try checkEndpoints(easeInExpo);
    try checkEndpoints(easeOutExpo);
    try checkEndpoints(easeInOutExpo);
    try checkEndpoints(easeInElastic);
    try checkEndpoints(easeOutElastic);
    try checkEndpoints(easeInOutElastic);
    try checkEndpoints(easeInBack);
    try checkEndpoints(easeOutBack);
    try checkEndpoints(easeInOutBack);
    try checkEndpoints(easeInBounce);
    try checkEndpoints(easeOutBounce);
    try checkEndpoints(easeInOutBounce);
}

test "easing shapes" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        easeInQuad(@as(f32, 0.5)),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.75),
        easeOutQuad(@as(f32, 0.5)),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        easeInOutQuad(@as(f32, 0.5)),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        easeInOutSine(@as(f32, 0.5)),
        1e-6,
    );
    try std.testing.expect(easeOutBack(@as(f32, 0.8)) > 1);
    try std.testing.expect(easeInBack(@as(f32, 0.2)) < 0);
    try std.testing.expect(
        easeOutBounce(@as(f32, 0.7)) >= 0 and
            easeOutBounce(@as(f32, 0.7)) <= 1,
    );
}

test "easing vectors" {
    const inputs: @Vector(3, f32) = .{ 0, 0.5, 1 };
    const values = easeInOutCubic(inputs);
    try std.testing.expectApproxEqAbs(@as(f32, 0), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), values[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[2], 1e-6);
    const bounced = easeOutBounce(inputs);
    try std.testing.expectApproxEqAbs(@as(f32, 0), bounced[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), bounced[2], 1e-6);
}
