//! RGB and RGBA colors with transfer functions.
//!
//! Color components are stored directly, so no
//! implicit gamma conversion happens. `srgbToLinear`
//! and `linearToSrgb` convert between the encoded
//! sRGB space and linear light.
//!
//! HSV and HSL use a hue in `0..1` and saturation,
//! value, or lightness in `0..1`. Conversions return
//! plain vectors for those spaces.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");

/// Returns an RGB color type with components of `T`.
///
/// # Args
/// - `T`: the component type.
///
/// # Return
/// A struct with red, green, and blue.
pub fn Rgb(comptime T: type) type {
    return struct {
        /// Red component.
        r: T,
        /// Green component.
        g: T,
        /// Blue component.
        b: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Returns an RGBA color type with components of `T`.
///
/// # Args
/// - `T`: the component type.
///
/// # Return
/// A struct with red, green, blue, and alpha.
pub fn Rgba(comptime T: type) type {
    return struct {
        /// Red component.
        r: T,
        /// Green component.
        g: T,
        /// Blue component.
        b: T,
        /// Alpha component.
        a: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Builds a color from a three-component vector.
///
/// # Args
/// - `value`: components in `(r, g, b)` order.
///
/// # Return
/// The color.
///
/// # Compile Error
/// - `value` is not a three-component vector.
pub fn rgbFromVector(value: anytype) Rgb(
    scalar.ScalarOf(@TypeOf(value)),
) {
    return .{ .r = value[0], .g = value[1], .b = value[2] };
}

/// Packs a color into a three-component vector.
///
/// # Args
/// - `color`: the color to pack.
///
/// # Return
/// Components in `(r, g, b)` order.
pub fn rgbToVector(color: anytype) @Vector(3, @TypeOf(color.r)) {
    return .{ color.r, color.g, color.b };
}

/// Builds a color from a four-component vector.
///
/// # Args
/// - `value`: components in `(r, g, b, a)` order.
///
/// # Return
/// The color.
///
/// # Compile Error
/// - `value` is not a four-component vector.
pub fn rgbaFromVector(value: anytype) Rgba(
    scalar.ScalarOf(@TypeOf(value)),
) {
    return .{
        .r = value[0],
        .g = value[1],
        .b = value[2],
        .a = value[3],
    };
}

/// Packs a color into a four-component vector.
///
/// # Args
/// - `color`: the color to pack.
///
/// # Return
/// Components in `(r, g, b, a)` order.
pub fn rgbaToVector(color: anytype) @Vector(4, @TypeOf(color.r)) {
    return .{ color.r, color.g, color.b, color.a };
}

/// Converts sRGB encoded values to linear light.
///
/// # Args
/// - `value`: encoded components; a float or float
///   vector.
///
/// # Return
/// Linear components.
///
/// # Note
/// Negative components produce unspecified output.
pub fn srgbToLinear(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const v = toLike(T, value);
    const low = v / broadcast(T, 12.92);
    const base = (v + broadcast(T, 0.055)) /
        broadcast(T, 1.055);
    const high = @exp2(broadcast(T, 2.4) * @log2(base));
    return selectLte(T, v, 0.04045, low, high);
}

/// Converts linear light to sRGB encoded values.
///
/// # Args
/// - `value`: linear components; a float or float
///   vector.
///
/// # Return
/// sRGB encoded components.
///
/// # Note
/// Negative components produce unspecified output.
pub fn linearToSrgb(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const v = toLike(T, value);
    const low = v * broadcast(T, 12.92);
    const high = broadcast(T, 1.055) *
        @exp2(broadcast(T, 1.0 / 2.4) * @log2(v)) -
        broadcast(T, 0.055);
    return selectLte(T, v, 0.0031308, low, high);
}

/// Applies gamma correction to encoded values.
///
/// # Args
/// - `value`: encoded components; a float or float
///   vector.
/// - `gamma`: the encoding exponent.
///
/// # Return
/// Linear components.
///
/// # Note
/// Negative components produce unspecified output.
pub fn gammaToLinear(value: anytype, gamma: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const v = toLike(T, value);
    return @exp2(toLike(T, gamma) * @log2(v));
}

/// Removes gamma correction from linear values.
///
/// # Args
/// - `value`: linear components; a float or float
///   vector.
/// - `gamma`: the encoding exponent.
///
/// # Return
/// Encoded components.
///
/// # Note
/// Negative components produce unspecified output.
pub fn linearToGamma(value: anytype, gamma: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    const v = toLike(T, value);
    const exponent = toLike(T, gamma);
    return @exp2((1 / exponent) * @log2(v));
}

/// Converts HSV to RGB.
///
/// # Args
/// - `hsv`: hue, saturation, and value.
///
/// # Return
/// The RGB color.
///
/// # Note
/// Hue wraps modulo one; saturation above one
/// produces out-of-range components.
pub fn hsvToRgb(hsv: anytype) Rgb(
    scalar.ScalarOf(@TypeOf(hsv)),
) {
    const T = scalar.ScalarOf(@TypeOf(hsv));
    const value: @Vector(3, T) = hsv;
    const hue = value[0] * 6;
    const saturation = value[1];
    const brightness = value[2];
    const sector = @floor(hue);
    const fraction = hue - sector;
    const p = brightness * (1 - saturation);
    const q = brightness * (1 - saturation * fraction);
    const t = brightness *
        (1 - saturation * (1 - fraction));
    const index: u32 = @intFromFloat(@mod(sector, 6));
    return switch (index) {
        0 => .{ .r = brightness, .g = t, .b = p },
        1 => .{ .r = q, .g = brightness, .b = p },
        2 => .{ .r = p, .g = brightness, .b = t },
        3 => .{ .r = p, .g = q, .b = brightness },
        4 => .{ .r = t, .g = p, .b = brightness },
        else => .{ .r = brightness, .g = p, .b = q },
    };
}

/// Converts RGB to HSV.
///
/// # Args
/// - `color`: the color to convert.
///
/// # Return
/// Hue, saturation, and value.
///
/// # Note
/// Hue is undefined for gray colors and is
/// reported as zero.
pub fn rgbToHsv(color: anytype) @Vector(3, @TypeOf(color.r)) {
    const T = @TypeOf(color.r);
    const r = color.r;
    const g = color.g;
    const b = color.b;
    const maximum = @max(r, @max(g, b));
    const minimum = @min(r, @min(g, b));
    const delta = maximum - minimum;

    var hue: T = 0;
    if (delta != 0) {
        if (maximum == r) {
            hue = @mod((g - b) / delta, 6);
        } else if (maximum == g) {
            hue = (b - r) / delta + 2;
        } else {
            hue = (r - g) / delta + 4;
        }
        hue = hue / 6;
    }
    const saturation = if (maximum == 0)
        0
    else
        delta / maximum;
    return .{ hue, saturation, maximum };
}

/// Converts HSL to RGB.
///
/// # Args
/// - `hsl`: hue, saturation, and lightness.
///
/// # Return
/// The RGB color.
///
/// # Note
/// Hue wraps modulo one.
pub fn hslToRgb(hsl: anytype) Rgb(
    scalar.ScalarOf(@TypeOf(hsl)),
) {
    const T = scalar.ScalarOf(@TypeOf(hsl));
    const value: @Vector(3, T) = hsl;
    const hue = value[0] * 6;
    const saturation = value[1];
    const lightness = value[2];
    const chroma = (1 - @abs(2 * lightness - 1)) *
        saturation;
    const sector = @mod(hue, 6);
    const second = chroma *
        (1 - @abs(@mod(sector, 2) - 1));
    const base: @Vector(3, T) = switch (@as(u32, @intFromFloat(@floor(sector)))) {
        0 => .{ chroma, second, 0 },
        1 => .{ second, chroma, 0 },
        2 => .{ 0, chroma, second },
        3 => .{ 0, second, chroma },
        4 => .{ second, 0, chroma },
        else => .{ chroma, 0, second },
    };
    const offset = lightness - chroma / 2;
    return .{
        .r = base[0] + offset,
        .g = base[1] + offset,
        .b = base[2] + offset,
    };
}

/// Converts RGB to HSL.
///
/// # Args
/// - `color`: the color to convert.
///
/// # Return
/// Hue, saturation, and lightness.
///
/// # Note
/// Hue is undefined for gray colors and is
/// reported as zero.
pub fn rgbToHsl(color: anytype) @Vector(3, @TypeOf(color.r)) {
    const T = @TypeOf(color.r);
    const r = color.r;
    const g = color.g;
    const b = color.b;
    const maximum = @max(r, @max(g, b));
    const minimum = @min(r, @min(g, b));
    const delta = maximum - minimum;
    const lightness = (maximum + minimum) / 2;

    var hue: T = 0;
    if (delta != 0) {
        if (maximum == r) {
            hue = @mod((g - b) / delta, 6);
        } else if (maximum == g) {
            hue = (b - r) / delta + 2;
        } else {
            hue = (r - g) / delta + 4;
        }
        hue = hue / 6;
    }
    const denominator = 1 - @abs(2 * lightness - 1);
    const saturation = if (denominator == 0)
        0
    else
        delta / denominator;
    return .{ hue, saturation, lightness };
}

/// Computes the Rec.709 luminance of a color.
///
/// # Args
/// - `color`: the color to measure.
///
/// # Return
/// The weighted sum of the components.
///
/// # Note
/// The weights assume linear light; convert sRGB
/// values with `srgbToLinear` first.
pub fn luminance(color: anytype) @TypeOf(color.r) {
    return color.r * 0.2126 + color.g * 0.7152 +
        color.b * 0.0722;
}

/// Multiplies two colors componentwise.
///
/// # Args
/// - `a`: first color.
/// - `b`: second color.
///
/// # Return
/// The componentwise product.
pub fn multiply(a: anytype, b: anytype) @TypeOf(a) {
    return .{
        .r = a.r * b.r,
        .g = a.g * b.g,
        .b = a.b * b.b,
    };
}

/// Scales every color channel.
///
/// # Args
/// - `color`: the color to scale.
/// - `factor`: multiplier for all three channels.
///
/// # Return
/// The scaled color.
pub fn tint(color: anytype, factor: anytype) @TypeOf(color) {
    return .{
        .r = color.r * factor,
        .g = color.g * factor,
        .b = color.b * factor,
    };
}

/// Blends two colors linearly.
///
/// # Args
/// - `a`: color at `t = 0`.
/// - `b`: color at `t = 1`.
/// - `t`: blend factor; not clamped.
///
/// # Return
/// The blended color.
pub fn blend(a: anytype, b: anytype, t: anytype) @TypeOf(a) {
    const inverse = 1 - t;
    return .{
        .r = a.r * inverse + b.r * t,
        .g = a.g * inverse + b.g * t,
        .b = a.b * inverse + b.b * t,
    };
}

/// Adjusts color saturation toward gray.
///
/// # Args
/// - `color`: the color to adjust.
/// - `amount`: zero yields gray, one keeps the
///   color, and larger values exaggerate it.
///
/// # Return
/// The adjusted color.
pub fn adjustSaturation(color: anytype, amount: anytype) @TypeOf(color) {
    const gray = luminance(color);
    return .{
        .r = gray + (color.r - gray) * amount,
        .g = gray + (color.g - gray) * amount,
        .b = gray + (color.b - gray) * amount,
    };
}

/// Multiplies a color by its alpha.
///
/// # Args
/// - `color`: the color to premultiply.
///
/// # Return
/// RGB channels scaled by alpha.
pub fn premultiply(color: anytype) @TypeOf(color) {
    return .{
        .r = color.r * color.a,
        .g = color.g * color.a,
        .b = color.b * color.a,
        .a = color.a,
    };
}

/// Composites a source color over a destination.
///
/// # Args
/// - `source`: the foreground color.
/// - `destination`: the background color.
///
/// # Return
/// The composited color.
///
/// # Note
/// Both inputs must use straight, not premultiplied,
/// alpha.
pub fn over(source: anytype, destination: anytype) @TypeOf(source) {
    const inverse = 1 - source.a;
    const alpha = source.a + destination.a * inverse;
    return .{
        .r = source.r * source.a +
            destination.r * destination.a * inverse,
        .g = source.g * source.a +
            destination.g * destination.a * inverse,
        .b = source.b * source.a +
            destination.b * destination.a * inverse,
        .a = alpha,
    };
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

test "color packing" {
    const color = rgbFromVector(@Vector(3, f32){ 1, 0.5, 0 });
    try std.testing.expectEqual(@as(f32, 0.5), color.g);
    const packed_color = rgbToVector(color);
    try std.testing.expectEqual(@as(f32, 1), packed_color[0]);

    const with_alpha = rgbaFromVector(@Vector(4, f32){ 1, 0.5, 0, 0.25 });
    try std.testing.expectEqual(@as(f32, 0.25), with_alpha.a);
    const packed_alpha = rgbaToVector(with_alpha);
    try std.testing.expectEqual(@as(f32, 0), packed_alpha[2]);
}

test "srgb transfer" {
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        srgbToLinear(@as(f32, 0)),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        srgbToLinear(@as(f32, 1)),
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.2140),
        srgbToLinear(@as(f32, 0.5)),
        1e-3,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        linearToSrgb(srgbToLinear(@as(f32, 0.5))),
        1e-5,
    );
    const linear = srgbToLinear(@Vector(3, f32){ 0, 0.5, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), linear[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), linear[2], 1e-5);

    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        gammaToLinear(@as(f32, 0.5), 2),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        linearToGamma(@as(f32, 0.25), 2),
        1e-6,
    );
}

test "hsv" {
    const red = hsvToRgb(@Vector(3, f32){ 0, 1, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), red.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), red.g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), red.b, 1e-6);

    const green = hsvToRgb(@Vector(3, f32){ 1.0 / 3.0, 1, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), green.r, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), green.g, 1e-5);

    const blue = hsvToRgb(@Vector(3, f32){ 2.0 / 3.0, 1, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), blue.b, 1e-5);

    const color = Rgb(f32){ .r = 0.2, .g = 0.6, .b = 0.4 };
    const hsv = rgbToHsv(color);
    const round_trip = hsvToRgb(hsv);
    try std.testing.expectApproxEqAbs(color.r, round_trip.r, 1e-5);
    try std.testing.expectApproxEqAbs(color.g, round_trip.g, 1e-5);
    try std.testing.expectApproxEqAbs(color.b, round_trip.b, 1e-5);

    const gray = rgbToHsv(Rgb(f32){ .r = 0.5, .g = 0.5, .b = 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), gray[1], 1e-6);
}

test "hsl" {
    const red = hslToRgb(@Vector(3, f32){ 0, 1, 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), red.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), red.g, 1e-6);

    const green = hslToRgb(@Vector(3, f32){ 1.0 / 3.0, 1, 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), green.g, 1e-5);

    const color = Rgb(f32){ .r = 0.1, .g = 0.3, .b = 0.8 };
    const hsl = rgbToHsl(color);
    const round_trip = hslToRgb(hsl);
    try std.testing.expectApproxEqAbs(color.r, round_trip.r, 1e-5);
    try std.testing.expectApproxEqAbs(color.g, round_trip.g, 1e-5);
    try std.testing.expectApproxEqAbs(color.b, round_trip.b, 1e-5);
}

test "color grading" {
    const white = Rgb(f32){ .r = 1, .g = 1, .b = 1 };
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        luminance(white),
        1e-6,
    );
    const red = Rgb(f32){ .r = 1, .g = 0, .b = 0 };
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.2126),
        luminance(red),
        1e-6,
    );

    const half = tint(white, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), half.r, 1e-6);

    const product = multiply(
        Rgb(f32){ .r = 0.5, .g = 0.25, .b = 1 },
        Rgb(f32){ .r = 2, .g = 2, .b = 0.5 },
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), product.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), product.b, 1e-6);

    const mixed = blend(
        Rgb(f32){ .r = 0, .g = 0, .b = 0 },
        Rgb(f32){ .r = 1, .g = 1, .b = 1 },
        0.25,
    );
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), mixed.g, 1e-6);

    const gray = adjustSaturation(
        Rgb(f32){ .r = 1, .g = 0, .b = 0 },
        0,
    );
    try std.testing.expectApproxEqAbs(gray.r, gray.g, 1e-6);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.2126),
        gray.r,
        1e-6,
    );
}

test "alpha compositing" {
    const opaque_source = Rgba(f32){
        .r = 1,
        .g = 0,
        .b = 0,
        .a = 1,
    };
    const destination = Rgba(f32){
        .r = 0,
        .g = 0,
        .b = 1,
        .a = 1,
    };
    const composed = over(opaque_source, destination);
    try std.testing.expectApproxEqAbs(@as(f32, 1), composed.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), composed.b, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), composed.a, 1e-6);

    const transparent = Rgba(f32){
        .r = 1,
        .g = 1,
        .b = 1,
        .a = 0,
    };
    const kept = over(transparent, destination);
    try std.testing.expectApproxEqAbs(@as(f32, 0), kept.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), kept.b, 1e-6);

    const premultiplied = premultiply(Rgba(f32){
        .r = 1,
        .g = 0.5,
        .b = 0,
        .a = 0.5,
    });
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), premultiplied.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), premultiplied.a, 1e-6);
}
