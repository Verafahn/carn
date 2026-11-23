//! RGB and RGBA colors with transfer functions.
//!
//! Components are stored as given, so no implicit gamma
//! conversion happens. HSV and HSL use a hue in
//! `0..1` and saturation, value, or lightness in
//! `0..1`; those spaces are exchanged as vectors.

const std = @import("std");
const vector = @import("../linear/vector.zig");

/// Returns an RGB color type with components of `T`.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Return
/// A struct with red, green, and blue.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Rgb(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Rgb: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Red component.
        r: T,
        /// Green component.
        g: T,
        /// Blue component.
        b: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a color from a vector.
        ///
        /// # Args
        /// - `value`: components in `(r, g, b)` order.
        ///
        /// # Return
        /// The color.
        pub fn fromVector(value: vector.Vector(T, 3)) Self {
            return .{
                .r = value.at(0),
                .g = value.at(1),
                .b = value.at(2),
            };
        }

        /// Packs the color into a vector.
        ///
        /// # Return
        /// Components in `(r, g, b)` order.
        pub fn toVector(self: Self) vector.Vector(T, 3) {
            return vector.Vector(T, 3).fromElements(.{
                self.r,
                self.g,
                self.b,
            });
        }

        /// Converts sRGB encoded components to linear
        /// light.
        ///
        /// # Return
        /// The linear color.
        ///
        /// # Note
        /// Negative components produce unspecified
        /// output.
        pub fn srgbToLinear(self: Self) Self {
            return .{
                .r = srgbComponent(T, self.r),
                .g = srgbComponent(T, self.g),
                .b = srgbComponent(T, self.b),
            };
        }

        /// Converts linear components to sRGB encoding.
        ///
        /// # Return
        /// The encoded color.
        ///
        /// # Note
        /// Negative components produce unspecified
        /// output.
        pub fn linearToSrgb(self: Self) Self {
            return .{
                .r = linearComponent(T, self.r),
                .g = linearComponent(T, self.g),
                .b = linearComponent(T, self.b),
            };
        }

        /// Applies gamma correction.
        ///
        /// # Args
        /// - `gamma`: the encoding exponent.
        ///
        /// # Return
        /// The linear color.
        pub fn gammaToLinear(self: Self, gamma: T) Self {
            return .{
                .r = std.math.pow(T, self.r, gamma),
                .g = std.math.pow(T, self.g, gamma),
                .b = std.math.pow(T, self.b, gamma),
            };
        }

        /// Removes gamma correction.
        ///
        /// # Args
        /// - `gamma`: the encoding exponent.
        ///
        /// # Return
        /// The encoded color.
        pub fn linearToGamma(self: Self, gamma: T) Self {
            const exponent = 1 / gamma;
            return .{
                .r = std.math.pow(T, self.r, exponent),
                .g = std.math.pow(T, self.g, exponent),
                .b = std.math.pow(T, self.b, exponent),
            };
        }

        /// Converts the color to HSV.
        ///
        /// # Return
        /// Hue, saturation, and value.
        ///
        /// # Note
        /// Hue is undefined for gray colors and is
        /// reported as zero.
        pub fn toHsv(self: Self) vector.Vector(T, 3) {
            const maximum = @max(self.r, @max(self.g, self.b));
            const minimum = @min(self.r, @min(self.g, self.b));
            const delta = maximum - minimum;
            var hue: T = 0;
            if (delta != 0) {
                hue = hueFromComponents(
                    T,
                    self.r,
                    self.g,
                    self.b,
                    maximum,
                    delta,
                );
            }
            const saturation = if (maximum == 0)
                0
            else
                delta / maximum;
            return vector.Vector(T, 3).fromElements(.{
                hue,
                saturation,
                maximum,
            });
        }

        /// Converts HSV to a color.
        ///
        /// # Args
        /// - `hsv`: hue, saturation, and value.
        ///
        /// # Return
        /// The RGB color.
        ///
        /// # Note
        /// Hue wraps modulo one.
        pub fn fromHsv(hsv: vector.Vector(T, 3)) Self {
            const hue = hsv.at(0) * 6;
            const saturation = hsv.at(1);
            const brightness = hsv.at(2);
            const sector = @floor(hue);
            const fraction = hue - sector;
            const p = brightness * (1 - saturation);
            const q = brightness *
                (1 - saturation * fraction);
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

        /// Converts the color to HSL.
        ///
        /// # Return
        /// Hue, saturation, and lightness.
        ///
        /// # Note
        /// Hue is undefined for gray colors and is
        /// reported as zero.
        pub fn toHsl(self: Self) vector.Vector(T, 3) {
            const maximum = @max(self.r, @max(self.g, self.b));
            const minimum = @min(self.r, @min(self.g, self.b));
            const delta = maximum - minimum;
            const lightness = (maximum + minimum) / 2;
            var hue: T = 0;
            if (delta != 0) {
                hue = hueFromComponents(
                    T,
                    self.r,
                    self.g,
                    self.b,
                    maximum,
                    delta,
                );
            }
            const denominator = 1 - @abs(2 * lightness - 1);
            const saturation = if (denominator == 0)
                0
            else
                delta / denominator;
            return vector.Vector(T, 3).fromElements(.{
                hue,
                saturation,
                lightness,
            });
        }

        /// Converts HSL to a color.
        ///
        /// # Args
        /// - `hsl`: hue, saturation, and lightness.
        ///
        /// # Return
        /// The RGB color.
        ///
        /// # Note
        /// Hue wraps modulo one.
        pub fn fromHsl(hsl: vector.Vector(T, 3)) Self {
            const hue = hsl.at(0) * 6;
            const saturation = hsl.at(1);
            const lightness = hsl.at(2);
            const chroma = (1 - @abs(2 * lightness - 1)) *
                saturation;
            const sector = @mod(hue, 6);
            const second = chroma *
                (1 - @abs(@mod(sector, 2) - 1));
            const base: vector.Vector(T, 3) = switch (@as(u32, @intFromFloat(@floor(sector)))) {
                0 => vector.Vector(T, 3)
                    .fromElements(.{ chroma, second, 0 }),
                1 => vector.Vector(T, 3)
                    .fromElements(.{ second, chroma, 0 }),
                2 => vector.Vector(T, 3)
                    .fromElements(.{ 0, chroma, second }),
                3 => vector.Vector(T, 3)
                    .fromElements(.{ 0, second, chroma }),
                4 => vector.Vector(T, 3)
                    .fromElements(.{ second, 0, chroma }),
                else => vector.Vector(T, 3)
                    .fromElements(.{ chroma, 0, second }),
            };
            const offset = lightness - chroma / 2;
            return .{
                .r = base.at(0) + offset,
                .g = base.at(1) + offset,
                .b = base.at(2) + offset,
            };
        }

        /// Computes the Rec.709 luminance.
        ///
        /// # Return
        /// The weighted sum of the components.
        ///
        /// # Note
        /// The weights assume linear light; convert
        /// sRGB values with `srgbToLinear` first.
        pub fn luminance(self: Self) T {
            return self.r * 0.2126 + self.g * 0.7152 +
                self.b * 0.0722;
        }

        /// Multiplies two colors componentwise.
        ///
        /// # Args
        /// - `other`: the other color.
        ///
        /// # Return
        /// The componentwise product.
        pub fn multiply(self: Self, other: Self) Self {
            return .{
                .r = self.r * other.r,
                .g = self.g * other.g,
                .b = self.b * other.b,
            };
        }

        /// Scales every channel.
        ///
        /// # Args
        /// - `factor`: the multiplier.
        ///
        /// # Return
        /// The scaled color.
        pub fn scale(self: Self, factor: T) Self {
            return .{
                .r = self.r * factor,
                .g = self.g * factor,
                .b = self.b * factor,
            };
        }

        /// Blends two colors linearly.
        ///
        /// # Args
        /// - `other`: the color at `t = 1`.
        /// - `t`: the blend factor; not clamped.
        ///
        /// # Return
        /// The blended color.
        pub fn blend(self: Self, other: Self, t: T) Self {
            const inverse = 1 - t;
            return .{
                .r = self.r * inverse + other.r * t,
                .g = self.g * inverse + other.g * t,
                .b = self.b * inverse + other.b * t,
            };
        }

        /// Adjusts saturation toward gray.
        ///
        /// # Args
        /// - `amount`: zero yields gray, one keeps the
        ///   color, and larger values exaggerate it.
        ///
        /// # Return
        /// The adjusted color.
        pub fn adjustSaturation(self: Self, amount: T) Self {
            const gray = self.luminance();
            return .{
                .r = gray + (self.r - gray) * amount,
                .g = gray + (self.g - gray) * amount,
                .b = gray + (self.b - gray) * amount,
            };
        }
    };
}

/// Returns an RGBA color type with components of `T`.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Return
/// A struct with red, green, blue, and alpha.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Rgba(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Rgba: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Red component.
        r: T,
        /// Green component.
        g: T,
        /// Blue component.
        b: T,
        /// Alpha component.
        a: T,

        const Self = @This();

        /// The component type.
        pub const Component = T;

        /// Builds a color from a vector.
        ///
        /// # Args
        /// - `value`: components in `(r, g, b, a)`
        ///   order.
        ///
        /// # Return
        /// The color.
        pub fn fromVector(value: vector.Vector(T, 4)) Self {
            return .{
                .r = value.at(0),
                .g = value.at(1),
                .b = value.at(2),
                .a = value.at(3),
            };
        }

        /// Packs the color into a vector.
        ///
        /// # Return
        /// Components in `(r, g, b, a)` order.
        pub fn toVector(self: Self) vector.Vector(T, 4) {
            return vector.Vector(T, 4).fromElements(.{
                self.r,
                self.g,
                self.b,
                self.a,
            });
        }

        /// Drops the alpha component.
        ///
        /// # Return
        /// The RGB color.
        pub fn toRgb(self: Self) Rgb(T) {
            return .{ .r = self.r, .g = self.g, .b = self.b };
        }

        /// Multiplies the color by its alpha.
        ///
        /// # Return
        /// RGB channels scaled by alpha.
        pub fn premultiply(self: Self) Self {
            return .{
                .r = self.r * self.a,
                .g = self.g * self.a,
                .b = self.b * self.a,
                .a = self.a,
            };
        }

        /// Composites this color over another.
        ///
        /// # Args
        /// - `destination`: the background color.
        ///
        /// # Return
        /// The composited color.
        ///
        /// # Note
        /// Both inputs use straight, not premultiplied,
        /// alpha.
        pub fn over(self: Self, destination: Self) Self {
            const inverse = 1 - self.a;
            return .{
                .r = self.r * self.a +
                    destination.r * destination.a * inverse,
                .g = self.g * self.a +
                    destination.g * destination.a * inverse,
                .b = self.b * self.a +
                    destination.b * destination.a * inverse,
                .a = self.a + destination.a * inverse,
            };
        }
    };
}

fn srgbComponent(comptime T: type, value: T) T {
    if (value <= 0.04045) return value / 12.92;
    return std.math.pow(
        T,
        (value + 0.055) / 1.055,
        2.4,
    );
}

fn linearComponent(comptime T: type, value: T) T {
    if (value <= 0.0031308) return value * 12.92;
    return 1.055 * std.math.pow(T, value, 1.0 / 2.4) - 0.055;
}

fn hueFromComponents(
    comptime T: type,
    r: T,
    g: T,
    b: T,
    maximum: T,
    delta: T,
) T {
    var hue: T = 0;
    if (maximum == r) {
        hue = @mod((g - b) / delta, 6);
    } else if (maximum == g) {
        hue = (b - r) / delta + 2;
    } else {
        hue = (r - g) / delta + 4;
    }
    return hue / 6;
}

test "color packing and transfer" {
    const Vec3f = vector.Vec3(f32);
    const color = Rgb(f32).fromVector(
        Vec3f.fromElements(.{ 1, 0.5, 0 }),
    );
    try std.testing.expectEqual(@as(f32, 0.5), color.g);
    try std.testing.expectEqual(@as(f32, 1), color.toVector().at(0));

    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        (Rgb(f32){ .r = 0, .g = 0, .b = 0 })
            .srgbToLinear()
            .r,
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        (Rgb(f32){ .r = 1, .g = 1, .b = 1 })
            .srgbToLinear()
            .r,
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.2140),
        (Rgb(f32){ .r = 0.5, .g = 0.5, .b = 0.5 })
            .srgbToLinear()
            .r,
        1e-3,
    );
    const round_trip = (Rgb(f32){ .r = 0.5, .g = 0.5, .b = 0.5 })
        .srgbToLinear()
        .linearToSrgb();
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        round_trip.r,
        1e-5,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        (Rgb(f32){ .r = 0.5, .g = 0.5, .b = 0.5 })
            .gammaToLinear(2)
            .r,
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        (Rgb(f32){ .r = 0.25, .g = 0.25, .b = 0.25 })
            .linearToGamma(2)
            .r,
        1e-6,
    );

    const with_alpha = Rgba(f32).fromVector(
        vector.Vec4(f32).fromElements(.{ 1, 0.5, 0, 0.25 }),
    );
    try std.testing.expectEqual(@as(f32, 0.25), with_alpha.a);
    try std.testing.expectEqual(@as(f32, 0), with_alpha.toRgb().b);
}

test "hsv" {
    const Vec3f = vector.Vec3(f32);
    const red = Rgb(f32).fromHsv(
        Vec3f.fromElements(.{ 0, 1, 1 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), red.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), red.g, 1e-6);

    const green = Rgb(f32).fromHsv(
        Vec3f.fromElements(.{ 1.0 / 3.0, 1, 1 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), green.g, 1e-5);
    const blue = Rgb(f32).fromHsv(
        Vec3f.fromElements(.{ 2.0 / 3.0, 1, 1 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), blue.b, 1e-5);

    const color = Rgb(f32){ .r = 0.2, .g = 0.6, .b = 0.4 };
    const round_trip = Rgb(f32).fromHsv(color.toHsv());
    try std.testing.expectApproxEqAbs(color.r, round_trip.r, 1e-5);
    try std.testing.expectApproxEqAbs(color.g, round_trip.g, 1e-5);
    try std.testing.expectApproxEqAbs(color.b, round_trip.b, 1e-5);

    const gray = (Rgb(f32){ .r = 0.5, .g = 0.5, .b = 0.5 })
        .toHsv();
    try std.testing.expectApproxEqAbs(@as(f32, 0), gray.at(1), 1e-6);
}

test "hsl" {
    const Vec3f = vector.Vec3(f32);
    const red = Rgb(f32).fromHsl(
        Vec3f.fromElements(.{ 0, 1, 0.5 }),
    );
    try std.testing.expectApproxEqAbs(@as(f32, 1), red.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), red.g, 1e-6);

    const color = Rgb(f32){ .r = 0.1, .g = 0.3, .b = 0.8 };
    const round_trip = Rgb(f32).fromHsl(color.toHsl());
    try std.testing.expectApproxEqAbs(color.r, round_trip.r, 1e-5);
    try std.testing.expectApproxEqAbs(color.g, round_trip.g, 1e-5);
    try std.testing.expectApproxEqAbs(color.b, round_trip.b, 1e-5);
}

test "color grading and compositing" {
    const white = Rgb(f32){ .r = 1, .g = 1, .b = 1 };
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        white.luminance(),
        1e-6,
    );
    const red = Rgb(f32){ .r = 1, .g = 0, .b = 0 };
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.2126),
        red.luminance(),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        white.scale(0.5).r,
        1e-6,
    );
    const product = (Rgb(f32){ .r = 0.5, .g = 0.25, .b = 1 })
        .multiply(Rgb(f32){ .r = 2, .g = 2, .b = 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), product.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), product.b, 1e-6);
    const mixed = (Rgb(f32){ .r = 0, .g = 0, .b = 0 })
        .blend(white, 0.25);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), mixed.g, 1e-6);
    const gray = red.adjustSaturation(0);
    try std.testing.expectApproxEqAbs(gray.r, gray.g, 1e-6);

    const source = Rgba(f32){ .r = 1, .g = 0, .b = 0, .a = 1 };
    const destination = Rgba(f32){
        .r = 0,
        .g = 0,
        .b = 1,
        .a = 1,
    };
    const composed = source.over(destination);
    try std.testing.expectApproxEqAbs(@as(f32, 1), composed.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), composed.b, 1e-6);

    const transparent = Rgba(f32){
        .r = 1,
        .g = 1,
        .b = 1,
        .a = 0,
    };
    const kept = transparent.over(destination);
    try std.testing.expectApproxEqAbs(@as(f32, 0), kept.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), kept.b, 1e-6);

    const premultiplied = (Rgba(f32){
        .r = 1,
        .g = 0.5,
        .b = 0,
        .a = 0.5,
    }).premultiply();
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        premultiplied.r,
        1e-6,
    );
}
