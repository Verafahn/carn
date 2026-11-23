//! Row vectors as the `1xN` matrix case.
//!
//! `Vector(T, N)` is exactly `Matrix(T, 1, N)`.
//! The representation keeps the components in one
//! `@Vector`, so a one-row matrix needs no per-column
//! array. Every operation is a method, which keeps the
//! operand types fixed.
//!
//! All arithmetic is plain IEEE arithmetic on the
//! components; no tolerance is assumed.

const std = @import("std");
const matrix = @import("matrix.zig");

/// Returns a row vector with `N` components of `T`.
///
/// This is the `1xN` case of `Matrix`; see
/// `matrix.Matrix` for the general form.
///
/// # Args
/// - `T`: an integer or float component type.
/// - `N`: the component count; at least 1.
///
/// # Return
/// A struct holding `N` components.
///
/// # Note
/// A vector is a matrix with one row, so it also
/// offers matrix-style construction and
/// multiplication against matrices.
///
/// # Compile Error
/// - `N` is zero.
/// - `T` is not an integer or float.
pub fn Vector(comptime T: type, comptime N: usize) type {
    if (N == 0) {
        @compileError("Vector: `N` must be at least 1");
    }
    switch (@typeInfo(T)) {
        .int, .float => {},
        else => @compileError(
            "Vector: `T` must be an integer or float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Components in column order.
        elements: @Vector(N, T),

        const Self = @This();

        /// The component type.
        pub const Component = T;
        /// The number of components.
        pub const count: usize = N;
        /// Raw storage of the components.
        pub const Storage = @Vector(N, T);
        /// Row count of the `1xN` matrix view.
        pub const row_count: usize = 1;
        /// Column count of the `1xN` matrix view.
        pub const column_count: usize = N;

        /// Builds a vector from its components.
        ///
        /// # Args
        /// - `elements`: the component values.
        ///
        /// # Return
        /// A vector holding `elements`.
        pub fn fromElements(elements: Storage) Self {
            return .{ .elements = elements };
        }

        /// Builds a vector by replicating a scalar.
        ///
        /// # Args
        /// - `value`: the value copied to every lane.
        ///
        /// # Return
        /// A vector full of `value`.
        pub fn splat(value: T) Self {
            return .{ .elements = @splat(value) };
        }

        /// Returns the additive identity.
        ///
        /// # Return
        /// A vector with every component zero.
        pub fn zero() Self {
            return Self.splat(0);
        }

        /// Reads one component.
        ///
        /// # Args
        /// - `index`: a valid component index.
        ///
        /// # Return
        /// The component at `index`.
        ///
        /// # Compile Error
        /// - `index` is at least `count`.
        pub fn at(self: Self, comptime index: usize) T {
            if (index >= N) {
                @compileError("at: index out of range");
            }
            return self.elements[index];
        }

        /// Writes one component.
        ///
        /// # Args
        /// - `index`: a valid component index.
        /// - `value`: the new component value.
        ///
        /// # Compile Error
        /// - `index` is at least `count`.
        pub fn set(
            self: *Self,
            comptime index: usize,
            value: T,
        ) void {
            if (index >= N) {
                @compileError("set: index out of range");
            }
            self.elements[index] = value;
        }

        /// Adds two vectors componentwise.
        ///
        /// # Args
        /// - `other`: the vector to add.
        ///
        /// # Return
        /// The componentwise sum.
        pub fn add(self: Self, other: Self) Self {
            return .{ .elements = self.elements + other.elements };
        }

        /// Subtracts two vectors componentwise.
        ///
        /// # Args
        /// - `other`: the vector to subtract.
        ///
        /// # Return
        /// The componentwise difference.
        pub fn sub(self: Self, other: Self) Self {
            return .{ .elements = self.elements - other.elements };
        }

        /// Negates every component.
        ///
        /// # Return
        /// The componentwise negation.
        pub fn negate(self: Self) Self {
            return .{ .elements = -self.elements };
        }

        /// Multiplies two vectors componentwise.
        ///
        /// # Args
        /// - `other`: the vector to multiply.
        ///
        /// # Return
        /// The Hadamard product.
        pub fn mulElements(self: Self, other: Self) Self {
            return .{ .elements = self.elements * other.elements };
        }

        /// Divides two vectors componentwise.
        ///
        /// # Args
        /// - `other`: the divisor.
        ///
        /// # Return
        /// The componentwise quotient.
        ///
        /// # Note
        /// A zero divisor produces an infinity or NaN.
        pub fn divElements(self: Self, other: Self) Self {
            return .{ .elements = self.elements / other.elements };
        }

        /// Scales every component.
        ///
        /// # Args
        /// - `factor`: the multiplier.
        ///
        /// # Return
        /// The scaled vector.
        pub fn scale(self: Self, factor: T) Self {
            return .{
                .elements = self.elements *
                    @as(Storage, @splat(factor)),
            };
        }

        /// Computes the dot product.
        ///
        /// # Args
        /// - `other`: the vector to multiply with.
        ///
        /// # Return
        /// The sum of the componentwise products.
        pub fn dot(self: Self, other: Self) T {
            return self.dotElements(other.elements);
        }

        /// Computes the three-dimensional cross product.
        ///
        /// # Args
        /// - `other`: the second vector.
        ///
        /// # Return
        /// `self x other`, using a right-handed basis.
        ///
        /// # Compile Error
        /// - `count` is not 3.
        pub fn cross(self: Self, other: Self) Self {
            if (N != 3) {
                @compileError(
                    "cross: the vector must have " ++
                        "3 components",
                );
            }
            return .{ .elements = .{
                self.elements[1] * other.elements[2] -
                    self.elements[2] * other.elements[1],
                self.elements[2] * other.elements[0] -
                    self.elements[0] * other.elements[2],
                self.elements[0] * other.elements[1] -
                    self.elements[1] * other.elements[0],
            } };
        }

        /// Returns the squared length.
        ///
        /// # Return
        /// The sum of the squared components.
        pub fn lengthSquared(self: Self) T {
            return @reduce(.Add, self.elements * self.elements);
        }

        /// Returns the Euclidean length.
        ///
        /// # Return
        /// The square root of `lengthSquared`.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn length(self: Self) T {
            assertFloat(T);
            return @sqrt(self.lengthSquared());
        }

        /// Normalizes the vector.
        ///
        /// # Return
        /// A unit vector with the same direction.
        ///
        /// # Note
        /// A zero or infinite input yields a non-finite
        /// result; use `normalizeSafe` when that is
        /// possible.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn normalize(self: Self) Self {
            assertFloat(T);
            const len = self.length();
            return .{
                .elements = self.elements /
                    @as(Storage, @splat(len)),
            };
        }

        /// Normalizes a vector unless it is too short.
        ///
        /// # Args
        /// - `epsilon_value`: smallest normalized
        ///   length.
        ///
        /// # Return
        /// A unit vector, or a zero vector when the
        /// length is at most `epsilon_value`.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn normalizeSafe(
            self: Self,
            epsilon_value: T,
        ) Self {
            assertFloat(T);
            if (!self.isNormalizable(epsilon_value)) {
                return Self.zero();
            }
            return self.normalize();
        }

        /// Returns the squared distance to a point.
        ///
        /// # Args
        /// - `other`: the other point.
        ///
        /// # Return
        /// The squared length of the difference.
        pub fn distanceSquared(self: Self, other: Self) T {
            return self.sub(other).lengthSquared();
        }

        /// Returns the distance to a point.
        ///
        /// # Args
        /// - `other`: the other point.
        ///
        /// # Return
        /// The Euclidean distance.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn distance(self: Self, other: Self) T {
            assertFloat(T);
            return self.sub(other).length();
        }

        /// Reflects a vector about a normal.
        ///
        /// # Args
        /// - `normal`: the unit surface normal.
        ///
        /// # Return
        /// The reflected direction.
        ///
        /// # Note
        /// `normal` should be a unit vector.
        pub fn reflect(self: Self, normal: Self) Self {
            return self.sub(
                normal.scale(2 * self.dot(normal)),
            );
        }

        /// Refracts a vector through a surface.
        ///
        /// # Args
        /// - `normal`: the unit surface normal,
        ///   opposed to `self`.
        /// - `eta`: incident over transmitted index of
        ///   refraction.
        ///
        /// # Return
        /// The refracted direction, or `null` on total
        /// internal reflection.
        ///
        /// # Note
        /// Both directions must be unit vectors.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn refract(
            self: Self,
            normal: Self,
            eta: T,
        ) ?Self {
            assertFloat(T);
            const cos_i = normal.dot(self);
            const k = 1 - eta * eta * (1 - cos_i * cos_i);
            if (k < 0) return null;
            return self.scale(eta).sub(normal.scale(
                eta * cos_i + @sqrt(k),
            ));
        }

        /// Rearranges components by a compile-time
        /// mask.
        ///
        /// # Args
        /// - `Mask`: component indices; every entry
        ///   must be in `0..count`.
        ///
        /// # Return
        /// A vector whose lane `i` is `self[Mask[i]]`.
        ///
        /// # Note
        /// The result length is the mask length, so it
        /// may differ from the source length.
        ///
        /// # Compile Error
        /// - A mask entry is negative or out of range.
        pub fn swizzle(
            self: Self,
            comptime Mask: []const i32,
        ) Vector(T, Mask.len) {
            const indices = comptime blk: {
                var result: [Mask.len]i32 = undefined;
                for (Mask, 0..) |index, i| {
                    if (index < 0 or index >= N) {
                        @compileError(
                            "swizzle: index out of range",
                        );
                    }
                    result[i] = index;
                }
                break :blk result;
            };
            return .{ .elements = @shuffle(
                T,
                self.elements,
                @as(Storage, undefined),
                indices,
            ) };
        }

        /// Clamps every component into a range.
        ///
        /// # Args
        /// - `lower`: minimum per component.
        /// - `upper`: maximum per component.
        ///
        /// # Return
        /// The componentwise clamp.
        pub fn clamp(self: Self, lower: Self, upper: Self) Self {
            return .{ .elements = @min(
                @max(self.elements, lower.elements),
                upper.elements,
            ) };
        }

        /// Clamps every component into `0..1`.
        ///
        /// # Return
        /// The componentwise clamp to zero and one.
        pub fn saturate(self: Self) Self {
            return self.clamp(Self.zero(), Self.splat(1));
        }

        /// Reports whether every component is near
        /// zero.
        ///
        /// # Args
        /// - `epsilon_value`: non-negative tolerance.
        ///
        /// # Return
        /// `true` when every component is within
        /// `epsilon_value` of zero.
        pub fn isZero(self: Self, epsilon_value: T) bool {
            return @reduce(.And, @abs(self.elements) <=
                @as(Storage, @splat(epsilon_value)));
        }

        /// Reports whether the length is usable.
        ///
        /// # Args
        /// - `epsilon_value`: smallest accepted length.
        ///
        /// # Return
        /// `true` when the length exceeds
        /// `epsilon_value`.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn isNormalizable(
            self: Self,
            epsilon_value: T,
        ) bool {
            assertFloat(T);
            return self.length() > epsilon_value;
        }

        /// Reports whether any component is NaN.
        ///
        /// # Return
        /// `true` when at least one lane is NaN.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn isNan(self: Self) bool {
            assertFloat(T);
            return @reduce(.Or, self.elements != self.elements);
        }

        /// Reports whether any component is infinite.
        ///
        /// # Return
        /// `true` when at least one lane is `+inf` or
        /// `-inf`.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn isInf(self: Self) bool {
            assertFloat(T);
            return @reduce(.Or, @abs(self.elements) ==
                @as(Storage, @splat(std.math.inf(T))));
        }

        /// Reports whether every component is finite.
        ///
        /// # Return
        /// `true` when no lane is NaN or infinite.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn isFinite(self: Self) bool {
            assertFloat(T);
            return !self.isNan() and !self.isInf();
        }

        /// Compares vectors by absolute difference.
        ///
        /// # Args
        /// - `other`: the vector to compare with.
        /// - `epsilon_value`: non-negative tolerance.
        ///
        /// # Return
        /// `true` when every lane is within the
        /// tolerance.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn approxEqAbs(
            self: Self,
            other: Self,
            epsilon_value: T,
        ) bool {
            assertFloat(T);
            const difference = @abs(self.elements -
                other.elements);
            return @reduce(.And, difference <=
                @as(Storage, @splat(epsilon_value)));
        }

        /// Compares vectors by relative difference.
        ///
        /// # Args
        /// - `other`: the vector to compare with.
        /// - `epsilon_value`: positive tolerance.
        ///
        /// # Return
        /// `true` when every lane is relatively
        /// close.
        ///
        /// # Compile Error
        /// - The components are not floats.
        pub fn approxEqRel(
            self: Self,
            other: Self,
            epsilon_value: T,
        ) bool {
            assertFloat(T);
            const difference = @abs(self.elements -
                other.elements);
            const tolerance = @max(
                @abs(self.elements),
                @abs(other.elements),
            ) * @as(Storage, @splat(epsilon_value));
            return @reduce(.And, difference <= tolerance);
        }

        /// Multiplies this row vector by a matrix.
        ///
        /// # Args
        /// - `ResultCols`: the matrix column count.
        /// - `m`: a matrix with `count` rows.
        ///
        /// # Return
        /// The resulting row vector.
        ///
        /// # Compile Error
        /// - `m` does not have `count` rows.
        pub fn mulMatrix(
            self: Self,
            comptime ResultCols: usize,
            m: matrix.Matrix(T, N, ResultCols),
        ) Vector(T, ResultCols) {
            var result: Vector(T, ResultCols) = undefined;
            if (N == 1) {
                inline for (0..ResultCols) |c| {
                    result.elements[c] =
                        self.elements[0] * m.elements[c];
                }
                return result;
            }
            inline for (0..ResultCols) |c| {
                result.elements[c] = self.dotElements(
                    m.column(c),
                );
            }
            return result;
        }

        /// Transposes the row vector into a column
        /// matrix.
        ///
        /// # Return
        /// A matrix with `count` rows and one column.
        pub fn transpose(self: Self) matrix.Matrix(T, N, 1) {
            if (N == 1) return self;
            var result: matrix.Matrix(T, N, 1) = undefined;
            inline for (0..N) |index| {
                result.columns[0][index] = self.elements[index];
            }
            return result;
        }

        fn dotElements(self: Self, raw: Storage) T {
            return @reduce(.Add, self.elements * raw);
        }
    };
}

/// Returns a two-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `1x2` matrix type.
pub fn Vec2(comptime T: type) type {
    return Vector(T, 2);
}

/// Returns a three-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `1x3` matrix type.
pub fn Vec3(comptime T: type) type {
    return Vector(T, 3);
}

/// Returns a four-component vector of `T`.
///
/// # Args
/// - `T`: an integer or float component type.
///
/// # Return
/// The `1x4` matrix type.
pub fn Vec4(comptime T: type) type {
    return Vector(T, 4);
}

/// Half-precision two-component vector.
pub const Half2 = Vec2(f16);
/// Half-precision three-component vector.
pub const Half3 = Vec3(f16);
/// Half-precision four-component vector.
pub const Half4 = Vec4(f16);
/// Signed 32-bit two-component vector.
pub const IVec2 = Vec2(i32);
/// Signed 32-bit three-component vector.
pub const IVec3 = Vec3(i32);
/// Signed 32-bit four-component vector.
pub const IVec4 = Vec4(i32);
/// Unsigned 32-bit two-component vector.
pub const UVec2 = Vec2(u32);
/// Unsigned 32-bit three-component vector.
pub const UVec3 = Vec3(u32);
/// Unsigned 32-bit four-component vector.
pub const UVec4 = Vec4(u32);

fn assertFloat(comptime T: type) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "expected float components, got `" ++
                @typeName(T) ++ "`",
        ),
    }
}

test "vector construction" {
    const value = Vec3(f32).fromElements(.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(f32, 1), value.at(0));
    try std.testing.expectEqual(@as(f32, 3), value.at(2));
    try std.testing.expectEqual(@as(f32, 2), value.elements[1]);

    var mutable = value;
    mutable.set(1, 5);
    try std.testing.expectEqual(@as(f32, 5), mutable.at(1));

    const repeated = Vec2(i32).splat(7);
    try std.testing.expectEqual(@as(i32, 7), repeated.at(1));
    try std.testing.expectEqual(@as(i32, 0), Vec2(i32).zero().at(0));
}

test "vector arithmetic" {
    const a = Vec3(f32).fromElements(.{ 1, 2, 3 });
    const b = Vec3(f32).fromElements(.{ 4, 5, 6 });
    const sum = a.add(b);
    try std.testing.expectEqual(@as(f32, 5), sum.at(0));
    try std.testing.expectEqual(@as(f32, 9), sum.at(2));
    const difference = b.sub(a);
    try std.testing.expectEqual(@as(f32, 3), difference.at(0));
    const negated = a.negate();
    try std.testing.expectEqual(@as(f32, -1), negated.at(0));
    const product = a.mulElements(b);
    try std.testing.expectEqual(@as(f32, 4), product.at(0));
    try std.testing.expectEqual(@as(f32, 18), product.at(2));
    const quotient = b.divElements(a);
    try std.testing.expectEqual(@as(f32, 4), quotient.at(0));
    const scaled = a.scale(2);
    try std.testing.expectEqual(@as(f32, 6), scaled.at(2));
}

test "dot and cross" {
    const a = Vec3(f32).fromElements(.{ 1, 2, 3 });
    const b = Vec3(f32).fromElements(.{ 4, 5, 6 });
    try std.testing.expectEqual(@as(f32, 32), a.dot(b));

    const unit_x = Vec3(f32).fromElements(.{ 1, 0, 0 });
    const unit_y = Vec3(f32).fromElements(.{ 0, 1, 0 });
    const z = unit_x.cross(unit_y);
    try std.testing.expectEqual(@as(f32, 0), z.at(0));
    try std.testing.expectEqual(@as(f32, 1), z.at(2));
    try std.testing.expectEqual(@as(f32, 0), z.dot(unit_y));
}

test "length and normalization" {
    const value = Vec3(f32).fromElements(.{ 3, 4, 0 });
    try std.testing.expectEqual(@as(f32, 25), value.lengthSquared());
    try std.testing.expectEqual(@as(f32, 5), value.length());
    try std.testing.expectEqual(@as(f32, 5), value.distance(
        Vec3(f32).zero(),
    ));
    try std.testing.expectEqual(@as(f32, 25), value.distanceSquared(
        Vec3(f32).zero(),
    ));

    const unit = value.normalize();
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        unit.length(),
        1e-6,
    );
    const safe = Vec3(f32).zero().normalizeSafe(1e-6);
    try std.testing.expectEqual(@as(f32, 0), safe.at(0));
    try std.testing.expect(value.isNormalizable(1e-6));
    try std.testing.expect(!Vec3(f32).zero().isNormalizable(1e-6));
}

test "reflect and refract" {
    const down = Vec3(f32)
        .fromElements(.{ 1, -1, 0 })
        .normalize();
    const up = Vec3(f32).fromElements(.{ 0, 1, 0 });
    const reflected = down.reflect(up);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.70710678),
        reflected.at(1),
        1e-6,
    );

    const incident = Vec3(f32).fromElements(.{ 0, -1, 0 });
    const refracted = incident.refract(up, 0.5).?;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        refracted.length(),
        1e-6,
    );
    const sideways = Vec3(f32).fromElements(.{ 1, 0, 0 });
    try std.testing.expect(sideways.refract(up, 2) == null);
}

test "swizzle and clamp" {
    const value = Vec4(f32).fromElements(.{ 10, 20, 30, 40 });
    const xy = value.swizzle(&.{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 10), xy.at(0));
    try std.testing.expectEqual(@as(f32, 20), xy.at(1));
    const reversed = value.swizzle(&.{ 3, 1 });
    try std.testing.expectEqual(@as(f32, 40), reversed.at(0));

    const mixed = Vec3(f32).fromElements(.{ -1, 0.5, 3 });
    const clamped = mixed.clamp(
        Vec3(f32).zero(),
        Vec3(f32).splat(1),
    );
    try std.testing.expectEqual(@as(f32, 0), clamped.at(0));
    try std.testing.expectEqual(@as(f32, 1), clamped.at(2));
    try std.testing.expectEqual(
        @as(f32, 1),
        mixed.saturate().at(2),
    );
}

test "vector predicates" {
    const value = Vec3(f32).fromElements(.{ 1, 2, 3 });
    try std.testing.expect(value.isFinite());
    try std.testing.expect(!value.isNan());
    try std.testing.expect(!value.isInf());
    try std.testing.expect(!value.isZero(1e-6));
    try std.testing.expect(
        Vec3(f32).fromElements(.{ 0, 0, 1e-9 }).isZero(1e-6),
    );
    try std.testing.expect(
        value.approxEqAbs(
            Vec3(f32).fromElements(.{ 1, 2, 3.00001 }),
            1e-4,
        ),
    );
    try std.testing.expect(
        !value.approxEqRel(
            Vec3(f32).fromElements(.{ 1, 2, 4 }),
            1e-4,
        ),
    );
    const nan = Vec3(f32).fromElements(.{
        std.math.nan(f32),
        0,
        0,
    });
    try std.testing.expect(nan.isNan());
    try std.testing.expect(!nan.isFinite());
}
