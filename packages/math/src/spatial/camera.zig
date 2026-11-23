//! Camera and projection matrix factories.
//!
//! The matrices are row-vector matrices: transform a
//! point with `Matrix.transformPoint`. View space is
//! right-handed and looks down negative Z. Clip depth
//! spans `0..1`, matching Vulkan and D3D rather than
//! OpenGL.

const std = @import("std");
const vector = @import("../linear/vector.zig");
const matrix = @import("../linear/matrix.zig");

/// Returns a two-dimensional viewport type.
///
/// # Args
/// - `T`: the coordinate component type.
///
/// # Return
/// A struct holding a pixel width and height.
pub fn Viewport(comptime T: type) type {
    return struct {
        /// Viewport width in pixels.
        width: T,
        /// Viewport height in pixels.
        height: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Camera and projection factories for `T`.
///
/// # Args
/// - `T`: the component type; a float.
///
/// # Compile Error
/// - `T` is not a float type.
pub fn Camera(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError(
            "Camera: `T` must be a float, got `" ++
                @typeName(T) ++ "`",
        ),
    }

    return struct {
        /// Builds a right-handed view matrix.
        ///
        /// # Args
        /// - `eye`: camera position in world space.
        /// - `target`: point the camera looks at.
        /// - `up`: approximate up direction; it must
        ///   not be parallel to the view direction.
        /// - `epsilon_value`: smallest accepted cross
        ///   product length.
        ///
        /// # Return
        /// A world-to-view matrix, or `null` when the
        /// direction degenerates.
        ///
        /// # Note
        /// The camera looks down negative Z, so visible
        /// points have a negative view-space `z`.
        pub fn lookAt(
            eye: vector.Vector(T, 3),
            target: vector.Vector(T, 3),
            up: vector.Vector(T, 3),
            epsilon_value: T,
        ) ?matrix.Mat4(T) {
            const forward_raw = target.sub(eye);
            if (!forward_raw.isNormalizable(epsilon_value)) {
                return null;
            }
            const forward = forward_raw.normalize();

            const right_raw = forward.cross(up);
            if (!right_raw.isNormalizable(epsilon_value)) {
                return null;
            }
            const right = right_raw.normalize();
            const camera_up = right.cross(forward);

            const eye_right = right.dot(eye);
            const eye_up = camera_up.dot(eye);
            const eye_forward = forward.dot(eye);

            return matrix.Mat4(T).fromColumns(.{
                .{
                    right.at(0),
                    right.at(1),
                    right.at(2),
                    -eye_right,
                },
                .{
                    camera_up.at(0),
                    camera_up.at(1),
                    camera_up.at(2),
                    -eye_up,
                },
                .{
                    -forward.at(0),
                    -forward.at(1),
                    -forward.at(2),
                    eye_forward,
                },
                .{ 0, 0, 0, 1 },
            });
        }

        /// Builds a standard perspective projection.
        ///
        /// The near plane maps to depth 0 and the far
        /// plane to depth 1.
        ///
        /// # Args
        /// - `fov_y`: vertical field of view in radians.
        /// - `aspect`: width divided by height.
        /// - `near`: distance to the near plane.
        /// - `far`: distance to the far plane.
        ///
        /// # Return
        /// A view-to-clip matrix.
        ///
        /// # Note
        /// Perspective division happens after the
        /// multiplication, so clip `w` is `-z`.
        ///
        /// # Panic
        /// A safety check fails when `fov_y` is outside
        /// `(0, pi)`, or when `aspect` or `near` is not
        /// positive, or when `far` is not greater than
        /// `near`.
        pub fn perspective(
            fov_y: T,
            aspect: T,
            near: T,
            far: T,
        ) matrix.Mat4(T) {
            std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
            std.debug.assert(aspect > 0);
            std.debug.assert(near > 0);
            std.debug.assert(far > near);

            const focal = 1 / @tan(fov_y / 2);
            const range = 1 / (near - far);
            return matrix.Mat4(T).fromRows(.{
                vector.Vec4(T).fromElements(.{
                    focal / aspect,
                    0,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{ 0, focal, 0, 0 }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    far * range,
                    -1,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    near * far * range,
                    0,
                }),
            });
        }

        /// Builds a reversed-depth projection.
        ///
        /// The near plane maps to depth 1 and the far
        /// plane to depth 0.
        ///
        /// # Args
        /// - `fov_y`: vertical field of view in radians.
        /// - `aspect`: width divided by height.
        /// - `near`: distance to the near plane.
        /// - `far`: distance to the far plane.
        ///
        /// # Return
        /// A view-to-clip matrix.
        ///
        /// # Panic
        /// A safety check fails when the arguments are
        /// outside the ranges listed on `perspective`.
        pub fn perspectiveReversed(
            fov_y: T,
            aspect: T,
            near: T,
            far: T,
        ) matrix.Mat4(T) {
            std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
            std.debug.assert(aspect > 0);
            std.debug.assert(near > 0);
            std.debug.assert(far > near);

            const focal = 1 / @tan(fov_y / 2);
            const range = 1 / (far - near);
            return matrix.Mat4(T).fromRows(.{
                vector.Vec4(T).fromElements(.{
                    focal / aspect,
                    0,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{ 0, focal, 0, 0 }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    near * range,
                    -1,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    near * far * range,
                    0,
                }),
            });
        }

        /// Builds a projection with an infinite far
        /// plane.
        ///
        /// The near plane maps to depth 0 and depth
        /// approaches 1 as distance grows.
        ///
        /// # Args
        /// - `fov_y`: vertical field of view in radians.
        /// - `aspect`: width divided by height.
        /// - `near`: distance to the near plane.
        ///
        /// # Return
        /// A view-to-clip matrix.
        ///
        /// # Panic
        /// A safety check fails when `fov_y` is outside
        /// `(0, pi)`, or when `aspect` or `near` is not
        /// positive.
        pub fn perspectiveInfinite(
            fov_y: T,
            aspect: T,
            near: T,
        ) matrix.Mat4(T) {
            std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
            std.debug.assert(aspect > 0);
            std.debug.assert(near > 0);

            const focal = 1 / @tan(fov_y / 2);
            return matrix.Mat4(T).fromRows(.{
                vector.Vec4(T).fromElements(.{
                    focal / aspect,
                    0,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{ 0, focal, 0, 0 }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    -1,
                    -1,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    -near,
                    0,
                }),
            });
        }

        /// Builds a reversed-depth projection with an
        /// infinite far plane.
        ///
        /// The near plane maps to depth 1 and depth
        /// approaches 0 as distance grows.
        ///
        /// # Args
        /// - `fov_y`: vertical field of view in radians.
        /// - `aspect`: width divided by height.
        /// - `near`: distance to the near plane.
        ///
        /// # Return
        /// A view-to-clip matrix.
        ///
        /// # Panic
        /// A safety check fails when `fov_y` is outside
        /// `(0, pi)`, or when `aspect` or `near` is not
        /// positive.
        pub fn perspectiveInfiniteReversed(
            fov_y: T,
            aspect: T,
            near: T,
        ) matrix.Mat4(T) {
            std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
            std.debug.assert(aspect > 0);
            std.debug.assert(near > 0);

            const focal = 1 / @tan(fov_y / 2);
            return matrix.Mat4(T).fromRows(.{
                vector.Vec4(T).fromElements(.{
                    focal / aspect,
                    0,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{ 0, focal, 0, 0 }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    0,
                    -1,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    near,
                    0,
                }),
            });
        }

        /// Builds a right-handed orthographic
        /// projection.
        ///
        /// # Args
        /// - `left`: minimum view-space x.
        /// - `right`: maximum view-space x.
        /// - `bottom`: minimum view-space y.
        /// - `top`: maximum view-space y.
        /// - `near`: distance to the near plane.
        /// - `far`: distance to the far plane.
        ///
        /// # Return
        /// A view-to-clip matrix.
        ///
        /// # Panic
        /// A safety check fails when an extent or the
        /// depth range is empty.
        pub fn orthographic(
            left: T,
            right: T,
            bottom: T,
            top: T,
            near: T,
            far: T,
        ) matrix.Mat4(T) {
            std.debug.assert(right != left);
            std.debug.assert(top != bottom);
            std.debug.assert(far > near);

            const width = right - left;
            const height = top - bottom;
            const depth = far - near;
            return matrix.Mat4(T).fromRows(.{
                vector.Vec4(T).fromElements(.{
                    2 / width,
                    0,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    2 / height,
                    0,
                    0,
                }),
                vector.Vec4(T).fromElements(.{
                    0,
                    0,
                    -1 / depth,
                    0,
                }),
                vector.Vec4(T).fromElements(.{
                    -(left + right) / width,
                    -(top + bottom) / height,
                    -near / depth,
                    1,
                }),
            });
        }

        /// Maps a screen position to normalized device
        /// coordinates.
        ///
        /// # Args
        /// - `viewport`: pixel size of the target.
        /// - `screen`: pixel position; origin top left.
        /// - `depth`: normalized depth in `0..1`.
        ///
        /// # Return
        /// The NDC position with Y pointing down.
        ///
        /// # Panic
        /// A safety check fails when a viewport extent
        /// is zero.
        pub fn toNdc(
            viewport: Viewport(T),
            screen: vector.Vector(T, 2),
            depth: T,
        ) vector.Vector(T, 3) {
            std.debug.assert(viewport.width > 0);
            std.debug.assert(viewport.height > 0);
            return vector.Vector(T, 3).fromElements(.{
                2 * screen.at(0) / viewport.width - 1,
                2 * screen.at(1) / viewport.height - 1,
                depth,
            });
        }

        /// Transforms an NDC position to world space.
        ///
        /// # Args
        /// - `inverse_view_projection`: inverse of the
        ///   matrix that produced the clip position.
        /// - `ndc`: normalized device coordinates.
        ///
        /// # Return
        /// The world-space position.
        ///
        /// # Note
        /// The homogeneous `w` is divided out when it
        /// is not zero; otherwise the raw result is
        /// returned.
        pub fn unproject(
            inverse_view_projection: matrix.Mat4(T),
            ndc: vector.Vector(T, 3),
        ) vector.Vector(T, 3) {
            const clip = vector.Vec4(T)
                .fromElements(.{
                    ndc.at(0),
                    ndc.at(1),
                    ndc.at(2),
                    1,
                })
                .mulMatrix(4, inverse_view_projection);
            if (clip.at(3) == 0) {
                return vector.Vector(T, 3).fromElements(.{
                    clip.at(0),
                    clip.at(1),
                    clip.at(2),
                });
            }
            const w = clip.at(3);
            return vector.Vector(T, 3).fromElements(.{
                clip.at(0) / w,
                clip.at(1) / w,
                clip.at(2) / w,
            });
        }

        /// Maps a screen position to world space.
        ///
        /// # Args
        /// - `inverse_view_projection`: inverse of the
        ///   matrix that produced the clip position.
        /// - `viewport`: pixel size of the target.
        /// - `screen`: pixel position; origin top left.
        /// - `depth`: normalized depth in `0..1`.
        ///
        /// # Return
        /// The world-space position.
        ///
        /// # Note
        /// This combines `toNdc` and `unproject`.
        ///
        /// # Panic
        /// A safety check fails when a viewport extent
        /// is zero.
        pub fn unprojectScreen(
            inverse_view_projection: matrix.Mat4(T),
            viewport: Viewport(T),
            screen: vector.Vector(T, 2),
            depth: T,
        ) vector.Vector(T, 3) {
            return unproject(
                inverse_view_projection,
                toNdc(viewport, screen, depth),
            );
        }
    };
}

test "look at" {
    const Vec3f = vector.Vec3(f32);
    const eye = Vec3f.fromElements(.{ 0, 0, 5 });
    const view = Camera(f32).lookAt(
        eye,
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 1, 0 }),
        1e-6,
    ).?;

    const origin = view.transformPoint(Vec3f.zero());
    try std.testing.expectApproxEqAbs(@as(f32, -5), origin.at(2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), origin.at(0), 1e-6);

    const camera_position = view.transformPoint(eye);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        camera_position.at(2),
        1e-6,
    );

    try std.testing.expect(Camera(f32).lookAt(
        eye,
        eye,
        Vec3f.fromElements(.{ 0, 1, 0 }),
        1e-6,
    ) == null);
    try std.testing.expect(Camera(f32).lookAt(
        eye,
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 0, -1 }),
        1e-6,
    ) == null);
}

test "perspective depth range" {
    const Vec3f = vector.Vec3(f32);
    const projection = Camera(f32).perspective(
        @as(f32, std.math.pi) * 0.5,
        1,
        0.1,
        100,
    );
    const near_point = projection.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -0.1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_point.at(2),
        1e-5,
    );
    const far_point = projection.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -100 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        far_point.at(2),
        1e-4,
    );

    const reversed = Camera(f32).perspectiveReversed(
        @as(f32, std.math.pi) * 0.5,
        1,
        0.1,
        100,
    );
    const near_reversed = reversed.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -0.1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        near_reversed.at(2),
        1e-5,
    );
    const far_reversed = reversed.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -100 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        far_reversed.at(2),
        1e-4,
    );
}

test "infinite perspective" {
    const Vec3f = vector.Vec3(f32);
    const projection = Camera(f32).perspectiveInfinite(
        @as(f32, std.math.pi) * 0.5,
        1,
        0.1,
    );
    const near_point = projection.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -0.1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_point.at(2),
        1e-5,
    );
    const far_point = projection.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -10000 }),
    );
    try std.testing.expect(far_point.at(2) > 0.99);

    const reversed = Camera(f32).perspectiveInfiniteReversed(
        @as(f32, std.math.pi) * 0.5,
        1,
        0.1,
    );
    const near_reversed = reversed.transformPoint(
        Vec3f.fromElements(.{ 0, 0, -0.1 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        near_reversed.at(2),
        1e-5,
    );
}

test "orthographic corners" {
    const Vec3f = vector.Vec3(f32);
    const projection = Camera(f32).orthographic(
        -2,
        2,
        -1,
        1,
        0.5,
        10,
    );
    const near_corner = projection.transformPoint(
        Vec3f.fromElements(.{ -2, -1, -0.5 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1),
        near_corner.at(0),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_corner.at(2),
        1e-6,
    );
    const far_corner = projection.transformPoint(
        Vec3f.fromElements(.{ 2, 1, -10 }),
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        far_corner.at(2),
        1e-6,
    );
}

test "unproject round trip" {
    const Vec3f = vector.Vec3(f32);
    const view = Camera(f32).lookAt(
        Vec3f.fromElements(.{ 1, 2, 3 }),
        Vec3f.zero(),
        Vec3f.fromElements(.{ 0, 1, 0 }),
        1e-6,
    ).?;
    const projection = Camera(f32).perspective(
        @as(f32, std.math.pi) / 3,
        1.5,
        0.1,
        100,
    );
    const view_projection = view.mul(projection);
    const inverse_vp = view_projection.inverse(1e-9).?;

    const viewport = Viewport(f32){ .width = 800, .height = 600 };
    const screen = vector.Vec2(f32).fromElements(.{ 400, 300 });
    const depth: f32 = 0.4;
    const world = Camera(f32).unprojectScreen(
        inverse_vp,
        viewport,
        screen,
        depth,
    );
    const clip = vector.Vec4(f32)
        .fromElements(.{ world.at(0), world.at(1), world.at(2), 1 })
        .mulMatrix(4, view_projection);
    try std.testing.expectApproxEqAbs(
        0,
        clip.at(0) / clip.at(3),
        1e-3,
    );
    try std.testing.expectApproxEqAbs(
        depth,
        clip.at(2) / clip.at(3),
        1e-3,
    );

    const ndc = Camera(f32).toNdc(viewport, screen, depth);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ndc.at(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ndc.at(1), 1e-6);
}
