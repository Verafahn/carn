//! Camera and projection matrix factories.
//!
//! The matrices follow the row-vector convention:
//! transform a point with `mat.mulVec(m, p)`. View
//! space is right-handed and looks down negative Z.
//! Clip depth spans `0..1`, matching Vulkan and D3D
//! rather than OpenGL.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const Vec3f = vec.Vec3(f32);

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

/// Builds a right-handed view matrix.
///
/// # Args
/// - `eye`: camera position in world space.
/// - `target`: point the camera looks at.
/// - `up`: approximate up direction; must not be
///   parallel to the view direction.
/// - `epsilon`: smallest accepted cross product
///   length.
///
/// # Return
/// A world-to-view matrix, or `null` when the
/// direction or the up vector degenerates.
///
/// # Note
/// The camera looks down negative Z, so a visible
/// point has a negative view-space `z`.
///
/// # Compile Error
/// - The arguments are not three-component
///   float vectors of the same type.
pub fn lookAt(
    eye: anytype,
    target: anytype,
    up: anytype,
    epsilon: anytype,
) ?mat.Mat4(scalar.ScalarOf(@TypeOf(eye))) {
    const eye_value = eye;
    const target_value: @TypeOf(eye) = target;
    const up_value: @TypeOf(eye) = up;

    const forward_raw = target_value - eye_value;
    if (!scalar.isNormalizable(forward_raw, epsilon)) return null;
    const forward = vec.normalize(forward_raw);

    const right_raw = vec.cross(forward, up_value);
    if (!scalar.isNormalizable(right_raw, epsilon)) return null;
    const right = vec.normalize(right_raw);
    const camera_up = vec.cross(right, forward);

    const eye_right = vec.dot(right, eye_value);
    const eye_up = vec.dot(camera_up, eye_value);
    const eye_forward = vec.dot(forward, eye_value);

    return .{ .columns = .{
        .{ right[0], right[1], right[2], -eye_right },
        .{ camera_up[0], camera_up[1], camera_up[2], -eye_up },
        .{
            -forward[0],
            -forward[1],
            -forward[2],
            eye_forward,
        },
        .{ 0, 0, 0, 1 },
    } };
}

/// Builds a standard perspective projection.
///
/// The near plane maps to depth 0 and the far plane
/// to depth 1.
///
/// # Args
/// - `fov_y`: vertical field of view in radians.
/// - `aspect`: viewport width divided by height.
/// - `near`: distance to the near plane.
/// - `far`: distance to the far plane.
///
/// # Return
/// A view-to-clip matrix.
///
/// # Note
/// Perspective division happens after the
/// multiplication, so the clip `w` is `-z`.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when `fov_y` is outside
/// `(0, pi)`, or when `aspect` is not positive, or
/// when `near` is not positive, or when `far` is not
/// greater than `near`.
pub fn perspective(
    fov_y: anytype,
    aspect: anytype,
    near: anytype,
    far: anytype,
) mat.Mat4(@TypeOf(fov_y)) {
    const T = @TypeOf(fov_y);
    const aspect_value: T = aspect;
    const near_value: T = near;
    const far_value: T = far;
    std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
    std.debug.assert(aspect_value > 0);
    std.debug.assert(near_value > 0);
    std.debug.assert(far_value > near_value);

    const focal = 1 / @tan(fov_y / 2);
    const range = 1 / (near_value - far_value);
    return mat.Mat4(T).fromRows(.{
        .{ focal / aspect_value, 0, 0, 0 },
        .{ 0, focal, 0, 0 },
        .{ 0, 0, far_value * range, -1 },
        .{ 0, 0, near_value * far_value * range, 0 },
    });
}

/// Builds a perspective projection with reversed
/// depth.
///
/// The near plane maps to depth 1 and the far plane
/// to depth 0, which pairs with a depth buffer that
/// compares greater depth as nearer.
///
/// # Args
/// - `fov_y`: vertical field of view in radians.
/// - `aspect`: viewport width divided by height.
/// - `near`: distance to the near plane.
/// - `far`: distance to the far plane.
///
/// # Return
/// A view-to-clip matrix.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when the arguments are
/// outside the ranges listed on `perspective`.
pub fn perspectiveReversed(
    fov_y: anytype,
    aspect: anytype,
    near: anytype,
    far: anytype,
) mat.Mat4(@TypeOf(fov_y)) {
    const T = @TypeOf(fov_y);
    const aspect_value: T = aspect;
    const near_value: T = near;
    const far_value: T = far;
    std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
    std.debug.assert(aspect_value > 0);
    std.debug.assert(near_value > 0);
    std.debug.assert(far_value > near_value);

    const focal = 1 / @tan(fov_y / 2);
    const range = 1 / (far_value - near_value);
    return mat.Mat4(T).fromRows(.{
        .{ focal / aspect_value, 0, 0, 0 },
        .{ 0, focal, 0, 0 },
        .{ 0, 0, near_value * range, -1 },
        .{ 0, 0, near_value * far_value * range, 0 },
    });
}

/// Builds a perspective projection with an infinite
/// far plane.
///
/// The near plane maps to depth 0 and depth
/// approaches 1 as distance grows.
///
/// # Args
/// - `fov_y`: vertical field of view in radians.
/// - `aspect`: viewport width divided by height.
/// - `near`: distance to the near plane.
///
/// # Return
/// A view-to-clip matrix.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when `fov_y` is outside
/// `(0, pi)`, or when `aspect` or `near` is not
/// positive.
pub fn perspectiveInfinite(
    fov_y: anytype,
    aspect: anytype,
    near: anytype,
) mat.Mat4(@TypeOf(fov_y)) {
    const T = @TypeOf(fov_y);
    const aspect_value: T = aspect;
    const near_value: T = near;
    std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
    std.debug.assert(aspect_value > 0);
    std.debug.assert(near_value > 0);

    const focal = 1 / @tan(fov_y / 2);
    return mat.Mat4(T).fromRows(.{
        .{ focal / aspect_value, 0, 0, 0 },
        .{ 0, focal, 0, 0 },
        .{ 0, 0, -1, -1 },
        .{ 0, 0, -near_value, 0 },
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
/// - `aspect`: viewport width divided by height.
/// - `near`: distance to the near plane.
///
/// # Return
/// A view-to-clip matrix.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when `fov_y` is outside
/// `(0, pi)`, or when `aspect` or `near` is not
/// positive.
pub fn perspectiveInfiniteReversed(
    fov_y: anytype,
    aspect: anytype,
    near: anytype,
) mat.Mat4(@TypeOf(fov_y)) {
    const T = @TypeOf(fov_y);
    const aspect_value: T = aspect;
    const near_value: T = near;
    std.debug.assert(fov_y > 0 and fov_y < std.math.pi);
    std.debug.assert(aspect_value > 0);
    std.debug.assert(near_value > 0);

    const focal = 1 / @tan(fov_y / 2);
    return mat.Mat4(T).fromRows(.{
        .{ focal / aspect_value, 0, 0, 0 },
        .{ 0, focal, 0, 0 },
        .{ 0, 0, 0, -1 },
        .{ 0, 0, near_value, 0 },
    });
}

/// Builds a right-handed orthographic projection.
///
/// The near plane maps to depth 0 and the far plane
/// to depth 1.
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
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when an extent or depth
/// range is empty.
pub fn orthographic(
    left: anytype,
    right: anytype,
    bottom: anytype,
    top: anytype,
    near: anytype,
    far: anytype,
) mat.Mat4(@TypeOf(left)) {
    const T = @TypeOf(left);
    const left_value = left;
    const right_value: T = right;
    const bottom_value: T = bottom;
    const top_value: T = top;
    const near_value: T = near;
    const far_value: T = far;
    std.debug.assert(right_value != left_value);
    std.debug.assert(top_value != bottom_value);
    std.debug.assert(far_value > near_value);

    const width = right_value - left_value;
    const height = top_value - bottom_value;
    const depth = far_value - near_value;
    return mat.Mat4(T).fromRows(.{
        .{ 2 / width, 0, 0, 0 },
        .{ 0, 2 / height, 0, 0 },
        .{ 0, 0, -1 / depth, 0 },
        .{
            -(left_value + right_value) / width,
            -(top_value + bottom_value) / height,
            -near_value / depth,
            1,
        },
    });
}

/// Maps a screen position to normalized device
/// coordinates.
///
/// # Args
/// - `viewport`: pixel size of the render target.
/// - `screen`: pixel position; origin at top left.
/// - `depth`: normalized depth in `0..1`.
///
/// # Return
/// The NDC position with Y pointing down.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when a viewport extent is
/// zero.
pub fn toNdc(
    viewport: anytype,
    screen: anytype,
    depth: anytype,
) @Vector(3, @TypeOf(depth)) {
    const T = @TypeOf(depth);
    const width: T = viewport.width;
    const height: T = viewport.height;
    std.debug.assert(width > 0 and height > 0);

    const screen_value: @Vector(2, T) = screen;
    return .{
        2 * screen_value[0] / width - 1,
        2 * screen_value[1] / height - 1,
        depth,
    };
}

/// Transforms an NDC position back to world space.
///
/// # Args
/// - `inverse_view_projection`: inverse of the
///   matrix that produced the clip position.
/// - `ndc`: normalized device coordinates in
///   `0..1` depth.
///
/// # Return
/// The world-space position.
///
/// # Note
/// The homogeneous `w` is divided out when it is not
/// zero; otherwise the raw result is returned.
///
/// # Compile Error
/// - The arguments are not four-component or
///   three-component float values of one type.
pub fn unproject(
    inverse_view_projection: anytype,
    ndc: anytype,
) @Vector(3, @TypeOf(ndc[0])) {
    const T = @TypeOf(ndc[0]);
    const ndc_value: @Vector(3, T) = ndc;
    const clip = mat.mulVec(
        inverse_view_projection,
        @Vector(4, T){ ndc_value[0], ndc_value[1], ndc_value[2], 1 },
    );
    if (clip[3] == 0) {
        return .{ clip[0], clip[1], clip[2] };
    }
    return .{
        clip[0] / clip[3],
        clip[1] / clip[3],
        clip[2] / clip[3],
    };
}

/// Maps a screen position to world space.
///
/// # Args
/// - `inverse_view_projection`: inverse of the
///   matrix that produced the clip position.
/// - `viewport`: pixel size of the render target.
/// - `screen`: pixel position; origin at top left.
/// - `depth`: normalized depth in `0..1`.
///
/// # Return
/// The world-space position.
///
/// # Note
/// This combines `toNdc` and `unproject`.
///
/// # Compile Error
/// - The arguments are not floats of one type.
///
/// # Panic
/// A safety check fails when a viewport extent is
/// zero.
pub fn unprojectScreen(
    inverse_view_projection: anytype,
    viewport: anytype,
    screen: anytype,
    depth: anytype,
) @Vector(3, @TypeOf(depth)) {
    const ndc = toNdc(viewport, screen, depth);
    return unproject(inverse_view_projection, ndc);
}

test "look at" {
    const eye: Vec3f = .{ 0, 0, 5 };
    const target: Vec3f = .{ 0, 0, 0 };
    const up: Vec3f = .{ 0, 1, 0 };
    const view = lookAt(eye, target, up, 1e-6).?;

    const origin = mat.mulVec(view, @Vector(4, f32){ 0, 0, 0, 1 });
    try std.testing.expectApproxEqAbs(@as(f32, 0), origin[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), origin[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -5), origin[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), origin[3], 1e-6);

    const camera_position = mat.mulVec(
        view,
        @Vector(4, f32){ eye[0], eye[1], eye[2], 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        camera_position[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        camera_position[2],
        1e-6,
    );

    try std.testing.expect(
        lookAt(eye, eye, up, 1e-6) == null,
    );
    try std.testing.expect(
        lookAt(eye, target, Vec3f{ 0, 0, -1 }, 1e-6) == null,
    );
}

test "perspective depth range" {
    const projection = perspective(
        @as(f32, std.math.pi) / 2,
        1,
        0.1,
        100,
    );
    const near_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -0.1, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_point[2] / near_point[3],
        1e-5,
    );
    const far_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -100, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        far_point[2] / far_point[3],
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 100),
        far_point[3],
        1e-5,
    );
}

test "reversed perspective depth range" {
    const projection = perspectiveReversed(
        @as(f32, std.math.pi) / 2,
        1,
        0.1,
        100,
    );
    const near_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -0.1, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        near_point[2] / near_point[3],
        1e-5,
    );
    const far_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -100, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        far_point[2] / far_point[3],
        1e-4,
    );
}

test "infinite perspective" {
    const projection = perspectiveInfinite(
        @as(f32, std.math.pi) / 2,
        1,
        0.1,
    );
    const near_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -0.1, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_point[2] / near_point[3],
        1e-5,
    );
    const far_point = mat.mulVec(
        projection,
        @Vector(4, f32){ 0, 0, -10000, 1 },
    );
    try std.testing.expect(
        far_point[2] / far_point[3] > 0.99,
    );

    const reversed = perspectiveInfiniteReversed(
        @as(f32, std.math.pi) / 2,
        1,
        0.1,
    );
    const near_reversed = mat.mulVec(
        reversed,
        @Vector(4, f32){ 0, 0, -0.1, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        near_reversed[2] / near_reversed[3],
        1e-5,
    );
}

test "orthographic corners" {
    const projection = orthographic(
        @as(f32, -2),
        2,
        -1,
        1,
        0.5,
        10,
    );
    const near_corner = mat.mulVec(
        projection,
        @Vector(4, f32){ -2, -1, -0.5, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1),
        near_corner[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1),
        near_corner[1],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        near_corner[2],
        1e-6,
    );
    const far_corner = mat.mulVec(
        projection,
        @Vector(4, f32){ 2, 1, -10, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        far_corner[2],
        1e-6,
    );
}

test "unproject round trip" {
    const view = lookAt(
        Vec3f{ 1, 2, 3 },
        Vec3f{ 0, 0, 0 },
        Vec3f{ 0, 1, 0 },
        1e-6,
    ).?;
    const projection = perspective(
        @as(f32, std.math.pi) / 3,
        1.5,
        0.1,
        100,
    );
    const view_projection = mat.multiply(view, projection);
    const inverse_vp = mat.inverse(view_projection, 1e-9).?;

    const viewport = Viewport(f32){ .width = 800, .height = 600 };
    const screen: @Vector(2, f32) = .{ 400, 300 };
    const depth: f32 = 0.4;
    const world = unprojectScreen(inverse_vp, viewport, screen, depth);
    const clip = mat.mulVec(
        view_projection,
        @Vector(4, f32){ world[0], world[1], world[2], 1 },
    );
    try std.testing.expectApproxEqAbs(0, clip[0] / clip[3], 1e-3);
    try std.testing.expectApproxEqAbs(0, clip[1] / clip[3], 1e-3);
    try std.testing.expectApproxEqAbs(depth, clip[2] / clip[3], 1e-3);

    const ndc = toNdc(viewport, screen, depth);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ndc[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), ndc[1], 1e-6);
}
