//! A generic math library for host code and shaders.
//!
//! ```zig
//! const math = @import("math");
//! const m = math.quat.toMat3(rotation);
//! const p = math.mat.mulVec(m, point);
//! ```
//!
//! Conventions:
//! - Vectors are row matrices stored as `@Vector`.
//! - Matrices are column-major; `mulVec` applies a
//!   row vector to a matrix.
//! - Every tolerance is passed by the caller as
//!   `epsilon`. `Epsilon` only suggests constants.
//!
//! The library keeps no global state, allocates no
//! memory, and avoids floating point tolerances of
//! its own, so the same code compiles for the host
//! and for the SPIR-V backend. Types that cross the
//! shader boundary use only vectors and scalars.
//!
//! Inverse trigonometric helpers such as
//! `euler.fromQuat` and `quat.slerp` rely on `std`
//! routines that may not lower on SPIR-V.

pub const scalar = @import("scalar.zig");
pub const vec = @import("vec.zig");
pub const mat = @import("mat.zig");
pub const quat = @import("quat.zig");
pub const euler = @import("euler.zig");
pub const basis = @import("basis.zig");
pub const camera = @import("camera.zig");
pub const bounds = @import("bounds.zig");
pub const geometry = @import("geometry.zig");
pub const interp = @import("interp.zig");
pub const easing = @import("easing.zig");
pub const color = @import("color.zig");
pub const sampling = @import("sampling.zig");
pub const transform = @import("transform.zig");

pub const Epsilon = scalar.Epsilon;
pub const Fixed = scalar.Fixed;

pub const Vector = vec.Vector;
pub const Vec2 = vec.Vec2;
pub const Vec3 = vec.Vec3;
pub const Vec4 = vec.Vec4;
pub const Half2 = vec.Half2;
pub const Half3 = vec.Half3;
pub const Half4 = vec.Half4;
pub const IVec2 = vec.IVec2;
pub const IVec3 = vec.IVec3;
pub const IVec4 = vec.IVec4;
pub const UVec2 = vec.UVec2;
pub const UVec3 = vec.UVec3;
pub const UVec4 = vec.UVec4;

pub const Matrix = mat.Matrix;
pub const Mat2 = mat.Mat2;
pub const Mat3 = mat.Mat3;
pub const Mat4 = mat.Mat4;
pub const Mat3x4 = mat.Mat3x4;
pub const Decomposition = mat.Decomposition;

pub const Quat = quat.Quat;
pub const Euler = euler.Euler;
pub const Order = euler.Order;

pub const Viewport = camera.Viewport;

pub const Plane = bounds.Plane;
pub const Classification = bounds.Classification;
pub const Sphere = bounds.Sphere;
pub const Aabb = bounds.Aabb;
pub const Obb = bounds.Obb;
pub const Frustum = bounds.Frustum;
pub const Capsule = bounds.Capsule;

pub const Ray = geometry.Ray;
pub const LineSegment = geometry.LineSegment;
pub const Triangle = geometry.Triangle;
pub const TriangleHit = geometry.TriangleHit;
pub const SphereHit = geometry.SphereHit;

pub const Rgb = color.Rgb;
pub const Rgba = color.Rgba;

test {
    _ = scalar;
    _ = vec;
    _ = mat;
    _ = quat;
    _ = euler;
    _ = basis;
    _ = camera;
    _ = bounds;
    _ = geometry;
    _ = interp;
    _ = easing;
    _ = color;
    _ = sampling;
    _ = transform;
}
