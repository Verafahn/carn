//! A generic math library for host code and shaders.
//!
//! ```zig
//! const math = @import("math");
//! const m = math.Quat(f32).fromAxisAngle(axis, angle);
//! const rotation = m.toMat3();
//! const p = point.mulMatrix(3, rotation);
//! ```
//!
//! Conventions:
//! - A vector is the `1xN` matrix case: `Vector(T, N)`
//!   is `Matrix(T, 1, N)`.
//! - Matrices are column-major; a row vector
//!   multiplies with `mulMatrix`.
//! - Every tolerance is passed by the caller as
//!   `epsilon`. `Epsilon` only suggests constants.
//! - Operations are methods, so operand types are
//!   fixed at the call site.
//!
//! The library keeps no global state, allocates no
//! memory, and uses no implicit tolerances, so the same
//! code compiles for the host and for the SPIR-V
//! backend. Types crossing the shader boundary use only
//! vectors and scalars.
//!
//! Inverse trigonometric helpers such as
//! `Euler.fromQuat` and `Quat.slerp` rely on `std`
//! routines that may not lower on SPIR-V.

pub const base = @import("base/root.zig");
pub const linear = @import("linear/root.zig");
pub const spatial = @import("spatial/root.zig");
pub const curve = @import("curve/root.zig");
pub const visual = @import("visual/root.zig");

pub const Epsilon = base.epsilon.Epsilon;
pub const Fixed = base.fixed.Fixed;

pub const Vector = linear.vector.Vector;
pub const Vec2 = linear.vector.Vec2;
pub const Vec3 = linear.vector.Vec3;
pub const Vec4 = linear.vector.Vec4;
pub const Half2 = linear.vector.Half2;
pub const Half3 = linear.vector.Half3;
pub const Half4 = linear.vector.Half4;
pub const IVec2 = linear.vector.IVec2;
pub const IVec3 = linear.vector.IVec3;
pub const IVec4 = linear.vector.IVec4;
pub const UVec2 = linear.vector.UVec2;
pub const UVec3 = linear.vector.UVec3;
pub const UVec4 = linear.vector.UVec4;

pub const Matrix = linear.matrix.Matrix;
pub const Mat2 = linear.matrix.Mat2;
pub const Mat3 = linear.matrix.Mat3;
pub const Mat4 = linear.matrix.Mat4;
pub const Mat3x4 = linear.matrix.Mat3x4;
pub const Quat = linear.quaternion.Quat;
pub const Euler = linear.euler.Euler;
pub const Order = linear.euler.Order;
pub const Basis = linear.basis.Basis;
pub const Decomposition = linear.decompose.Decomposition;

pub const Plane = spatial.plane.Plane;
pub const Classification = spatial.plane.Classification;
pub const Sphere = spatial.sphere.Sphere;
pub const Aabb = spatial.aabb.Aabb;
pub const Obb = spatial.obb.Obb;
pub const Frustum = spatial.frustum.Frustum;
pub const Capsule = spatial.capsule.Capsule;
pub const LineSegment = spatial.segment.LineSegment;
pub const Triangle = spatial.triangle.Triangle;
pub const TriangleHit = spatial.triangle.TriangleHit;
pub const Ray = spatial.ray.Ray;
pub const SphereHit = spatial.ray.SphereHit;
pub const Camera = spatial.camera.Camera;
pub const Viewport = spatial.camera.Viewport;
pub const Transform = spatial.transform.Transform;

pub const Rgb = visual.color.Rgb;
pub const Rgba = visual.color.Rgba;
pub const Sampling = visual.sampling.Sampling;

test {
    _ = base;
    _ = linear;
    _ = spatial;
    _ = curve;
    _ = visual;
}
