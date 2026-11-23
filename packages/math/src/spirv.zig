//! SPIR-V compilation check for the math module.
//!
//! The build compiles this file for the
//! `spirv32-vulkan` target; host code never imports
//! it. The entry point calls one operation per group
//! so the shader backend analyses the shared paths.

const std = @import("std");
const spirv = std.spirv;
const math = @import("root.zig");

export fn main() callconv(.spirv_vertex) void {
    const Vec2f = math.Vec2(f32);
    const Vec3f = math.Vec3(f32);

    const position = Vec3f.fromElements(.{ 1, 2, 3 });
    const unit = position.normalize();
    const rotation = math.Quat(f32).fromAxisAngle(unit, 0.5);
    const matrix = rotation.toMat3();
    const rotated = position.mulMatrix(3, matrix);
    const normal = position.cross(rotated).normalize();
    const plane = math.Plane(f32).fromPointNormal(rotated, normal);
    const height = plane.signedDistance(rotated);
    const reflected = position.reflect(unit);
    const blended = math.curve.interp.lerp(
        Vec3f,
        reflected,
        Vec3f.splat(height),
        math.curve.easing.easeInOutQuad(f32, 0.25),
    );

    const sample = math.Sampling(f32).uniformSphere(Vec2f
        .fromElements(.{
        math.Sampling(f32).halton(3, 2),
        math.Sampling(f32).sobol2(3).at(1),
    }));
    const ray = math.Ray(f32).fromPoints(position, sample);
    const hit = ray.intersectSphere(
        math.Sphere(f32){ .center = rotated, .radius = 1 },
        1e-6,
    );
    const distance = if (hit) |value| value.far else 0;

    const color = math.Rgb(f32)
        .fromHsv(Vec3f.fromElements(.{ 0.5, 0.5, 1 }))
        .linearToSrgb();
    const shaded = color.toVector();
    const angle = math.Euler(f32).degreesToRadians(90);
    const local = math.Camera(f32)
        .perspective(angle, 1.5, 0.1, 100)
        .transformPoint(blended);
    const frame = math.Basis(f32)
        .fromNormal(local.normalize(), 1e-6);
    const spread = if (frame) |value| value.tangent.at(0) else 0;

    spirv.position_out.* = .{
        shaded.at(0) + spread + distance + angle,
        shaded.at(1),
        shaded.at(2),
        1,
    };
}
