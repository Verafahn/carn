//! SPIR-V compilation check for the math module.
//!
//! The build compiles this file for the
//! `spirv32-vulkan` target; host code never imports
//! it. The entry point calls one operation per
//! module so the shader backend analyses the shared
//! code paths.

const std = @import("std");
const spirv = std.spirv;
const math = @import("root.zig");

export fn main() callconv(.spirv_vertex) void {
    const position: math.Vec3(f32) = .{ 1, 2, 3 };
    const unit = math.vec.normalize(position);
    const rotation = math.quat.fromAxisAngle(unit, 0.5);
    const matrix = math.quat.toMat3(rotation);
    const rotated = math.mat.mulVec(matrix, position);
    const plane = math.bounds.fromPointNormal(
        rotated,
        math.vec.normalize(math.vec.cross(position, rotated)),
    );
    const height = math.bounds.signedDistance(plane, rotated);
    const reflected = math.vec.reflect(position, unit);
    const blended = math.interp.lerp(
        reflected,
        math.vec.splat(math.Vec3(f32), height),
        math.easing.easeInOutQuad(@as(f32, 0.25)),
    );
    const sample = math.sampling.uniformSphere(@Vector(2, f32){
        math.sampling.halton(3, 2),
        math.sampling.sobol2(3)[1],
    });
    const ray = math.geometry.Ray(f32){
        .origin = position,
        .direction = sample,
    };
    const hit = math.geometry.raySphere(
        ray,
        math.bounds.Sphere(f32){ .center = rotated, .radius = 1 },
        1e-6,
    );
    const distance = if (hit) |value| value.far else 0;
    const color = math.color.hsvToRgb(@Vector(3, f32){
        math.color.rgbToHsv(math.Rgb(f32){
            .r = 1,
            .g = 0.5,
            .b = 0.25,
        })[0],
        0.5,
        1,
    });
    const shaded = math.color.linearToSrgb(@Vector(3, f32){
        color.r,
        color.g,
        color.b,
    });
    const angle = math.euler.degreesToRadians(@as(f32, 90));
    const local = math.transform.transformPoint(
        math.camera.perspective(
            angle,
            1.5,
            0.1,
            100,
        ),
        blended,
    );
    const target = math.basis.orthonormalBasis(
        math.vec.normalize(local),
        1e-6,
    );
    const spread = if (target) |frame| frame[0][0] else 0;

    spirv.position_out.* = .{
        shaded[0] + spread + distance + angle,
        shaded[1],
        shaded[2],
        1,
    };
}
