//! Planes, bounding volumes, and frusta.
//!
//! Every type stores plain vectors and scalars, so
//! no allocation or global state is involved.
//! Tolerance-based tests take `epsilon` from the
//! caller.
//!
//! The functions are generic: each argument must
//! expose the documented fields with matching
//! component types, or the call fails to compile.

const std = @import("std");
const scalar = @import("scalar.zig");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const Vec3f = vec.Vec3(f32);

/// A plane in Hessian normal form.
///
/// # Args
/// - `T`: the component type.
pub fn Plane(comptime T: type) type {
    return struct {
        /// Unit normal of the plane.
        normal: @Vector(3, T),
        /// Signed distance from the origin.
        distance: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Where a point lies relative to a plane.
pub const Classification = enum {
    /// Negative distance; behind the plane.
    behind,
    /// Distance within the caller tolerance.
    on,
    /// Positive distance; in front of the plane.
    front,
};

/// Returns the signed distance from a point.
///
/// # Args
/// - `plane`: the plane to test.
/// - `point`: the point to measure.
///
/// # Return
/// `dot(normal, point) + distance`; positive values
/// are in front of the plane.
///
/// # Note
/// The distance is metric only when the plane
/// normal is a unit vector.
pub fn signedDistance(plane: anytype, point: anytype) @TypeOf(
    plane.distance,
) {
    return vec.dot(plane.normal, point) + plane.distance;
}

/// Classifies a point against a plane.
///
/// # Args
/// - `plane`: the plane to test.
/// - `point`: the point to classify.
/// - `epsilon`: distance within which a point is on
///   the plane.
///
/// # Return
/// `behind`, `on`, or `front`.
pub fn classify(
    plane: anytype,
    point: anytype,
    epsilon: anytype,
) Classification {
    const distance = signedDistance(plane, point);
    if (distance > epsilon) return .front;
    if (distance < -epsilon) return .behind;
    return .on;
}

/// Builds a plane from a point and a normal.
///
/// # Args
/// - `point`: a point on the plane.
/// - `normal`: unit normal of the plane.
///
/// # Return
/// The plane through `point`.
///
/// # Note
/// The normal is used as given; pass a unit vector
/// so `signedDistance` stays metric.
pub fn fromPointNormal(point: anytype, normal: anytype) Plane(
    scalar.ScalarOf(@TypeOf(point)),
) {
    return .{
        .normal = normal,
        .distance = -vec.dot(normal, point),
    };
}

/// Normalizes a plane.
///
/// # Args
/// - `plane`: the plane to normalize.
///
/// # Return
/// A plane with a unit normal.
///
/// # Note
/// A zero normal produces non-finite values; the
/// caller owns that precondition.
pub fn normalizePlane(plane: anytype) @TypeOf(plane) {
    const factor = 1 / vec.length(plane.normal);
    return .{
        .normal = plane.normal * @as(
            @TypeOf(plane.normal),
            @splat(factor),
        ),
        .distance = plane.distance * factor,
    };
}

/// Flips a plane to face the other way.
///
/// # Args
/// - `plane`: the plane to flip.
///
/// # Return
/// The plane with all signs negated.
pub fn flip(plane: anytype) @TypeOf(plane) {
    return .{
        .normal = -plane.normal,
        .distance = -plane.distance,
    };
}

/// Projects a point onto a plane.
///
/// # Args
/// - `plane`: the target plane.
/// - `point`: the point to project.
///
/// # Return
/// The closest point on the plane.
///
/// # Note
/// The plane normal must be a unit vector.
pub fn projectPoint(plane: anytype, point: anytype) @TypeOf(point) {
    const distance = signedDistance(plane, point);
    return point - plane.normal * @as(
        @TypeOf(point),
        @splat(distance),
    );
}

/// A sphere with a center and a radius.
///
/// # Args
/// - `T`: the component type.
pub fn Sphere(comptime T: type) type {
    return struct {
        /// Center of the sphere.
        center: @Vector(3, T),
        /// Radius of the sphere.
        radius: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Tests whether a point is inside a sphere.
///
/// # Args
/// - `sphere`: the volume to test.
/// - `point`: the point to test.
///
/// # Return
/// `true` when the point is inside or on the
/// surface.
pub fn sphereContainsPoint(sphere: anytype, point: anytype) bool {
    const offset = point - sphere.center;
    return vec.lengthSquared(offset) <=
        sphere.radius * sphere.radius;
}

/// Tests whether two spheres overlap.
///
/// # Args
/// - `a`: first sphere.
/// - `b`: second sphere.
///
/// # Return
/// `true` when the surfaces touch or overlap.
pub fn spheresIntersect(a: anytype, b: anytype) bool {
    const offset = a.center - b.center;
    const radius = a.radius + b.radius;
    return vec.lengthSquared(offset) <= radius * radius;
}

/// Computes the smallest sphere enclosing two spheres.
///
/// # Args
/// - `a`: first sphere.
/// - `b`: second sphere.
///
/// # Return
/// The merged sphere.
///
/// # Note
/// When one sphere contains the other, that sphere
/// is returned unchanged.
pub fn mergeSpheres(a: anytype, b: anytype) @TypeOf(a) {
    const offset = b.center - a.center;
    const distance = vec.length(offset);
    if (a.radius >= distance + b.radius) return a;
    if (b.radius >= distance + a.radius) return b;

    const radius = (distance + a.radius + b.radius) / 2;
    const center_point = a.center +
        offset * @as(@TypeOf(offset), @splat(
            (radius - a.radius) / distance,
        ));
    return .{ .center = center_point, .radius = radius };
}

/// An axis-aligned bounding box.
///
/// # Args
/// - `T`: the component type.
pub fn Aabb(comptime T: type) type {
    return struct {
        /// Corner with the smallest coordinates.
        min: @Vector(3, T),
        /// Corner with the largest coordinates.
        max: @Vector(3, T),

        /// The component type.
        pub const Component = T;
    };
}

/// Computes the union of two boxes.
///
/// # Args
/// - `a`: first box.
/// - `b`: second box.
///
/// # Return
/// The smallest box containing both.
pub fn mergeAabb(a: anytype, b: anytype) @TypeOf(a) {
    return .{
        .min = @min(a.min, b.min),
        .max = @max(a.max, b.max),
    };
}

/// Tests whether a point is inside a box.
///
/// # Args
/// - `box`: the volume to test.
/// - `point`: the point to test.
///
/// # Return
/// `true` when the point is inside or on the
/// surface.
pub fn aabbContainsPoint(box: anytype, point: anytype) bool {
    return @reduce(.And, point >= box.min) and
        @reduce(.And, point <= box.max);
}

/// Tests whether two boxes overlap.
///
/// # Args
/// - `a`: first box.
/// - `b`: second box.
///
/// # Return
/// `true` when the boxes touch or overlap.
pub fn aabbsIntersect(a: anytype, b: anytype) bool {
    return @reduce(.And, a.min <= b.max) and
        @reduce(.And, a.max >= b.min);
}

/// Returns the center of a box.
///
/// # Args
/// - `box`: the box to measure.
///
/// # Return
/// The midpoint of the two corners.
pub fn aabbCenter(box: anytype) @TypeOf(box.min) {
    return (box.min + box.max) / @as(
        @TypeOf(box.min),
        @splat(2),
    );
}

/// Returns the size of a box.
///
/// # Args
/// - `box`: the box to measure.
///
/// # Return
/// The extent of the box along each axis.
pub fn aabbSize(box: anytype) @TypeOf(box.min) {
    return box.max - box.min;
}

/// Computes the bounding sphere of a box.
///
/// # Args
/// - `box`: the box to enclose.
///
/// # Return
/// A sphere around the box; it is not the smallest
/// possible sphere.
pub fn aabbToSphere(box: anytype) Sphere(
    scalar.ScalarOf(@TypeOf(box.min)),
) {
    const half = aabbSize(box) / @as(
        @TypeOf(box.min),
        @splat(2),
    );
    return .{
        .center = aabbCenter(box),
        .radius = vec.length(half),
    };
}

/// Builds a box from a center and extents.
///
/// # Args
/// - `center`: midpoint of the box.
/// - `half_extents`: half the size along each axis.
///
/// # Return
/// The box spanning the extents.
pub fn aabbFromCenterHalfExtents(
    center: anytype,
    half_extents: anytype,
) Aabb(scalar.ScalarOf(@TypeOf(center))) {
    const V = @TypeOf(center);
    const extents: V = half_extents;
    return .{
        .min = center - extents,
        .max = center + extents,
    };
}

/// Builds a box from a point slice.
///
/// # Args
/// - `points`: non-empty list of points.
///
/// # Return
/// The smallest box containing the points, or
/// `null` when the slice is empty.
pub fn aabbFromPoints(points: anytype) ?Aabb(
    scalar.ScalarOf(elementType(@TypeOf(points))),
) {
    if (points.len == 0) return null;
    var box_min = points[0];
    var box_max = points[0];
    for (points[1..]) |point| {
        box_min = @min(box_min, point);
        box_max = @max(box_max, point);
    }
    return .{ .min = box_min, .max = box_max };
}

/// Returns the closest point inside a box.
///
/// # Args
/// - `box`: the box to clamp to.
/// - `point`: the point to clamp.
///
/// # Return
/// `point` when inside, otherwise the nearest
/// surface point.
pub fn aabbClosestPoint(box: anytype, point: anytype) @TypeOf(point) {
    return @min(@max(point, box.min), box.max);
}

/// An oriented bounding box.
///
/// # Args
/// - `T`: the component type.
pub fn Obb(comptime T: type) type {
    return struct {
        /// Center of the box.
        center: @Vector(3, T),
        /// Half size along each local axis.
        half_extents: @Vector(3, T),
        /// Local-to-world rotation.
        rotation: mat.Mat3(T),

        /// The component type.
        pub const Component = T;
    };
}

/// Builds an oriented box from an axis-aligned one.
///
/// # Args
/// - `box`: the source box.
///
/// # Return
/// An oriented box with no rotation.
pub fn obbFromAabb(box: anytype) Obb(
    scalar.ScalarOf(@TypeOf(box.min)),
) {
    const T = scalar.ScalarOf(@TypeOf(box.min));
    return .{
        .center = aabbCenter(box),
        .half_extents = aabbSize(box) / @as(
            @TypeOf(box.min),
            @splat(2),
        ),
        .rotation = mat.Mat3(T).identity(),
    };
}

/// Tests whether a point is inside an oriented box.
///
/// # Args
/// - `box`: the volume to test.
/// - `point`: the point to test.
///
/// # Return
/// `true` when the point is inside or on the
/// surface.
pub fn obbContainsPoint(box: anytype, point: anytype) bool {
    const offset = point - box.center;
    const local = mat.mulVec(
        mat.transpose(box.rotation),
        offset,
    );
    const extent = box.half_extents;
    return @reduce(.And, @abs(local) <= extent);
}

/// Returns one corner of an oriented box.
///
/// # Args
/// - `box`: the box to read.
/// - `index`: corner index in `0..8`.
///
/// # Return
/// The world-space corner; bit 0 selects the X
/// sign, bit 1 the Y sign, and bit 2 the Z sign.
///
/// # Panic
/// `index` is at least 8.
pub fn obbCorner(box: anytype, index: usize) @TypeOf(box.center) {
    std.debug.assert(index < 8);
    const T = scalar.ScalarOf(@TypeOf(box.center));
    const x: T = if (index & 1 == 0) -1 else 1;
    const y: T = if (index & 2 == 0) -1 else 1;
    const z: T = if (index & 4 == 0) -1 else 1;
    const local = @Vector(3, T){
        x * box.half_extents[0],
        y * box.half_extents[1],
        z * box.half_extents[2],
    };
    return box.center + mat.mulVec(box.rotation, local);
}

/// Computes a conservative axis-aligned box.
///
/// # Args
/// - `box`: the oriented box to enclose.
///
/// # Return
/// An axis-aligned box containing the rotated box.
pub fn obbToAabb(box: anytype) Aabb(
    scalar.ScalarOf(@TypeOf(box.center)),
) {
    const T = scalar.ScalarOf(@TypeOf(box.center));
    var extents: @Vector(3, T) = @splat(0);
    inline for (0..3) |axis| {
        var sum: T = 0;
        inline for (0..3) |component| {
            sum += box.half_extents[component] *
                @abs(box.rotation.columns[axis][component]);
        }
        extents[axis] = sum;
    }
    const center_point = box.center;
    return .{
        .min = center_point - extents,
        .max = center_point + extents,
    };
}

/// A viewing frustum made of six planes.
///
/// # Args
/// - `T`: the component type.
pub fn Frustum(comptime T: type) type {
    return struct {
        /// Left clipping plane.
        left: Plane(T),
        /// Right clipping plane.
        right: Plane(T),
        /// Bottom clipping plane.
        bottom: Plane(T),
        /// Top clipping plane.
        top: Plane(T),
        /// Near clipping plane.
        near: Plane(T),
        /// Far clipping plane.
        far: Plane(T),

        /// The component type.
        pub const Component = T;
    };
}

/// Extracts a frustum from a view-projection matrix.
///
/// # Args
/// - `view_projection`: a row-vector matrix that
///   maps view space to clip space.
///
/// # Return
/// The six normalized clipping planes.
///
/// # Note
/// The extraction assumes clip depth in `0..1`. With
/// reversed depth the `near` and `far` fields hold
/// the opposite planes, which does not affect the
/// intersection tests.
///
/// # Compile Error
/// - `view_projection` is not a `4x4` matrix.
pub fn frustumFromMatrix(view_projection: anytype) Frustum(
    scalar.ScalarOf(@TypeOf(view_projection)),
) {
    const M = @TypeOf(view_projection);
    if (M.row_count != 4 or M.column_count != 4) {
        @compileError(
            "frustumFromMatrix: matrix must be 4x4",
        );
    }
    const T = scalar.ScalarOf(M);
    const PlaneT = Plane(T);
    const c0 = view_projection.columns[0];
    const c1 = view_projection.columns[1];
    const c2 = view_projection.columns[2];
    const c3 = view_projection.columns[3];
    const left = PlaneT{
        .normal = .{
            c3[0] + c0[0],
            c3[1] + c0[1],
            c3[2] + c0[2],
        },
        .distance = c3[3] + c0[3],
    };
    const right = PlaneT{
        .normal = .{
            c3[0] - c0[0],
            c3[1] - c0[1],
            c3[2] - c0[2],
        },
        .distance = c3[3] - c0[3],
    };
    const bottom = PlaneT{
        .normal = .{
            c3[0] + c1[0],
            c3[1] + c1[1],
            c3[2] + c1[2],
        },
        .distance = c3[3] + c1[3],
    };
    const top = PlaneT{
        .normal = .{
            c3[0] - c1[0],
            c3[1] - c1[1],
            c3[2] - c1[2],
        },
        .distance = c3[3] - c1[3],
    };
    const near = PlaneT{
        .normal = .{ c2[0], c2[1], c2[2] },
        .distance = c2[3],
    };
    const far = PlaneT{
        .normal = .{
            c3[0] - c2[0],
            c3[1] - c2[1],
            c3[2] - c2[2],
        },
        .distance = c3[3] - c2[3],
    };
    return .{
        .left = normalizePlane(left),
        .right = normalizePlane(right),
        .bottom = normalizePlane(bottom),
        .top = normalizePlane(top),
        .near = normalizePlane(near),
        .far = normalizePlane(far),
    };
}

/// Tests whether a point is inside a frustum.
///
/// # Args
/// - `frustum`: the volume to test.
/// - `point`: the point to test.
///
/// # Return
/// `true` when the point is inside all planes.
pub fn frustumContainsPoint(frustum: anytype, point: anytype) bool {
    inline for (planeList(frustum)) |plane| {
        if (signedDistance(plane, point) < 0) return false;
    }
    return true;
}

/// Tests whether a sphere overlaps a frustum.
///
/// # Args
/// - `frustum`: the volume to test.
/// - `sphere`: the sphere to test.
///
/// # Return
/// `false` only when the sphere is fully outside
/// one plane.
///
/// # Note
/// A sphere outside the frustum corners may still
/// report `true`; the test is conservative.
pub fn frustumIntersectsSphere(frustum: anytype, sphere: anytype) bool {
    inline for (planeList(frustum)) |plane| {
        if (signedDistance(plane, sphere.center) < -sphere.radius) {
            return false;
        }
    }
    return true;
}

/// Tests whether a box overlaps a frustum.
///
/// # Args
/// - `frustum`: the volume to test.
/// - `box`: the axis-aligned box to test.
///
/// # Return
/// `false` only when the box is fully outside one
/// plane.
///
/// # Note
/// The test only rejects boxes behind a plane, so
/// corners may be rejected conservatively.
pub fn frustumIntersectsAabb(frustum: anytype, box: anytype) bool {
    const T = scalar.ScalarOf(@TypeOf(box.min));
    inline for (planeList(frustum)) |plane| {
        const positive = @select(
            T,
            plane.normal >= @as(@TypeOf(plane.normal), @splat(0)),
            box.max,
            box.min,
        );
        if (signedDistance(plane, positive) < 0) return false;
    }
    return true;
}

/// Tests whether an oriented box overlaps a frustum.
///
/// # Args
/// - `frustum`: the volume to test.
/// - `box`: the oriented box to test.
///
/// # Return
/// `false` only when the box is fully outside one
/// plane.
pub fn frustumIntersectsObb(frustum: anytype, box: anytype) bool {
    const transposed = mat.transpose(box.rotation);
    inline for (planeList(frustum)) |plane| {
        const local = mat.mulVec(transposed, plane.normal);
        const radius = vec.dot(box.half_extents, @abs(local));
        if (signedDistance(plane, box.center) < -radius) {
            return false;
        }
    }
    return true;
}

/// A capsule around a line segment.
///
/// # Args
/// - `T`: the component type.
pub fn Capsule(comptime T: type) type {
    return struct {
        /// First endpoint of the core segment.
        a: @Vector(3, T),
        /// Second endpoint of the core segment.
        b: @Vector(3, T),
        /// Distance from the segment to the surface.
        radius: T,

        /// The component type.
        pub const Component = T;
    };
}

/// Builds a capsule from a segment and a radius.
///
/// # Args
/// - `a`: first segment endpoint.
/// - `b`: second segment endpoint.
/// - `radius`: distance from the segment to the
///   surface.
///
/// # Return
/// The capsule around the segment.
pub fn capsuleFromSegment(
    a: anytype,
    b: anytype,
    radius: anytype,
) Capsule(scalar.ScalarOf(@TypeOf(a))) {
    const V = @TypeOf(a);
    const end: V = b;
    return .{ .a = a, .b = end, .radius = radius };
}

/// Tests whether a point is inside a capsule.
///
/// # Args
/// - `capsule`: the volume to test.
/// - `point`: the point to test.
///
/// # Return
/// `true` when the point is within the radius of
/// the core segment.
pub fn capsuleContainsPoint(capsule: anytype, point: anytype) bool {
    const distance = segmentDistance(capsule.a, capsule.b, point);
    return distance <= capsule.radius;
}

/// Tests whether a sphere overlaps a capsule.
///
/// # Args
/// - `capsule`: the volume to test.
/// - `sphere`: the sphere to test.
///
/// # Return
/// `true` when the surfaces touch or overlap.
pub fn capsuleIntersectsSphere(capsule: anytype, sphere: anytype) bool {
    const distance = segmentDistance(capsule.a, capsule.b, sphere.center);
    return distance <= capsule.radius + sphere.radius;
}

fn segmentDistance(a: anytype, b: anytype, point: anytype) @TypeOf(
    point[0],
) {
    const V = @TypeOf(point);
    const segment = b - a;
    const length_squared = vec.lengthSquared(segment);
    if (length_squared == 0) return vec.length(point - a);
    const offset = point - a;
    const factor = @min(@max(
        vec.dot(offset, segment) / length_squared,
        0,
    ), 1);
    const closest = a + segment * @as(V, @splat(factor));
    return vec.length(point - closest);
}

fn planeList(frustum: anytype) [6]@TypeOf(frustum.left) {
    return .{
        frustum.left,
        frustum.right,
        frustum.bottom,
        frustum.top,
        frustum.near,
        frustum.far,
    };
}

fn elementType(comptime S: type) type {
    return switch (@typeInfo(S)) {
        .pointer => |info| switch (@typeInfo(info.child)) {
            .array => |array| array.child,
            else => info.child,
        },
        .array => |info| info.child,
        else => @compileError(
            "expected a slice or array, got `" ++
                @typeName(S) ++ "`",
        ),
    };
}

test "plane" {
    const plane = fromPointNormal(
        @as(Vec3f, .{ 0, 0, 0 }),
        Vec3f{ 0, 0, 1 },
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        signedDistance(plane, Vec3f{ 0, 0, 2 }),
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        signedDistance(plane, Vec3f{ 0, 0, -2 }),
        1e-6,
    );
    try std.testing.expectEqual(
        Classification.front,
        classify(plane, Vec3f{ 0, 0, 1 }, 1e-6),
    );
    try std.testing.expectEqual(
        Classification.behind,
        classify(plane, Vec3f{ 0, 0, -1 }, 1e-6),
    );
    try std.testing.expectEqual(
        Classification.on,
        classify(plane, Vec3f{ 1, 1, 0 }, 1e-6),
    );
    const flipped = flip(plane);
    try std.testing.expectApproxEqAbs(
        @as(f32, -2),
        signedDistance(flipped, Vec3f{ 0, 0, 2 }),
        1e-6,
    );
    const projected = projectPoint(plane, Vec3f{ 1, 2, 3 });
    try std.testing.expectApproxEqAbs(@as(f32, 1), projected[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), projected[2], 1e-6);

    const scaled = Plane(f32){
        .normal = .{ 0, 0, 2 },
        .distance = -4,
    };
    const normalized = normalizePlane(scaled);
    try std.testing.expectApproxEqAbs(@as(f32, 1), normalized.normal[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2), normalized.distance, 1e-6);
}

test "sphere" {
    const unit = Sphere(f32){
        .center = .{ 0, 0, 0 },
        .radius = 1,
    };
    try std.testing.expect(sphereContainsPoint(unit, Vec3f{ 0, 0, 1 }));
    try std.testing.expect(!sphereContainsPoint(unit, Vec3f{ 0, 0, 2 }));

    const other = Sphere(f32){
        .center = .{ 2, 0, 0 },
        .radius = 1,
    };
    try std.testing.expect(spheresIntersect(unit, other));
    const far = Sphere(f32){
        .center = .{ 5, 0, 0 },
        .radius = 1,
    };
    try std.testing.expect(!spheresIntersect(unit, far));

    const merged = mergeSpheres(unit, far);
    try std.testing.expectApproxEqAbs(
        @as(f32, 2.5),
        merged.center[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3.5),
        merged.radius,
        1e-6,
    );
    try std.testing.expect(
        sphereContainsPoint(merged, Vec3f{ 0, 0, 1 }),
    );
    try std.testing.expect(
        sphereContainsPoint(merged, Vec3f{ 5, 0, 1 }),
    );

    const contained = mergeSpheres(unit, Sphere(f32){
        .center = .{ 0, 0, 0 },
        .radius = 0.5,
    });
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        contained.radius,
        1e-6,
    );
}

test "aabb" {
    const a = Aabb(f32){
        .min = .{ -1, -1, -1 },
        .max = .{ 1, 1, 1 },
    };
    const b = Aabb(f32){
        .min = .{ 0, 0, 0 },
        .max = .{ 2, 2, 2 },
    };
    try std.testing.expect(aabbContainsPoint(a, Vec3f{ 0, 0, 0 }));
    try std.testing.expect(!aabbContainsPoint(a, Vec3f{ 2, 0, 0 }));
    try std.testing.expect(aabbsIntersect(a, b));
    try std.testing.expect(!aabbsIntersect(a, Aabb(f32){
        .min = .{ 3, 3, 3 },
        .max = .{ 4, 4, 4 },
    }));

    const merged = mergeAabb(a, b);
    try std.testing.expectEqual(@as(f32, -1), merged.min[0]);
    try std.testing.expectEqual(@as(f32, 2), merged.max[0]);

    const sphere = aabbToSphere(a);
    try std.testing.expectApproxEqAbs(
        @as(f32, @sqrt(3)),
        sphere.radius,
        1e-5,
    );
    const closest = aabbClosestPoint(a, Vec3f{ 3, 0, 0 });
    try std.testing.expectEqual(@as(f32, 1), closest[0]);

    const points = [_]Vec3f{
        .{ 1, 2, 3 },
        .{ -1, 0, 5 },
    };
    const from_points = aabbFromPoints(&points).?;
    try std.testing.expectEqual(@as(f32, -1), from_points.min[0]);
    try std.testing.expectEqual(@as(f32, 5), from_points.max[2]);
    try std.testing.expect(
        aabbFromPoints(&[_]Vec3f{}) == null,
    );

    const from_center = aabbFromCenterHalfExtents(
        Vec3f{ 1, 1, 1 },
        Vec3f{ 2, 2, 2 },
    );
    try std.testing.expectEqual(@as(f32, -1), from_center.min[0]);
    try std.testing.expectEqual(@as(f32, 3), from_center.max[0]);
}

test "obb" {
    const box = Aabb(f32){
        .min = .{ 0, 0, 0 },
        .max = .{ 2, 4, 6 },
    };
    const oriented = obbFromAabb(box);
    try std.testing.expectApproxEqAbs(
        @as(f32, 1),
        oriented.center[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 2),
        oriented.half_extents[1],
        1e-6,
    );
    try std.testing.expect(obbContainsPoint(oriented, Vec3f{ 2, 4, 6 }));
    try std.testing.expect(!obbContainsPoint(oriented, Vec3f{ 3, 0, 0 }));

    const corner = obbCorner(oriented, 7);
    try std.testing.expectApproxEqAbs(@as(f32, 2), corner[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), corner[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), corner[2], 1e-6);

    const aligned = obbToAabb(oriented);
    try std.testing.expectApproxEqAbs(
        @as(f32, 0),
        aligned.min[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 6),
        aligned.max[2],
        1e-6,
    );
}

test "frustum" {
    const projection = mat.Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    });
    const frustum = frustumFromMatrix(projection);
    try std.testing.expect(frustumContainsPoint(
        frustum,
        Vec3f{ 0, 0, 0 },
    ));
    try std.testing.expect(!frustumContainsPoint(
        frustum,
        Vec3f{ 2, 0, 0 },
    ));
    try std.testing.expect(frustumIntersectsSphere(
        frustum,
        Sphere(f32){ .center = .{ 0, 0, 0 }, .radius = 0.5 },
    ));
    try std.testing.expect(!frustumIntersectsSphere(
        frustum,
        Sphere(f32){ .center = .{ 5, 0, 0 }, .radius = 0.5 },
    ));
    try std.testing.expect(frustumIntersectsAabb(
        frustum,
        Aabb(f32){
            .min = .{ -0.5, -0.5, -0.5 },
            .max = .{ 0.5, 0.5, 0.5 },
        },
    ));
    try std.testing.expect(!frustumIntersectsAabb(
        frustum,
        Aabb(f32){
            .min = .{ 2, 2, 2 },
            .max = .{ 3, 3, 3 },
        },
    ));
    try std.testing.expect(frustumIntersectsObb(
        frustum,
        obbFromAabb(Aabb(f32){
            .min = .{ -0.5, -0.5, -0.5 },
            .max = .{ 0.5, 0.5, 0.5 },
        }),
    ));
    try std.testing.expect(!frustumIntersectsObb(
        frustum,
        obbFromAabb(Aabb(f32){
            .min = .{ 2, 2, 2 },
            .max = .{ 3, 3, 3 },
        }),
    ));
}

test "frustum from a perspective matrix" {
    const projection = mat.Mat4(f32).fromRows(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, -1 },
        .{ 0, 0, 2, 0 },
    });
    const frustum = frustumFromMatrix(projection);

    try std.testing.expect(frustumContainsPoint(
        frustum,
        Vec3f{ 0, 0, -1.5 },
    ));
    try std.testing.expect(frustumIntersectsSphere(
        frustum,
        Sphere(f32){ .center = .{ 0, 0, -1.5 }, .radius = 0.1 },
    ));
    // Beyond the far plane at z = -1.
    try std.testing.expect(!frustumContainsPoint(
        frustum,
        Vec3f{ 0, 0, -0.5 },
    ));
    // Behind the near plane at z = -2.
    try std.testing.expect(!frustumContainsPoint(
        frustum,
        Vec3f{ 0, 0, -3 },
    ));
    // Outside the right plane at z = -2.
    try std.testing.expect(!frustumContainsPoint(
        frustum,
        Vec3f{ 3, 0, -2 },
    ));
    try std.testing.expect(frustumContainsPoint(
        frustum,
        Vec3f{ 1.9, 0, -2 },
    ));
    try std.testing.expect(frustumIntersectsAabb(
        frustum,
        Aabb(f32){
            .min = .{ -0.5, -0.5, -2.5 },
            .max = .{ 0.5, 0.5, -1.5 },
        },
    ));
    try std.testing.expect(!frustumIntersectsAabb(
        frustum,
        Aabb(f32){
            .min = .{ -0.5, -0.5, -0.5 },
            .max = .{ 0.5, 0.5, 0.0 },
        },
    ));
}

test "capsule" {
    const capsule = capsuleFromSegment(
        Vec3f{ 0, 0, 0 },
        Vec3f{ 0, 2, 0 },
        0.5,
    );
    try std.testing.expect(capsuleContainsPoint(
        capsule,
        Vec3f{ 0, 1, 0.4 },
    ));
    try std.testing.expect(!capsuleContainsPoint(
        capsule,
        Vec3f{ 0, 1, 0.6 },
    ));
    try std.testing.expect(capsuleIntersectsSphere(
        capsule,
        Sphere(f32){ .center = .{ 0, 1, 1 }, .radius = 0.5 },
    ));
    try std.testing.expect(!capsuleIntersectsSphere(
        capsule,
        Sphere(f32){ .center = .{ 0, 1, 2 }, .radius = 0.5 },
    ));
}
