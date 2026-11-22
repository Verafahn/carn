//! Shader code and its entry point.
//!
//! A shader handle pairs compiled code with one stage
//! and one entry point. Pipelines reference shader
//! handles; they do not copy code.

const Resource = @import("../Resource.zig");
const Device = @import("../Device.zig");
const Shader = @This();
const std = @import("std");

/// Code, language and entry point of one stage.
pub const Desc = struct {
    /// Debug name reported by validation.
    label: []const u8 = "",
    /// Shader bytes; borrowed for the create call.
    code: []const u8,
    /// Language that `code` is written in.
    language: Language = .spirv,
    /// Pipeline stage that executes `code`.
    stage: Stage,
    /// Entry point name inside `code`.
    entry: []const u8 = "main",
};

/// Pipeline stages supported by the RHI.
pub const Stage = enum {
    /// Vertex processing.
    vertex,
    /// Fragment processing.
    fragment,
    /// Compute processing.
    compute,
};

/// A set of shader stages.
pub const StageFlags = packed struct {
    /// Vertex processing.
    vertex: bool = false,
    /// Fragment processing.
    fragment: bool = false,
    /// Compute processing.
    compute: bool = false,
};

/// Languages accepted by `Shader.Desc`.
pub const Language = enum {
    /// SPIR-V bytecode.
    spirv,
    /// WGSL source.
    wgsl,
    /// GLSL source.
    glsl,
    /// HLSL source.
    hlsl,
};

/// Backend handle and owning device.
handle: Resource,
device: Device,

/// Creates a shader and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `device`: the device that owns the shader.
/// - `desc`: code, language, stage and entry point.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live shader; release it with `destroy`.
///
/// # Error
/// The error contract matches `create`.
pub fn createWithAddr(
    device: Device,
    desc: Desc,
    ret_addr: usize,
) Device.CreateError!Shader {
    const rc_desc: Resource.Desc = .{ .shader = desc };
    const handle = try device.rawCreate(&rc_desc, ret_addr);
    return .{
        .handle = handle,
        .device = device,
    };
}

/// Creates a shader on `device`.
///
/// # Args
/// - `device`: the device that owns the shader.
/// - `desc`: code, language, stage and entry point.
///
/// # Return
/// A live shader; release it with `destroy`.
///
/// # Error
/// - `error.InvalidDesc`: `code` is empty or cannot
///   be compiled for `stage`.
/// - `error.Unsupport`: the backend cannot consume
///   `language` or `stage`.
/// - `error.TooManyObjects`: the backend cannot track
///   another shader.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is no longer usable.
pub fn create(device: Device, desc: Desc) Device.CreateError!Shader {
    return createWithAddr(device, desc, @returnAddress());
}

/// Destroys the shader and attributes diagnostics to
/// `ret_addr`.
///
/// # Args
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// The lifetime contract matches `destroy`.
///
/// # Panic
/// In safety builds, the handle kind is not `shader`.
pub fn destroyWithAddr(self: Shader, ret_addr: usize) void {
    std.debug.assert(self.handle.kind == .shader);
    self.device.rawDestroy(self.handle, ret_addr);
}

/// Destroys the shader.
///
/// # Note
/// Pipelines created from the shader must not be in
/// use. The handle must not be used afterwards.
///
/// # Panic
/// In safety builds, the handle kind is not `shader`.
pub fn destroy(self: Shader) void {
    self.destroyWithAddr(@returnAddress());
}
