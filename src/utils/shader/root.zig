//! Shader boundary helpers shared by host and SPIR-V code.
//!
//! Each helper describes one value type `T` once and
//! exposes matching accessors on the host, in the
//! shader, or between two stages. `T` must contain
//! scalars, vectors, fixed-size arrays of them, or
//! structs of them. Matrix types from `math` are
//! structs of column arrays, so they cross the
//! boundary unchanged.
//!
//! Use plain `struct` types for `T`: `extern struct`
//! cannot contain vectors on the host.
//!
//! An array of three-component vectors does not map
//! to the shader layout because its elements need
//! 16-byte alignment; such a field is rejected at
//! compile time.

const std = @import("std");
const builtin = @import("builtin");
const rhi = @import("render/rhi");

/// Describes the direction of a `DataStream`.
///
/// A stream may be an input, an output, or both. A
/// `DataStream` with neither flag set is rejected.
pub const Mode = packed struct {
    /// The stream is readable as a stage input.
    input: bool = false,
    /// The stream is writable as a stage output.
    output: bool = false,
};

/// Shares one vertex type between the host and a vertex
/// shader.
///
/// `submit` writes a slice of vertices into a
/// caller-owned host-visible buffer, and `load` reads
/// one vertex per shader invocation. Struct fields
/// become input locations in declaration order,
/// starting at 0.
///
/// # Args
/// - `T`: one vertex; a scalar, vector, fixed
///   array, or struct of them.
///
/// # Note
/// The host creates and owns a host-visible buffer
/// and passes it to `submit`. Only one `Vertex` type
/// can feed a shader because locations start at 0.
/// Do not write the same buffer concurrently.
///
/// # Compile Error
/// - `T` contains a type other than a scalar,
///   vector, fixed array, or struct of them.
/// - An integer is `isize` or `usize`.
/// - A struct is a tuple, packed, empty, or has
///   `comptime` or explicitly aligned fields.
/// - A vector has a length other than 2, 3, or 4.
/// - A 3-component vector has non-32-bit
///   components.
pub fn Vertex(comptime T: type) type {
    validateBoundary(T);

    return struct {
        const attributes = makeVertexAttributes(T);

        /// Layout of one vertex in the buffer that
        /// `submit` fills.
        ///
        /// Pass this in `PipelineGraphics.Desc`
        /// as the matching `vertex_buffers` entry.
        ///
        /// # Compile Error
        /// - A leaf type has no matching vertex format.
        pub const vertex_layout: rhi.Resource.PipelineGraphics.VertexBuffer = .{
            .stride = typeSize(T),
            .attributes = &attributes,
        };

        /// Uploads vertices to `buffer`.
        ///
        /// # Args
        /// - `value`: the vertices to serialize; each
        ///   element becomes one vertex.
        /// - `buffer`: a live host-visible buffer with
        ///   the vertex usage and room for every
        ///   element.
        ///
        /// # Error
        /// - `error.OutOfMemory`: the size overflows,
        ///   or host bookkeeping fails.
        /// - `error.InvalidResource`: `buffer` is not
        ///   live.
        /// - `error.NotHostVisible`: `buffer` was not
        ///   created with host memory.
        /// - `error.AlreadyMapped`: `buffer` is mapped.
        /// - `error.OutOfRange`: `buffer` is smaller
        ///   than `value.len` vertices.
        /// - `error.DeviceLost`, `error.DeviceFailure`:
        ///   the device is unusable.
        ///
        /// # Note
        /// The caller owns and destroys `buffer`, creates
        /// it with the stride in `vertex_layout`, and
        /// binds it before drawing. Record a draw with
        /// `value.len` vertices. The caller synchronizes
        /// the upload with GPU reads. An empty slice does
        /// nothing.
        ///
        /// # Compile Error
        /// - The target is SPIR-V, where host code
        ///   cannot run.
        /// - A leaf type has no matching vertex format.
        pub fn submit(
            value: []const T,
            buffer: rhi.Resource.Buffer,
        ) !void {
            if (comptime builtin.cpu.arch.isSpirv()) {
                @compileError("Vertex.submit is host-only");
            }
            if (value.len == 0) return;

            comptime validateVertex(T);

            const stride = comptime typeSize(T);
            const bytes = std.math.mul(usize, value.len, stride) catch
                return error.OutOfMemory;

            const memory = try buffer.map(0, bytes, .{ .write = true });
            defer buffer.unmap(&.{.{ .offset = 0, .size = bytes }});

            @memset(memory, 0);
            for (value, 0..) |vertex, index| {
                writeValue(T, memory, index * stride, vertex);
            }
        }

        /// Loads the current vertex from the shader input.
        ///
        /// # Return
        /// The vertex value for this shader invocation.
        ///
        /// # Note
        /// The host binds the buffer passed to `submit`
        /// with the layout in `vertex_layout`.
        ///
        /// # Compile Error
        /// - The target is not SPIR-V.
        /// - The host layout does not match the shader
        ///   type layout.
        pub fn load() T {
            if (comptime !builtin.cpu.arch.isSpirv()) {
                @compileError("Vertex.load is shader-only");
            }

            comptime verifyLayout(T);

            const input = @extern(*addrspace(.input) Mirror(T), .{
                .name = "carn.vertex." ++ @typeName(T),
                .decoration = .{ .location = 0 },
            });
            return fromMirror(T, input.*);
        }
    };
}

/// Shares one uniform block between the host and
/// shaders.
///
/// `write` uploads a value to a host-visible uniform
/// buffer, and shaders read it at descriptor set 0,
/// binding 0.
///
/// # Args
/// - `T`: the uniform block; a non-empty struct of
///   scalars, vectors, fixed arrays, or nested
///   structs.
///
/// # Note
/// The host creates a host-visible buffer and a
/// descriptor set bound at `set` and `binding`, then
/// passes the buffer to `write`. Because the set and
/// binding are fixed, a pipeline can use only one
/// `Uniform` type. Do not write the same buffer
/// concurrently. The caller synchronizes uploads
/// with GPU reads. The device must support the
/// component widths in `T`.
///
/// # Compile Error
/// - `T` is not a struct type.
/// - `T` violates the boundary type rules listed on
///   `Vertex`.
pub fn Uniform(comptime T: type) type {
    switch (@typeInfo(T)) {
        .@"struct" => {},
        else => @compileError("Uniform: `T` must be a struct type"),
    }
    validateBoundary(T);

    return struct {
        /// Descriptor set that shaders read.
        pub const set: u32 = 0;
        /// Binding index that shaders read.
        pub const binding: u32 = 0;

        /// Minimum byte size accepted by `write`.
        ///
        /// The block is rounded up to a 16-byte
        /// boundary.
        pub const buffer_size: usize = alignForward(typeSize(T), 16);

        /// Serializes `value` into `buffer`.
        ///
        /// # Args
        /// - `value`: the uniform value to upload.
        /// - `buffer`: a live host-visible buffer with
        ///   the uniform usage and at least
        ///   `buffer_size` bytes.
        ///
        /// # Error
        /// - `error.OutOfMemory`: host bookkeeping fails.
        /// - `error.InvalidResource`: `buffer` is not
        ///   live.
        /// - `error.NotHostVisible`: `buffer` was not
        ///   created with host memory.
        /// - `error.AlreadyMapped`: `buffer` is mapped.
        /// - `error.OutOfRange`: `buffer` is too small.
        /// - `error.DeviceLost`, `error.DeviceFailure`:
        ///   the device is unusable.
        ///
        /// # Note
        /// The caller owns and destroys `buffer` and
        /// binds it to the descriptor set at `set` and
        /// `binding`. The caller synchronizes the upload
        /// with GPU reads.
        ///
        /// # Compile Error
        /// - The target is SPIR-V, where host code
        ///   cannot run.
        pub fn write(
            value: T,
            buffer: rhi.Resource.Buffer,
        ) !void {
            if (comptime builtin.cpu.arch.isSpirv()) {
                @compileError("Uniform.write is host-only");
            }

            const size = buffer_size;
            const memory = try buffer.map(0, size, .{ .write = true });
            defer buffer.unmap(&.{.{ .offset = 0, .size = size }});

            @memset(memory, 0);
            writeValue(T, memory, 0, value);
        }

        /// Reads the current uniform block.
        ///
        /// # Return
        /// The value most recently written by `write`.
        ///
        /// # Compile Error
        /// - The target is not SPIR-V.
        /// - The host layout does not match the shader
        ///   type layout.
        pub fn read() T {
            if (comptime !builtin.cpu.arch.isSpirv()) {
                @compileError("Uniform.read is shader-only");
            }

            comptime verifyLayout(T);

            const uniform_value = @extern(*addrspace(.uniform) Mirror(T), .{
                .name = "carn.uniform." ++ @typeName(T),
                .decoration = .{ .descriptor = .{
                    .set = set,
                    .binding = binding,
                } },
            });
            return fromMirror(T, uniform_value.*);
        }
    };
}

/// Shares one stage-to-stage value between two shaders.
///
/// `write` stores a value in the output interface and
/// `read` loads it from the input interface. Locations
/// start at 0 and follow the struct field order.
///
/// # Args
/// - `T`: the stream value; a scalar, vector, or
///   struct of them.
/// - `mode`: the enabled directions.
///
/// # Note
/// All streams start at location 0, so one shader pair
/// can carry only one `DataStream` type. The device
/// must support the component widths in `T`.
///
/// # Compile Error
/// - `mode` enables neither input nor output.
/// - `T` violates the boundary type rules listed on
///   `Vertex`.
pub fn DataStream(comptime T: type, comptime mode: Mode) type {
    if (!mode.input and !mode.output) {
        @compileError("DataStream: `mode` must enable input or output");
    }
    validateBoundary(T);

    return struct {
        /// Reads the stage input.
        ///
        /// # Return
        /// The value written by the previous stage.
        ///
        /// # Compile Error
        /// - The target is not SPIR-V.
        /// - `mode.input` is false.
        /// - The host layout does not match the shader
        ///   type layout.
        pub fn read() T {
            if (comptime !builtin.cpu.arch.isSpirv()) {
                @compileError("DataStream.read is shader-only");
            }
            if (comptime !mode.input) {
                @compileError("DataStream.read needs `mode.input`");
            }

            comptime verifyLayout(T);

            const input = @extern(*addrspace(.input) Mirror(T), .{
                .name = "carn.stream.input." ++ @typeName(T),
                .decoration = .{ .location = 0 },
            });
            return fromMirror(T, input.*);
        }

        /// Writes the stage output.
        ///
        /// # Args
        /// - `value`: the value passed to the next stage.
        ///
        /// # Compile Error
        /// - The target is not SPIR-V.
        /// - `mode.output` is false.
        /// - The host layout does not match the shader
        ///   type layout.
        pub fn write(value: T) void {
            if (comptime !builtin.cpu.arch.isSpirv()) {
                @compileError("DataStream.write is shader-only");
            }
            if (comptime !mode.output) {
                @compileError("DataStream.write needs `mode.output`");
            }

            comptime verifyLayout(T);

            const output = @extern(*addrspace(.output) Mirror(T), .{
                .name = "carn.stream.output." ++ @typeName(T),
                .decoration = .{ .location = 0 },
            });
            output.* = toMirror(T, value);
        }
    };
}

/// Rejects types that cannot cross the shader boundary.
fn validateBoundary(comptime T: type) void {
    if (T == isize or T == usize) {
        @compileError(
            "carn shader: use a fixed-width integer, not `" ++
                @typeName(T) ++ "`",
        );
    }

    switch (@typeInfo(T)) {
        .int => |info| {
            if (!validIntBits(info.bits)) {
                @compileError(
                    "carn shader: unsupported integer width `" ++
                        @typeName(T) ++ "`",
                );
            }
        },
        .float => |info| {
            if (info.bits != 16 and info.bits != 32 and info.bits != 64) {
                @compileError(
                    "carn shader: unsupported float width `" ++
                        @typeName(T) ++ "`",
                );
            }
        },
        .vector => |info| {
            if (info.len < 2 or info.len > 4) {
                @compileError(
                    "carn shader: vector length must be 2, 3, or 4",
                );
            }
            validateScalar(info.child);
            if (info.len == 3 and scalarBits(info.child) != 32) {
                @compileError(
                    "carn shader: a 3-component vector needs 32-bit " ++
                        "components",
                );
            }
        },
        .array => |info| validateBoundary(info.child),
        .@"struct" => |info| {
            if (info.is_tuple) {
                @compileError(
                    "carn shader: `" ++ @typeName(T) ++
                        "` is a tuple; use named fields",
                );
            }
            if (info.layout == .@"packed") {
                @compileError(
                    "carn shader: `" ++ @typeName(T) ++
                        "` must not be packed",
                );
            }
            if (info.field_names.len == 0) {
                @compileError(
                    "carn shader: `" ++ @typeName(T) ++
                        "` must declare at least one field",
                );
            }
            inline for (
                info.field_names,
                info.field_types,
                info.field_attrs,
            ) |name, FieldTy, attr| {
                if (attr.@"comptime") {
                    @compileError(
                        "carn shader: field `" ++ @typeName(T) ++ "." ++
                            name ++ "` must not be `comptime`",
                    );
                }
                if (attr.@"align" != null) {
                    @compileError(
                        "carn shader: field `" ++ @typeName(T) ++ "." ++
                            name ++ "` must not set `align`",
                    );
                }
                validateBoundary(FieldTy);
            }
        },
        else => @compileError(
            "carn shader: `" ++ @typeName(T) ++
                "` is not a scalar, vector, or struct",
        ),
    }
}

/// Rejects scalar types that cannot be vector components.
fn validateScalar(comptime T: type) void {
    if (T == isize or T == usize) {
        @compileError(
            "carn shader: use a fixed-width integer, not `" ++
                @typeName(T) ++ "`",
        );
    }

    switch (@typeInfo(T)) {
        .int => |info| {
            if (!validIntBits(info.bits)) {
                @compileError(
                    "carn shader: unsupported integer width `" ++
                        @typeName(T) ++ "`",
                );
            }
        },
        .float => |info| {
            if (info.bits != 16 and info.bits != 32 and info.bits != 64) {
                @compileError(
                    "carn shader: unsupported float width `" ++
                        @typeName(T) ++ "`",
                );
            }
        },
        else => @compileError(
            "carn shader: vector components must be integers or floats",
        ),
    }
}

fn validIntBits(comptime bits: u16) bool {
    return bits == 8 or bits == 16 or bits == 32 or bits == 64;
}

fn scalarBits(comptime T: type) u16 {
    return switch (@typeInfo(T)) {
        .int => |info| info.bits,
        .float => |info| info.bits,
        else => unreachable,
    };
}

/// Returns the byte size of `T` at the shader boundary.
fn typeSize(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .int, .float => scalarSize(T),
        .vector => |info| blk: {
            const element = scalarSize(info.child);
            if (info.len == 3) break :blk 4 * element;
            break :blk element * info.len;
        },
        .@"struct" => |info| blk: {
            comptime var size: usize = 0;
            comptime var alignment: usize = 1;
            inline for (info.field_types) |FieldTy| {
                const field_alignment = comptime typeAlign(FieldTy);
                alignment = @max(alignment, field_alignment);
                size = comptime alignForward(size, field_alignment);
                size += comptime typeSize(FieldTy);
            }
            break :blk comptime alignForward(size, alignment);
        },
        .array => |info| blk: {
            const stride = comptime alignForward(
                typeSize(info.child),
                typeAlign(info.child),
            );
            break :blk info.len * stride;
        },
        else => unreachable,
    };
}

/// Returns the byte alignment of `T` at the boundary.
fn typeAlign(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .int, .float => scalarSize(T),
        .vector => |info| blk: {
            const element = scalarSize(info.child);
            if (info.len == 3) break :blk 4 * element;
            break :blk element * info.len;
        },
        .@"struct" => |info| blk: {
            comptime var alignment: usize = 1;
            inline for (info.field_types) |FieldTy| {
                alignment = @max(
                    alignment,
                    comptime typeAlign(FieldTy),
                );
            }
            break :blk alignment;
        },
        .array => |info| typeAlign(info.child),
        else => unreachable,
    };
}

fn scalarSize(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .int => |info| info.bits / 8,
        .float => |info| info.bits / 8,
        else => unreachable,
    };
}

fn hostValueSize(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .vector => |info| blk: {
            if (info.len == 3) {
                break :blk 3 * scalarSize(info.child);
            }
            break :blk @sizeOf(T);
        },
        else => @sizeOf(T),
    };
}

fn alignForward(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

fn fieldOffset(comptime T: type, comptime name: []const u8) usize {
    const info = @typeInfo(T).@"struct";
    comptime var offset: usize = 0;
    inline for (info.field_names, info.field_types) |field_name, FieldTy| {
        offset = comptime alignForward(offset, typeAlign(FieldTy));
        if (std.mem.eql(u8, field_name, name)) return offset;
        offset += comptime typeSize(FieldTy);
    }
    unreachable;
}

/// Copies `value` into `dst` at the shader layout offsets.
fn writeValue(
    comptime T: type,
    dst: []u8,
    base: usize,
    value: T,
) void {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, info.field_types) |name, FieldTy| {
                writeValue(
                    FieldTy,
                    dst,
                    base + fieldOffset(T, name),
                    @field(value, name),
                );
            }
        },
        .array => |info| {
            inline for (0..info.len) |index| {
                writeValue(
                    info.child,
                    dst,
                    base + index * typeSize(info.child),
                    value[index],
                );
            }
        },
        else => {
            const host_size = comptime hostValueSize(T);
            const shader_size = comptime typeSize(T);
            if (host_size > shader_size) {
                @compileError(
                    "carn shader: host type is wider than the shader " ++
                        "layout",
                );
            }
            const bytes = std.mem.asBytes(&value)[0..host_size];
            @memcpy(dst[base..][0..host_size], bytes);
        },
    }
}

/// Builds the extern-compatible type used by the shader.
fn Mirror(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            var names: [info.field_names.len][:0]const u8 = undefined;
            var types: [info.field_types.len]type = undefined;
            var attrs: [info.field_types.len]std.builtin.Type.Struct.FieldAttributes = undefined;
            inline for (info.field_names, info.field_types, 0..) |
                name,
                FieldTy,
                i,
            | {
                names[i] = name;
                types[i] = Mirror(FieldTy);
                attrs[i] = .{};
            }
            break :blk @Struct(.@"extern", null, &names, &types, &attrs);
        },
        .array => |info| [info.len]Mirror(info.child),
        else => T,
    };
}

fn fromMirror(comptime T: type, value: Mirror(T)) T {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            var result: T = undefined;
            inline for (info.field_names, info.field_types) |name, FieldTy| {
                @field(result, name) = fromMirror(
                    FieldTy,
                    @field(value, name),
                );
            }
            break :blk result;
        },
        else => value,
    };
}

fn toMirror(comptime T: type, value: T) Mirror(T) {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            var result: Mirror(T) = undefined;
            inline for (info.field_names, info.field_types) |name, FieldTy| {
                @field(result, name) = toMirror(
                    FieldTy,
                    @field(value, name),
                );
            }
            break :blk result;
        },
        else => value,
    };
}

/// Checks the host layout against the shader layout.
fn verifyLayout(comptime T: type) void {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            const ShaderType = Mirror(T);
            if (@sizeOf(ShaderType) != typeSize(T)) {
                @compileError(
                    "carn shader: size mismatch for `" ++
                        @typeName(T) ++ "`",
                );
            }
            inline for (info.field_names, info.field_types) |name, FieldTy| {
                if (@offsetOf(ShaderType, name) != fieldOffset(T, name)) {
                    @compileError(
                        "carn shader: offset mismatch for `" ++
                            @typeName(T) ++ "." ++ name ++ "`",
                    );
                }
                verifyLayout(FieldTy);
            }
        },
        .array => |info| {
            if (@sizeOf(Mirror(T)) != typeSize(T)) {
                @compileError(
                    "carn shader: array layout mismatch for `" ++
                        @typeName(T) ++ "`",
                );
            }
            verifyLayout(info.child);
        },
        else => {
            if (@sizeOf(T) != typeSize(T)) {
                @compileError(
                    "carn shader: size mismatch for `" ++
                        @typeName(T) ++ "`",
                );
            }
        },
    }
}

fn validateVertex(comptime T: type) void {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_types) |FieldTy| validateVertex(FieldTy);
        },
        .array => |info| validateVertex(info.child),
        else => {
            _ = vertexFormat(T);
        },
    }
}

fn vertexFormat(
    comptime T: type,
) rhi.Resource.PipelineGraphics.VertexFormat {
    switch (@typeInfo(T)) {
        .float => |info| {
            if (info.bits == 32) return .f32;
            @compileError(
                "Vertex: no vertex format for `" ++ @typeName(T) ++ "`",
            );
        },
        .int => |info| {
            if (info.signedness == .unsigned and info.bits == 32) {
                return .u32;
            }
            @compileError(
                "Vertex: no vertex format for `" ++ @typeName(T) ++ "`",
            );
        },
        .vector => |info| return vectorFormat(T, info),
        else => @compileError(
            "Vertex: `T` must contain only scalars and vectors",
        ),
    }
}

fn vectorFormat(
    comptime T: type,
    comptime info: std.builtin.Type.Vector,
) rhi.Resource.PipelineGraphics.VertexFormat {
    switch (info.len) {
        2 => switch (@typeInfo(info.child)) {
            .float => |f| if (f.bits == 32) return .f32x2,
            .int => |i| {
                if (i.signedness == .unsigned) {
                    if (i.bits == 8) return .u8x2;
                    if (i.bits == 16) return .u16x2;
                    if (i.bits == 32) return .u32x2;
                } else {
                    if (i.bits == 8) return .i8x2;
                    if (i.bits == 16) return .i16x2;
                }
            },
            else => {},
        },
        3 => switch (@typeInfo(info.child)) {
            .float => |f| if (f.bits == 32) return .f32x3,
            .int => |i| if (i.signedness == .unsigned and i.bits == 32) {
                return .u32x3;
            },
            else => {},
        },
        4 => switch (@typeInfo(info.child)) {
            .float => |f| if (f.bits == 32) return .f32x4,
            .int => |i| {
                if (i.signedness == .unsigned) {
                    if (i.bits == 8) return .u8x4;
                    if (i.bits == 16) return .u16x4;
                    if (i.bits == 32) return .u32x4;
                } else {
                    if (i.bits == 8) return .i8x4;
                    if (i.bits == 16) return .i16x4;
                }
            },
            else => {},
        },
        else => {},
    }
    @compileError(
        "Vertex: no vertex format for `" ++ @typeName(T) ++ "`",
    );
}

fn vertexAttributeCount(comptime T: type) usize {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            comptime var count: usize = 0;
            inline for (info.field_types) |FieldTy| {
                count += comptime vertexAttributeCount(FieldTy);
            }
            break :blk count;
        },
        .array => |info| info.len * vertexAttributeCount(info.child),
        else => 1,
    };
}

fn makeVertexAttributes(
    comptime T: type,
) [vertexAttributeCount(T)]rhi.Resource.PipelineGraphics.VertexAttribute {
    comptime validateVertex(T);
    const Attribute = rhi.Resource.PipelineGraphics.VertexAttribute;
    var attributes: [vertexAttributeCount(T)]Attribute = undefined;
    var next: usize = 0;
    fillVertexAttributes(T, 0, &attributes, &next);
    if (next != attributes.len) unreachable;
    return attributes;
}

fn fillVertexAttributes(
    comptime T: type,
    base: usize,
    dst: []rhi.Resource.PipelineGraphics.VertexAttribute,
    next: *usize,
) void {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, info.field_types) |name, FieldTy| {
                fillVertexAttributes(
                    FieldTy,
                    base + fieldOffset(T, name),
                    dst,
                    next,
                );
            }
        },
        .array => |info| {
            inline for (0..info.len) |index| {
                fillVertexAttributes(
                    info.child,
                    base + index * typeSize(info.child),
                    dst,
                    next,
                );
            }
        },
        else => {
            dst[next.*] = .{
                .location = @intCast(next.*),
                .offset = @intCast(base),
                .format = vertexFormat(T),
            };
            next.* += 1;
        },
    }
}

test "vertex metadata follows field order" {
    const VertexData = struct {
        position: @Vector(3, f32),
        color: @Vector(4, f32),
    };
    const V = Vertex(VertexData);

    try std.testing.expectEqual(@as(u32, 32), V.vertex_layout.stride);
    try std.testing.expectEqual(@as(usize, 2), V.vertex_layout.attributes.len);
    try std.testing.expectEqual(
        @as(u32, 0),
        V.vertex_layout.attributes[0].location,
    );
    try std.testing.expectEqual(
        @as(u32, 0),
        V.vertex_layout.attributes[0].offset,
    );
    try std.testing.expectEqual(
        .f32x3,
        V.vertex_layout.attributes[0].format,
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        V.vertex_layout.attributes[1].location,
    );
    try std.testing.expectEqual(
        @as(u32, 16),
        V.vertex_layout.attributes[1].offset,
    );
    try std.testing.expectEqual(
        .f32x4,
        V.vertex_layout.attributes[1].format,
    );
}

test "vertex submit uploads a slice" {
    const null_backend = @import("render/null");

    var backend = try null_backend.init(std.testing.allocator, .{});
    defer {
        const status = backend.deinit();
        std.testing.expect(status.ok()) catch @panic("leaked resources");
    }

    const VertexData = struct {
        position: @Vector(3, f32),
        color: @Vector(4, f32),
    };
    const V = Vertex(VertexData);
    const stride: usize = V.vertex_layout.stride;
    const buffer = try rhi.Resource.Buffer.create(backend.device(), .{
        .label = "test vertices",
        .size = 2 * stride,
        .usage = .{ .vertex = true },
        .memory = .host,
    });
    defer buffer.destroy();

    var first: VertexData = undefined;
    @memset(std.mem.asBytes(&first), 0);
    first.position = .{ 1, 2, 3 };
    first.color = .{ 4, 5, 6, 7 };

    var second: VertexData = undefined;
    @memset(std.mem.asBytes(&second), 0);
    second.position = .{ 8, 9, 10 };
    second.color = .{ 11, 12, 13, 14 };

    const empty: [0]VertexData = .{};
    try V.submit(&empty, buffer);

    const one = [_]VertexData{first};
    try V.submit(&one, buffer);

    const both = [_]VertexData{ first, second };
    try V.submit(&both, buffer);

    const bytes = try buffer.map(0, 2 * stride, .{ .read = true });
    defer buffer.unmap(&.{});

    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&first),
        bytes[0..stride],
    );
    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&second),
        bytes[stride..][0..stride],
    );

    const too_small = try rhi.Resource.Buffer.create(
        backend.device(),
        .{
            .label = "too small",
            .size = stride,
            .usage = .{ .vertex = true },
            .memory = .host,
        },
    );
    defer too_small.destroy();
    try std.testing.expectError(
        error.OutOfRange,
        V.submit(&both, too_small),
    );
}

test "uniform write uploads a block" {
    const null_backend = @import("render/null");

    var backend = try null_backend.init(std.testing.allocator, .{});
    defer {
        const status = backend.deinit();
        std.testing.expect(status.ok()) catch @panic("leaked resources");
    }

    const UniformData = struct {
        camera: @Vector(4, f32),
        time: f32,
    };
    const U = Uniform(UniformData);
    const buffer = try rhi.Resource.Buffer.create(backend.device(), .{
        .label = "test uniform",
        .size = U.buffer_size,
        .usage = .{ .uniform = true },
        .memory = .host,
    });
    defer buffer.destroy();

    var value: UniformData = undefined;
    @memset(std.mem.asBytes(&value), 0);
    value.camera = .{ 1, 2, 3, 4 };
    value.time = 5;
    try U.write(value, buffer);

    const bytes = try buffer.map(0, U.buffer_size, .{ .read = true });
    defer buffer.unmap(&.{});

    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&value),
        bytes[0..@sizeOf(UniformData)],
    );
    try std.testing.expectEqual(@as(u32, 0), U.set);
    try std.testing.expectEqual(@as(u32, 0), U.binding);
}

test "uniform small block is padded" {
    const null_backend = @import("render/null");

    var backend = try null_backend.init(std.testing.allocator, .{});
    defer {
        const status = backend.deinit();
        std.testing.expect(status.ok()) catch @panic("leaked resources");
    }

    const U = Uniform(struct { value: f32 });
    const buffer = try rhi.Resource.Buffer.create(backend.device(), .{
        .label = "test small uniform",
        .size = U.buffer_size,
        .usage = .{ .uniform = true },
        .memory = .host,
    });
    defer buffer.destroy();

    const value: f32 = 3.5;
    try U.write(.{ .value = value }, buffer);

    const bytes = try buffer.map(0, U.buffer_size, .{ .read = true });
    defer buffer.unmap(&.{});

    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&value),
        bytes[0..@sizeOf(f32)],
    );
    for (bytes[4..16]) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "data stream mode is preserved" {
    const Stream = DataStream(
        struct { color: @Vector(4, f32) },
        .{ .input = true, .output = true },
    );
    const Input = DataStream(
        struct { color: @Vector(4, f32) },
        .{ .input = true },
    );
    const Output = DataStream(
        struct { color: @Vector(4, f32) },
        .{ .output = true },
    );

    try std.testing.expect(@hasDecl(Stream, "read"));
    try std.testing.expect(@hasDecl(Stream, "write"));
    try std.testing.expect(@hasDecl(Input, "read"));
    try std.testing.expect(@hasDecl(Output, "write"));
}

test "math matrices forward through the boundary" {
    const math = @import("math");
    const null_backend = @import("render/null");

    const Camera = struct {
        view_projection: math.Mat4(f32),
        tint: math.Vec4(f32),
    };
    const U = Uniform(Camera);
    try std.testing.expectEqual(@as(usize, 80), U.buffer_size);

    const MeshVertex = struct {
        transform: math.Mat4(f32),
        position: math.Vec4(f32),
    };
    const layout = Vertex(MeshVertex).vertex_layout;
    try std.testing.expectEqual(@as(u32, 80), layout.stride);
    try std.testing.expectEqual(@as(usize, 5), layout.attributes.len);
    try std.testing.expectEqual(@as(u32, 0), layout.attributes[0].location);
    try std.testing.expectEqual(@as(u32, 4), layout.attributes[4].location);
    try std.testing.expectEqual(@as(u32, 64), layout.attributes[4].offset);

    var backend = try null_backend.init(std.testing.allocator, .{});
    defer {
        const status = backend.deinit();
        std.testing.expect(status.ok()) catch @panic("leaked resources");
    }

    const buffer = try rhi.Resource.Buffer.create(backend.device(), .{
        .label = "test math uniform",
        .size = U.buffer_size,
        .usage = .{ .uniform = true },
        .memory = .host,
    });
    defer buffer.destroy();

    var value: Camera = undefined;
    @memset(std.mem.asBytes(&value), 0);
    value.view_projection = math.Mat4(f32).fromRows(.{
        math.Vec4(f32).fromElements(.{ 1, 2, 3, 4 }),
        math.Vec4(f32).fromElements(.{ 5, 6, 7, 8 }),
        math.Vec4(f32).fromElements(.{ 9, 10, 11, 12 }),
        math.Vec4(f32).fromElements(.{ 13, 14, 15, 16 }),
    });
    value.tint = math.Vec4(f32)
        .fromElements(.{ 0.25, 0.5, 0.75, 1 });
    try U.write(value, buffer);

    const bytes = try buffer.map(0, U.buffer_size, .{ .read = true });
    defer buffer.unmap(&.{});

    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&value.view_projection),
        bytes[0..@sizeOf(math.Mat4(f32))],
    );
    try std.testing.expectEqualSlices(
        u8,
        std.mem.asBytes(&value.tint),
        bytes[64..80],
    );
}
