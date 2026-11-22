//! Conversions from RHI descriptions to Vulkan enumerants.
//!
//! Every function here is pure: it maps an RHI value to
//! the Vulkan value the backend uses, or returns null
//! when the RHI value has no Vulkan equivalent.

const rhi = @import("rhi");
const vk = @import("vk");

const Resource = rhi.Resource;
const Device = rhi.Device;
const Texture = Resource.Texture;
const Buffer = Resource.Buffer;
const Shader = Resource.Shader;
const Sampler = Resource.Sampler;
const Swapchain = Resource.Swapchain;
const PipelineGraphics = Resource.PipelineGraphics;
const DescriptorSetLayout = Resource.DescriptorSetLayout;

/// Converts an RHI pixel format to a Vulkan format.
///
/// # Args
/// - `format`: format to convert.
///
/// # Return
/// The matching `vk.Format`, or null when the format
/// cannot be represented.
pub fn formatToVk(format: Resource.Format) ?vk.Format {
    return switch (format) {
        .r8_unorm => .r8_unorm,
        .r8_snorm => .r8_snorm,
        .r8_uint => .r8_uint,
        .r8_sint => .r8_sint,
        .r8g8_unorm => .r8g8_unorm,
        .r8g8_snorm => .r8g8_snorm,
        .r8g8_uint => .r8g8_uint,
        .r8g8_sint => .r8g8_sint,
        .r8g8b8a8_unorm => .r8g8b8a8_unorm,
        .r8g8b8a8_snorm => .r8g8b8a8_snorm,
        .r8g8b8a8_uint => .r8g8b8a8_uint,
        .r8g8b8a8_sint => .r8g8b8a8_sint,
        .r8g8b8a8_srgb => .r8g8b8a8_srgb,
        .b8g8r8a8_unorm => .b8g8r8a8_unorm,
        .b8g8r8a8_srgb => .b8g8r8a8_srgb,
        .r16_uint => .r16_uint,
        .r16_sint => .r16_sint,
        .r16_float => .r16_sfloat,
        .r16g16_uint => .r16g16_uint,
        .r16g16_sint => .r16g16_sint,
        .r16g16_float => .r16g16_sfloat,
        .r16g16b16a16_uint => .r16g16b16a16_uint,
        .r16g16b16a16_sint => .r16g16b16a16_sint,
        .r16g16b16a16_float => .r16g16b16a16_sfloat,
        .r32_uint => .r32_uint,
        .r32_sint => .r32_sint,
        .r32_float => .r32_sfloat,
        .r32g32_uint => .r32g32_uint,
        .r32g32_sint => .r32g32_sint,
        .r32g32_float => .r32g32_sfloat,
        .r32g32b32_uint => .r32g32b32_uint,
        .r32g32b32_sint => .r32g32b32_sint,
        .r32g32b32_float => .r32g32b32_sfloat,
        .r32g32b32a32_uint => .r32g32b32a32_uint,
        .r32g32b32a32_sint => .r32g32b32a32_sint,
        .r32g32b32a32_float => .r32g32b32a32_sfloat,
        .r10g10b10a2_unorm => .a2b10g10r10_unorm_pack32,
        .b10g10r10a2_unorm => .a2r10g10b10_unorm_pack32,
        .r11g11b10_float => .b10g11r11_ufloat_pack32,
        .depth16_unorm => .d16_unorm,
        .depth24_plus_stencil8 => .d24_unorm_s8_uint,
        .depth32_float => .d32_sfloat,
        .stencil8 => .s8_uint,
    };
}

/// Reports the planes of a format.
///
/// # Args
/// - `format`: format to inspect.
///
/// # Return
/// The aspect mask used by image views, barriers
/// and copies.
pub fn formatAspect(format: Resource.Format) vk.ImageAspectFlags {
    return switch (format) {
        .depth16_unorm, .depth32_float => .{ .depth = true },
        .depth24_plus_stencil8 => .{ .depth = true, .stencil = true },
        .stencil8 => .{ .stencil = true },
        else => .{ .color = true },
    };
}

/// Selects the primary plane of a format.
///
/// # Args
/// - `pixel_format`: format to inspect.
///
/// # Return
/// Depth for formats with depth, otherwise stencil
/// for stencil-only formats, otherwise color. Single
/// aspect copies and views use this mask.
pub fn primaryAspect(pixel_format: Resource.Format) vk.ImageAspectFlags {
    const aspect = formatAspect(pixel_format);
    if (aspect.depth) return .{ .depth = true };
    if (aspect.stencil) return .{ .stencil = true };
    return .{ .color = true };
}

/// Reports whether a format carries depth or stencil.
///
/// # Args
/// - `format`: format to inspect.
///
/// # Return
/// True for the four depth-stencil formats.
pub fn isDepthStencil(format: Resource.Format) bool {
    return switch (format) {
        .depth16_unorm,
        .depth24_plus_stencil8,
        .depth32_float,
        .stencil8,
        => true,
        else => false,
    };
}

/// Reports whether a format has a stencil plane.
///
/// # Args
/// - `format`: format to inspect.
///
/// # Return
/// True when the format contains a stencil channel.
pub fn hasStencil(format: Resource.Format) bool {
    return switch (format) {
        .depth24_plus_stencil8, .stencil8 => true,
        else => false,
    };
}

/// Reports whether a format has integer channels.
///
/// # Args
/// - `pixel_format`: format to inspect.
///
/// # Return
/// True for unsigned and signed integer formats.
pub fn isIntegerFormat(pixel_format: Resource.Format) bool {
    return switch (pixel_format) {
        .r8_uint,
        .r8_sint,
        .r8g8_uint,
        .r8g8_sint,
        .r8g8b8a8_uint,
        .r8g8b8a8_sint,
        .r16_uint,
        .r16_sint,
        .r16g16_uint,
        .r16g16_sint,
        .r16g16b16a16_uint,
        .r16g16b16a16_sint,
        .r32_uint,
        .r32_sint,
        .r32g32_uint,
        .r32g32_sint,
        .r32g32b32_uint,
        .r32g32b32_sint,
        .r32g32b32a32_uint,
        .r32g32b32a32_sint,
        => true,
        else => false,
    };
}

/// Returns the byte size of one texel.
///
/// # Args
/// - `format`: format to measure.
///
/// # Return
/// Packed texel size in bytes. Used to convert row
/// strides when recording buffer/image copies.
pub fn bytesPerTexel(format: Resource.Format) u32 {
    return switch (format) {
        .r8_unorm,
        .r8_snorm,
        .r8_uint,
        .r8_sint,
        .stencil8,
        => 1,
        .r8g8_unorm,
        .r8g8_snorm,
        .r8g8_uint,
        .r8g8_sint,
        .r16_uint,
        .r16_sint,
        .r16_float,
        .depth16_unorm,
        => 2,
        .r8g8b8a8_unorm,
        .r8g8b8a8_snorm,
        .r8g8b8a8_uint,
        .r8g8b8a8_sint,
        .r8g8b8a8_srgb,
        .b8g8r8a8_unorm,
        .b8g8r8a8_srgb,
        .r16g16_uint,
        .r16g16_sint,
        .r16g16_float,
        .r32_uint,
        .r32_sint,
        .r32_float,
        .r10g10b10a2_unorm,
        .b10g10r10a2_unorm,
        .r11g11b10_float,
        .depth24_plus_stencil8,
        .depth32_float,
        => 4,
        .r16g16b16a16_uint,
        .r16g16b16a16_sint,
        .r16g16b16a16_float,
        .r32g32_uint,
        .r32g32_sint,
        .r32g32_float,
        => 8,
        .r32g32b32_uint, .r32g32b32_sint, .r32g32b32_float => 12,
        .r32g32b32a32_uint,
        .r32g32b32a32_sint,
        .r32g32b32a32_float,
        => 16,
    };
}

/// Converts an RHI texture kind to a Vulkan image type.
///
/// # Args
/// - `kind`: dimensionality of the texture.
///
/// # Return
/// The matching `vk.ImageType`.
pub fn imageTypeToVk(kind: Texture.Kind) vk.ImageType {
    return switch (kind) {
        .d1 => .@"1d",
        .d2 => .@"2d",
        .d3 => .@"3d",
    };
}

/// Chooses the image view type of a texture.
///
/// # Args
/// - `desc`: texture description.
///
/// # Return
/// 1D, 2D, 2D array or 3D, depending on kind and
/// layer count.
pub fn imageViewTypeToVk(desc: Texture.Desc) vk.ImageViewType {
    return switch (desc.kind) {
        .d1 => if (desc.layers > 1) .@"1d_array" else .@"1d",
        .d3 => .@"3d",
        .d2 => if (desc.layers > 1) .@"2d_array" else .@"2d",
    };
}

/// Converts a sample count to a Vulkan sample mask.
///
/// # Args
/// - `samples`: samples per texel.
///
/// # Return
/// A mask with exactly one sample-count bit.
pub fn sampleCountToVk(samples: Texture.SampleCount) vk.SampleCountFlags {
    return switch (samples) {
        .x1 => .{ .@"1" = true },
        .x2 => .{ .@"2" = true },
        .x4 => .{ .@"4" = true },
        .x8 => .{ .@"8" = true },
        .x16 => .{ .@"16" = true },
    };
}

/// Converts RHI texture usage to Vulkan image usage.
///
/// # Args
/// - `usage`: requested texture operations.
///
/// # Return
/// The matching `vk.ImageUsageFlags`.
pub fn textureUsageToVk(usage: Texture.Usage) vk.ImageUsageFlags {
    return .{
        .sampled = usage.sampled,
        .storage = usage.storage,
        .color_attachment = usage.color_attachment,
        .depth_stencil_attachment = usage.depth_stencil_attachment,
        .transfer_src = usage.transfer_src,
        .transfer_dst = usage.transfer_dst,
    };
}

/// Converts RHI buffer usage to Vulkan buffer usage.
///
/// # Args
/// - `usage`: requested buffer operations.
///
/// # Return
/// The matching `vk.BufferUsageFlags`.
pub fn bufferUsageToVk(usage: Buffer.Usage) vk.BufferUsageFlags {
    return .{
        .vertex_buffer = usage.vertex,
        .index_buffer = usage.index,
        .uniform_buffer = usage.uniform,
        .storage_buffer = usage.storage,
        .indirect_buffer = usage.indirect,
        .transfer_src = usage.transfer_src,
        .transfer_dst = usage.transfer_dst,
    };
}

/// Reports whether a buffer usage asks for nothing.
///
/// # Args
/// - `usage`: usage set to test.
///
/// # Return
/// True when every flag is clear.
pub fn bufferUsageEmpty(usage: Buffer.Usage) bool {
    return !usage.vertex and !usage.index and !usage.uniform and
        !usage.storage and !usage.indirect and
        !usage.transfer_src and !usage.transfer_dst;
}

/// Reports whether a texture usage asks for nothing.
///
/// # Args
/// - `usage`: usage set to test.
///
/// # Return
/// True when every flag is clear.
pub fn textureUsageEmpty(usage: Texture.Usage) bool {
    return !usage.sampled and !usage.storage and
        !usage.color_attachment and
        !usage.depth_stencil_attachment and
        !usage.transfer_src and !usage.transfer_dst;
}

/// Converts an RHI texture layout to a Vulkan layout.
///
/// # Args
/// - `layout`: layout to convert.
///
/// # Return
/// The matching `vk.ImageLayout`.
pub fn layoutToVk(layout: Texture.Layout) vk.ImageLayout {
    return switch (layout) {
        .undefined => .undefined,
        .general => .general,
        .color_attachment => .color_attachment_optimal,
        .depth_stencil_attachment => .depth_stencil_attachment_optimal,
        .shader_read => .shader_read_only_optimal,
        .transfer_src => .transfer_src_optimal,
        .transfer_dst => .transfer_dst_optimal,
        .present => .present_src_khr,
    };
}

/// Converts RHI shader stages to a Vulkan mask.
///
/// # Args
/// - `stages`: stages to convert.
///
/// # Return
/// The matching `vk.ShaderStageFlags`.
pub fn stageFlagsToVk(
    stages: Shader.StageFlags,
) vk.ShaderStageFlags {
    return .{
        .vertex = stages.vertex,
        .fragment = stages.fragment,
        .compute = stages.compute,
    };
}

/// Reports whether a stage mask asks for nothing.
///
/// # Args
/// - `stages`: stages to test.
///
/// # Return
/// True when every flag is clear.
pub fn stageFlagsEmpty(stages: Shader.StageFlags) bool {
    return !stages.vertex and !stages.fragment and !stages.compute;
}

/// Converts one RHI shader stage to a Vulkan mask.
///
/// # Args
/// - `stage`: stage to convert.
///
/// # Return
/// A mask with exactly the matching stage bit.
pub fn shaderStageToVk(stage: Shader.Stage) vk.ShaderStageFlags {
    return switch (stage) {
        .vertex => .{ .vertex = true },
        .fragment => .{ .fragment = true },
        .compute => .{ .compute = true },
    };
}

/// Converts a compare operation.
///
/// # Args
/// - `op`: RHI comparison.
///
/// # Return
/// The matching `vk.CompareOp`.
pub fn compareOpToVk(op: Resource.CompareOp) vk.CompareOp {
    return switch (op) {
        .never => .never,
        .less => .less,
        .equal => .equal,
        .less_equal => .less_or_equal,
        .greater => .greater,
        .not_equal => .not_equal,
        .greater_equal => .greater_or_equal,
        .always => .always,
    };
}

/// Converts a stencil operation.
///
/// # Args
/// - `op`: RHI stencil operation.
///
/// # Return
/// The matching `vk.StencilOp`.
pub fn stencilOpToVk(
    op: PipelineGraphics.StencilOp,
) vk.StencilOp {
    return switch (op) {
        .keep => .keep,
        .zero => .zero,
        .replace => .replace,
        .increment_clamp => .increment_and_clamp,
        .decrement_clamp => .decrement_and_clamp,
        .invert => .invert,
        .increment_wrap => .increment_and_wrap,
        .decrement_wrap => .decrement_and_wrap,
    };
}

/// Converts a blend factor.
///
/// # Args
/// - `factor`: RHI blend factor.
///
/// # Return
/// The matching `vk.BlendFactor`.
pub fn blendFactorToVk(
    factor: PipelineGraphics.BlendFactor,
) vk.BlendFactor {
    return switch (factor) {
        .zero => .zero,
        .one => .one,
        .src_color => .src_color,
        .one_minus_src_color => .one_minus_src_color,
        .dst_color => .dst_color,
        .one_minus_dst_color => .one_minus_dst_color,
        .src_alpha => .src_alpha,
        .one_minus_src_alpha => .one_minus_src_alpha,
        .dst_alpha => .dst_alpha,
        .one_minus_dst_alpha => .one_minus_dst_alpha,
        .constant => .constant_color,
        .one_minus_constant => .one_minus_constant_color,
        .src_alpha_saturate => .src_alpha_saturate,
    };
}

/// Converts a blend equation.
///
/// # Args
/// - `op`: RHI blend operation.
///
/// # Return
/// The matching `vk.BlendOp`.
pub fn blendOpToVk(op: PipelineGraphics.BlendOp) vk.BlendOp {
    return switch (op) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .reverse_subtract,
        .min => .min,
        .max => .max,
    };
}

/// Converts a primitive topology.
///
/// # Args
/// - `topology`: RHI topology.
///
/// # Return
/// The matching `vk.PrimitiveTopology`.
pub fn topologyToVk(
    topology: PipelineGraphics.Topology,
) vk.PrimitiveTopology {
    return switch (topology) {
        .point_list => .point_list,
        .line_strip => .line_strip,
        .line_list => .line_list,
        .triangle_strip => .triangle_strip,
        .triangle_list => .triangle_list,
        .triangle_fan => .triangle_fan,
    };
}

/// Converts a front-face winding rule.
///
/// # Args
/// - `face`: RHI winding rule.
///
/// # Return
/// The matching `vk.FrontFace`.
pub fn frontFaceToVk(
    face: PipelineGraphics.FrontFace,
) vk.FrontFace {
    return switch (face) {
        .counter_clockwise => .counter_clockwise,
        .clockwise => .clockwise,
    };
}

/// Converts a cull mode.
///
/// # Args
/// - `cull`: RHI cull mode.
///
/// # Return
/// The matching `vk.CullModeFlags`.
pub fn cullToVk(cull: PipelineGraphics.Cull) vk.CullModeFlags {
    return switch (cull) {
        .none => .{},
        .front => .{ .front = true },
        .back => .{ .back = true },
    };
}

/// Converts a polygon mode.
///
/// # Args
/// - `polygon`: RHI polygon mode.
///
/// # Return
/// The matching `vk.PolygonMode`.
pub fn polygonToVk(
    polygon: PipelineGraphics.Polygon,
) vk.PolygonMode {
    return switch (polygon) {
        .fill => .fill,
        .line => .line,
        .point => .point,
    };
}

/// Converts a vertex attribute format.
///
/// # Args
/// - `format`: RHI vertex format.
///
/// # Return
/// The matching `vk.Format`.
pub fn vertexFormatToVk(
    format: PipelineGraphics.VertexFormat,
) vk.Format {
    return switch (format) {
        .f32 => .r32_sfloat,
        .f32x2 => .r32g32_sfloat,
        .f32x3 => .r32g32b32_sfloat,
        .f32x4 => .r32g32b32a32_sfloat,
        .u8x2 => .r8g8_uint,
        .u8x4 => .r8g8b8a8_uint,
        .u8x4_unorm => .r8g8b8a8_unorm,
        .i8x2 => .r8g8_sint,
        .i8x4 => .r8g8b8a8_sint,
        .u16x2 => .r16g16_uint,
        .u16x4 => .r16g16b16a16_uint,
        .u16x2_unorm => .r16g16_unorm,
        .u16x4_unorm => .r16g16b16a16_unorm,
        .i16x2 => .r16g16_sint,
        .i16x4 => .r16g16b16a16_sint,
        .u32 => .r32_uint,
        .u32x2 => .r32g32_uint,
        .u32x3 => .r32g32b32_uint,
        .u32x4 => .r32g32b32a32_uint,
    };
}

/// Converts a sampler filter.
///
/// # Args
/// - `filter`: RHI filter.
///
/// # Return
/// The matching `vk.Filter`.
pub fn filterToVk(filter: Sampler.Filter) vk.Filter {
    return switch (filter) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

/// Converts a sampler wrap mode.
///
/// # Args
/// - `wrap`: RHI wrap mode.
///
/// # Return
/// The matching `vk.SamplerAddressMode`.
pub fn wrapToVk(wrap: Sampler.Wrap) vk.SamplerAddressMode {
    return switch (wrap) {
        .repeat => .repeat,
        .mirror_repeat => .mirrored_repeat,
        .clamp_to_edge => .clamp_to_edge,
        .clamp_to_border => .clamp_to_border,
        .mirror_clamp_to_edge => .mirror_clamp_to_edge,
    };
}

/// Converts a presentation mode.
///
/// # Args
/// - `mode`: RHI present mode.
///
/// # Return
/// The matching `vk.PresentModeKHR`.
pub fn presentModeToVk(
    mode: Swapchain.PresentMode,
) vk.PresentModeKHR {
    return switch (mode) {
        .fifo => .fifo_khr,
        .fifo_relaxed => .fifo_relaxed_khr,
        .mailbox => .mailbox_khr,
        .immediate => .immediate_khr,
    };
}

/// Converts a descriptor binding kind.
///
/// # Args
/// - `kind`: RHI descriptor kind.
///
/// # Return
/// The matching `vk.DescriptorType`, or null when
/// the kind cannot be represented.
pub fn descriptorTypeToVk(
    kind: DescriptorSetLayout.Kind,
) ?vk.DescriptorType {
    return switch (kind) {
        .uniform_buffer => .uniform_buffer,
        .storage_buffer => .storage_buffer,
        .sampled_texture => .sampled_image,
        .storage_texture => .storage_image,
        .sampler => .sampler,
        .combined_sampler => .combined_image_sampler,
    };
}

/// Converts barrier stages to a Vulkan stage mask.
///
/// # Args
/// - `stages`: RHI stage scope.
///
/// # Return
/// The matching pipeline stage mask. An empty
/// scope maps to `all_commands`.
pub fn stagesToVk(stages: Device.Command.Stage) vk.PipelineStageFlags {
    if (stages.all) return .{ .all_commands = true };
    var out = vk.PipelineStageFlags{};
    out.vertex_shader = stages.vertex;
    out.fragment_shader = stages.fragment;
    out.compute_shader = stages.compute;
    out.transfer = stages.transfer;
    out.host = stages.host;
    out.color_attachment_output = stages.color_attachment;
    if (stages.depth_stencil) {
        out.early_fragment_tests = true;
        out.late_fragment_tests = true;
    }
    return out;
}

/// Converts barrier accesses to a Vulkan access mask.
///
/// # Args
/// - `access`: RHI access scope.
///
/// # Return
/// The matching memory access mask.
pub fn accessToVk(access: Device.Command.Access) vk.AccessFlags {
    if (access.all) {
        return .{ .memory_read = true, .memory_write = true };
    }
    var out = vk.AccessFlags{};
    out.uniform_read = access.uniform_read;
    out.shader_read = access.storage_read;
    out.shader_write = access.storage_write;
    out.vertex_attribute_read = access.vertex_read;
    out.index_read = access.index_read;
    out.indirect_command_read = access.indirect_read;
    out.transfer_read = access.transfer_read;
    out.transfer_write = access.transfer_write;
    out.color_attachment_read = access.color_read;
    out.color_attachment_write = access.color_write;
    out.depth_stencil_attachment_read = access.depth_read;
    out.depth_stencil_attachment_write = access.depth_write;
    out.host_read = access.host_read;
    out.host_write = access.host_write;
    return out;
}

/// Builds a subresource range covering a whole texture.
///
/// # Args
/// - `desc`: texture description.
///
/// # Return
/// A range for every mip level, layer and plane.
pub fn fullSubresourceRange(
    desc: Texture.Desc,
) vk.ImageSubresourceRange {
    return .{
        .aspect_mask = formatAspect(desc.format),
        .base_mip_level = 0,
        .level_count = @intCast(desc.mips),
        .base_array_layer = 0,
        .layer_count = @intCast(desc.layers),
    };
}

/// Converts an RHI subresource range.
///
/// # Args
/// - `desc`: texture description.
/// - `range`: RHI range.
///
/// # Return
/// The matching Vulkan range. Zero counts expand
/// to the rest of the texture, and an empty aspect
/// mask selects every plane of the format.
pub fn subresourceRangeToVk(
    desc: Texture.Desc,
    range: Texture.SubresourceRange,
) vk.ImageSubresourceRange {
    const aspect: vk.ImageAspectFlags = if (range.aspects.color or
        range.aspects.depth or range.aspects.stencil)
        .{
            .color = range.aspects.color,
            .depth = range.aspects.depth,
            .stencil = range.aspects.stencil,
        }
    else
        formatAspect(desc.format);

    return .{
        .aspect_mask = aspect,
        .base_mip_level = range.mip,
        .level_count = if (range.mip_count == 0)
            vk.REMAINING_MIP_LEVELS
        else
            range.mip_count,
        .base_array_layer = range.layer,
        .layer_count = if (range.layer_count == 0)
            vk.REMAINING_ARRAY_LAYERS
        else
            range.layer_count,
    };
}
