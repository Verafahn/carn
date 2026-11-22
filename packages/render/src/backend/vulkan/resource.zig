//! Resource creation, destruction and host mapping.
//!
//! Every function here is called through the device
//! vtable in `root.zig`. Descriptions are validated
//! before native objects are created, and partially
//! created objects are released on failure.

const std = @import("std");
const builtin = @import("builtin");
const rhi = @import("rhi");
const vk = @import("vk");

const root = @import("root.zig");
const format = @import("format.zig");
const state = @import("state.zig");

const Backend = root.Backend;
const Device = rhi.Device;
const Resource = rhi.Resource;

const Win32 = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn GetModuleHandleW(
        module_name: ?[*:0]const u16,
    ) callconv(.winapi) ?std.os.windows.HMODULE;
} else struct {};
const Texture = Resource.Texture;
const Buffer = Resource.Buffer;
const Sampler = Resource.Sampler;
const Shader = Resource.Shader;
const Surface = Resource.Surface;
const Swapchain = Resource.Swapchain;
const DescriptorSetLayout = Resource.DescriptorSetLayout;
const DescriptorSet = Resource.DescriptorSet;
const PipelineLayout = Resource.PipelineLayout;
const PipelineGraphics = Resource.PipelineGraphics;
const PipelineCompute = Resource.PipelineCompute;
const CommandPool = Resource.CommandPool;
const CommandBuffer = Resource.CommandBuffer;
const Fence = Resource.Fence;
const Semaphore = Resource.Semaphore;

/// Creates a resource selected by `desc`.
///
/// # Args
/// - `self`: owning backend.
/// - `desc`: description of the resource.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A live handle whose kind matches `desc`.
///
/// # Error
/// - `error.InvalidDesc`: the description is rejected.
/// - `error.InvalidResource`: a referenced resource is
///   not live.
/// - `error.OutOfDeviceMemory`: device memory is
///   exhausted.
/// - `error.Unsupport`: the device cannot create this
///   resource.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub fn create(
    self: *Backend,
    desc: *const Resource.Desc,
    ret_addr: usize,
) Device.CreateError!Resource {
    _ = ret_addr;
    return switch (desc.*) {
        .buffer => |d| createBuffer(self, d),
        .command_buffer => |d| createCommandBuffer(self, d),
        .command_pool => |d| createCommandPool(self, d),
        .descriptor_set => |d| createDescriptorSet(self, d),
        .descriptor_set_layout => |d| createDescriptorSetLayout(self, d),
        .fence => |d| createFence(self, d),
        .pipeline_compute => |d| createPipelineCompute(self, d),
        .pipeline_graphics => |d| createPipelineGraphics(self, d),
        .pipeline_layout => |d| createPipelineLayout(self, d),
        .sampler => |d| createSampler(self, d),
        .semaphore => |d| createSemaphore(self, d),
        .shader => |d| createShader(self, d),
        .surface => |d| createSurface(self, d),
        .swapchain => |d| createSwapchain(self, d),
        .texture => |d| createTexture(self, d),
    };
}

/// Destroys a resource and releases native state.
///
/// # Args
/// - `self`: owning backend.
/// - `handle`: handle to destroy.
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// Invalid handles, double destroys, swapchain-owned
/// textures and surfaces still used by a swapchain are
/// reported as contract violations.
pub fn destroy(
    self: *Backend,
    handle: Resource,
    ret_addr: usize,
) void {
    _ = ret_addr;
    const entry = self.lookup(handle) orelse {
        self.report(
            "destroy of invalid {s} handle",
            .{@tagName(handle.kind)},
        );
        return;
    };

    if (entry.kind == .texture and entry.data.texture.owner != null) {
        self.report("destroy of a swapchain-owned texture", .{});
        return;
    }
    if (entry.kind == .surface and surfaceInUse(self, handle)) {
        self.report("destroy of a surface used by a swapchain", .{});
        return;
    }
    if (entry.kind == .command_buffer) {
        removeChild(self, handle);
    }

    destroyEntryAt(self, root.entryIndex(handle));
}

/// Maps a host-visible buffer into host memory.
///
/// # Args
/// - `self`: owning backend.
/// - `handle`: buffer handle.
/// - `offset`: first byte of the requested range.
/// - `size`: byte length of the requested range.
/// - `flags`: requested host access.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A slice covering `offset..offset + size`.
///
/// # Error
/// - `error.InvalidDesc`: the range is empty.
/// - `error.InvalidResource`: the handle is not a live
///   buffer.
/// - `error.NotHostVisible`: the buffer uses device
///   memory.
/// - `error.AlreadyMapped`: the buffer is mapped.
/// - `error.OutOfRange`: the range leaves the buffer.
pub fn map(
    self: *Backend,
    handle: Resource,
    offset: u64,
    size: u64,
    flags: Device.MapFlags,
    ret_addr: usize,
) Device.MapError![]u8 {
    _ = ret_addr;
    const entry = self.lookup(handle) orelse return error.InvalidResource;
    if (entry.kind != .buffer) return error.InvalidResource;
    const buffer = &entry.data.buffer;

    if (buffer.domain == .device) return error.NotHostVisible;
    if (buffer.mapped) return error.AlreadyMapped;
    if (size == 0) return error.InvalidDesc;
    if (!root.fits(offset, size, buffer.size)) return error.OutOfRange;

    const pointer = self.device_api.mapMemory(
        self.vk_device,
        buffer.memory,
        0,
        vk.WHOLE_SIZE,
        .{},
    ) catch |err| return mapError(err);
    const base: [*]u8 = @ptrCast(pointer orelse {
        self.device_api.unmapMemory(self.vk_device, buffer.memory);
        return error.NotHostVisible;
    });

    buffer.mapped = true;
    buffer.map_ptr = base;
    buffer.map_offset = offset;
    buffer.map_size = size;

    if (flags.read and !buffer.memory_flags.host_coherent) {
        invalidateRange(self, buffer, offset, size);
    }

    const start: usize = @intCast(offset);
    const end: usize = @intCast(offset + size);
    return base[start..end];
}

/// Releases a mapping created by `map`.
///
/// # Args
/// - `self`: owning backend.
/// - `handle`: mapped buffer.
/// - `dirty`: byte ranges written by the host.
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// Non-coherent memory is flushed before the mapping is
/// released.
pub fn unmap(
    self: *Backend,
    handle: Resource,
    dirty: []const Device.DirtyRange,
    ret_addr: usize,
) void {
    _ = ret_addr;
    const entry = self.lookup(handle) orelse {
        self.report("unmap of invalid buffer handle", .{});
        return;
    };
    if (entry.kind != .buffer) {
        self.report("unmap of non-buffer handle", .{});
        return;
    }
    const buffer = &entry.data.buffer;
    if (!buffer.mapped) {
        self.report("unmap of buffer that is not mapped", .{});
        return;
    }

    if (!buffer.memory_flags.host_coherent) {
        flushRanges(self, buffer, dirty);
    }

    self.device_api.unmapMemory(self.vk_device, buffer.memory);
    buffer.mapped = false;
    buffer.map_ptr = null;
    buffer.map_offset = 0;
    buffer.map_size = 0;
}

fn invalidateRange(
    self: *Backend,
    buffer: *state.BufferState,
    offset: u64,
    size: u64,
) void {
    const atom = @max(
        self.properties.limits.non_coherent_atom_size,
        1,
    );
    const start = offset - (offset % atom);
    const end = @min(alignUp(offset + size, atom), buffer.size);
    const range = vk.MappedMemoryRange{
        .memory = buffer.memory,
        .offset = start,
        .size = end - start,
    };
    self.device_api.invalidateMappedMemoryRanges(
        self.vk_device,
        &.{range},
    ) catch {};
}

fn flushRanges(
    self: *Backend,
    buffer: *state.BufferState,
    dirty: []const Device.DirtyRange,
) void {
    const atom = @max(
        self.properties.limits.non_coherent_atom_size,
        1,
    );

    if (dirty.len == 0) {
        const range = vk.MappedMemoryRange{
            .memory = buffer.memory,
            .offset = 0,
            .size = vk.WHOLE_SIZE,
        };
        self.device_api.flushMappedMemoryRanges(
            self.vk_device,
            &.{range},
        ) catch {};
        return;
    }

    for (dirty) |item| {
        if (item.size == 0 or !root.fits(item.offset, item.size, buffer.size)) {
            self.report("unmap dirty range is out of bounds", .{});
            continue;
        }
        const start = item.offset - (item.offset % atom);
        const end = @min(
            alignUp(item.offset + item.size, atom),
            buffer.size,
        );
        const range = vk.MappedMemoryRange{
            .memory = buffer.memory,
            .offset = start,
            .size = end - start,
        };
        self.device_api.flushMappedMemoryRanges(
            self.vk_device,
            &.{range},
        ) catch {};
    }
}

fn alignUp(value: u64, alignment: u64) u64 {
    const mask = alignment - 1;
    return (value + mask) & ~mask;
}

/// Releases every live resource.
///
/// # Args
/// - `self`: owning backend.
///
/// # Note
/// Children of a swapchain or command pool are released
/// with their parent. The registry is deinitialized.
pub fn releaseAll(self: *Backend) void {
    var i = self.entries.items.len;
    while (i > 0) {
        i -= 1;
        const entry = &self.entries.items[i];
        if (!entry.live) continue;
        if (entry.kind == .texture and entry.data.texture.owner != null) {
            continue;
        }
        destroyEntryAt(self, @intCast(i));
    }
    self.entries.deinit(self.allocator);
}

/// Releases native state of one registry entry.
///
/// # Args
/// - `self`: owning backend.
/// - `index`: registry slot.
///
/// # Note
/// The slot is recycled after its native objects are
/// destroyed. Child resources owned by the entry are
/// recycled with it.
pub fn destroyEntryAt(
    self: *Backend,
    index: u32,
) void {
    if (@as(usize, index) >= self.entries.items.len) return;
    const entry = &self.entries.items[index];
    if (!entry.live) return;

    switch (entry.data) {
        .buffer => |*buffer| {
            if (buffer.mapped) {
                self.device_api.unmapMemory(self.vk_device, buffer.memory);
                buffer.mapped = false;
            }
            self.device_api.destroyBuffer(
                self.vk_device,
                buffer.buffer,
                null,
            );
            self.device_api.freeMemory(
                self.vk_device,
                buffer.memory,
                null,
            );
        },
        .command_buffer => |*buffer| {
            releaseCommandBuffer(self, buffer);
        },
        .command_pool => |*pool| {
            self.device_api.destroyCommandPool(
                self.vk_device,
                pool.pool,
                null,
            );
            for (pool.buffers.items) |handle| {
                const child = self.lookup(handle) orelse continue;
                if (child.kind != .command_buffer) continue;
                releaseCommandBufferHost(self, &child.data.command_buffer);
                self.recycleEntry(root.entryIndex(handle));
            }
            pool.buffers.deinit(self.allocator);
        },
        .descriptor_set => |*set| {
            self.device_api.destroyDescriptorPool(
                self.vk_device,
                set.pool,
                null,
            );
        },
        .descriptor_set_layout => |*layout| {
            self.device_api.destroyDescriptorSetLayout(
                self.vk_device,
                layout.layout,
                null,
            );
            self.allocator.free(layout.bindings);
        },
        .fence => |*fence| {
            self.device_api.destroyFence(
                self.vk_device,
                fence.fence,
                null,
            );
        },
        .pipeline_compute => |*pipeline| {
            self.device_api.destroyPipeline(
                self.vk_device,
                pipeline.pipeline,
                null,
            );
        },
        .pipeline_graphics => |*pipeline| {
            self.device_api.destroyPipeline(
                self.vk_device,
                pipeline.pipeline,
                null,
            );
        },
        .pipeline_layout => |*layout| {
            self.device_api.destroyPipelineLayout(
                self.vk_device,
                layout.layout,
                null,
            );
            self.allocator.free(layout.sets);
            self.allocator.free(layout.push_constants);
        },
        .sampler => |*sampler| {
            self.device_api.destroySampler(
                self.vk_device,
                sampler.sampler,
                null,
            );
        },
        .semaphore => |*semaphore| {
            self.device_api.destroySemaphore(
                self.vk_device,
                semaphore.semaphore,
                null,
            );
        },
        .shader => |*shader| {
            self.device_api.destroyShaderModule(
                self.vk_device,
                shader.module,
                null,
            );
            self.allocator.free(shader.entry);
        },
        .surface => |*surface| {
            self.instance_api.destroySurfaceKHR(
                self.instance,
                surface.surface,
                null,
            );
        },
        .swapchain => |*chain| {
            for (chain.images) |handle| {
                const child = self.lookup(handle) orelse continue;
                if (child.kind != .texture) continue;
                self.device_api.destroyImageView(
                    self.vk_device,
                    child.data.texture.view,
                    null,
                );
                self.recycleEntry(root.entryIndex(handle));
            }
            self.device_api.destroySwapchainKHR(
                self.vk_device,
                chain.swapchain,
                null,
            );
            self.allocator.free(chain.images);
            self.allocator.free(chain.acquired);
        },
        .texture => |*texture| {
            self.device_api.destroyImageView(
                self.vk_device,
                texture.view,
                null,
            );
            if (texture.stencil_view) |stencil_view| {
                self.device_api.destroyImageView(
                    self.vk_device,
                    stencil_view,
                    null,
                );
            }
            if (texture.owner == null) {
                self.device_api.destroyImage(
                    self.vk_device,
                    texture.image,
                    null,
                );
                if (texture.memory) |memory| {
                    self.device_api.freeMemory(
                        self.vk_device,
                        memory,
                        null,
                    );
                }
            }
        },
    }

    self.recycleEntry(index);
}

fn releaseCommandBuffer(
    self: *Backend,
    buffer: *state.CommandBufferState,
) void {
    const pool = self.get(buffer.pool, .command_pool);
    if (pool) |native| {
        self.device_api.freeCommandBuffers(
            self.vk_device,
            native.pool,
            &.{buffer.native},
        );
    }
    releaseCommandBufferHost(self, buffer);
}

fn releaseCommandBufferHost(
    self: *Backend,
    buffer: *state.CommandBufferState,
) void {
    buffer.stored.deinit(self.allocator);
    buffer.arena.deinit();
}

fn removeChild(self: *Backend, handle: Resource) void {
    const entry = self.lookup(handle) orelse return;
    if (entry.kind != .command_buffer) return;
    const pool_handle = entry.data.command_buffer.pool;
    const pool = self.get(pool_handle, .command_pool) orelse return;
    for (pool.buffers.items, 0..) |child, i| {
        if (root.resourceEql(child, handle)) {
            _ = pool.buffers.swapRemove(i);
            return;
        }
    }
}

fn surfaceInUse(self: *Backend, handle: Resource) bool {
    for (self.entries.items) |entry| {
        if (!entry.live) continue;
        if (entry.kind != .swapchain) continue;
        if (root.resourceEql(entry.data.swapchain.surface, handle)) {
            return true;
        }
    }
    return false;
}

// -------------------------------------------------------------------
// Buffers and textures
// -------------------------------------------------------------------

fn createBuffer(
    self: *Backend,
    desc: Buffer.Desc,
) Device.CreateError!Resource {
    if (desc.size == 0) return error.InvalidDesc;
    if (format.bufferUsageEmpty(desc.usage)) return error.InvalidDesc;

    const native = self.device_api.createBuffer(
        self.vk_device,
        &.{
            .size = desc.size,
            .usage = format.bufferUsageToVk(desc.usage),
            .sharing_mode = .exclusive,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyBuffer(self.vk_device, native, null);

    const requirements = self.device_api.getBufferMemoryRequirements(
        self.vk_device,
        native,
    );

    const required: vk.MemoryPropertyFlags = switch (desc.memory) {
        .device => .{ .device_local = true },
        .host => .{ .host_visible = true },
        .unified => .{ .host_visible = true, .device_local = true },
    };
    const preferred: vk.MemoryPropertyFlags = switch (desc.memory) {
        .device => .{ .device_local = true },
        .host => .{ .host_visible = true, .host_coherent = true },
        .unified => .{
            .host_visible = true,
            .host_coherent = true,
            .device_local = true,
        },
    };

    const allocation = try allocateMemory(
        self,
        requirements,
        required,
        preferred,
    );
    errdefer self.device_api.freeMemory(
        self.vk_device,
        allocation.memory,
        null,
    );

    self.device_api.bindBufferMemory(
        self.vk_device,
        native,
        allocation.memory,
        0,
    ) catch |err| return createError(err);

    return self.newEntry(
        .buffer,
        .{ .buffer = .{
            .buffer = native,
            .memory = allocation.memory,
            .size = desc.size,
            .usage = desc.usage,
            .domain = desc.memory,
            .memory_flags = allocation.flags,
        } },
        desc.label,
    );
}

fn createTexture(
    self: *Backend,
    desc: Texture.Desc,
) Device.CreateError!Resource {
    if (!textureSupported(self, desc)) return error.InvalidDesc;

    const native_format = format.formatToVk(desc.format) orelse
        return error.InvalidDesc;

    const image = self.device_api.createImage(
        self.vk_device,
        &.{
            .image_type = format.imageTypeToVk(desc.kind),
            .format = native_format,
            .extent = .{
                .width = desc.extent.width,
                .height = desc.extent.height,
                .depth = desc.extent.depth,
            },
            .mip_levels = desc.mips,
            .array_layers = desc.layers,
            .samples = format.sampleCountToVk(desc.samples),
            .tiling = .optimal,
            .usage = format.textureUsageToVk(desc.usage),
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyImage(self.vk_device, image, null);

    const requirements = self.device_api.getImageMemoryRequirements(
        self.vk_device,
        image,
    );
    const allocation = try allocateMemory(
        self,
        requirements,
        .{ .device_local = true },
        .{ .device_local = true },
    );
    errdefer self.device_api.freeMemory(
        self.vk_device,
        allocation.memory,
        null,
    );

    self.device_api.bindImageMemory(
        self.vk_device,
        image,
        allocation.memory,
        0,
    ) catch |err| return createError(err);

    const view = try createImageView(
        self,
        image,
        desc,
        format.primaryAspect(desc.format),
    );
    errdefer self.device_api.destroyImageView(self.vk_device, view, null);

    var stencil_view: ?vk.ImageView = null;
    if (format.formatAspect(desc.format).depth and
        format.formatAspect(desc.format).stencil)
    {
        stencil_view = try createImageView(
            self,
            image,
            desc,
            .{ .stencil = true },
        );
    }
    errdefer if (stencil_view) |view_to_destroy| {
        self.device_api.destroyImageView(
            self.vk_device,
            view_to_destroy,
            null,
        );
    };

    var stored_desc = desc;
    stored_desc.label = "";

    return self.newEntry(
        .texture,
        .{ .texture = .{
            .image = image,
            .memory = allocation.memory,
            .view = view,
            .stencil_view = stencil_view,
            .desc = stored_desc,
        } },
        desc.label,
    );
}

fn createImageView(
    self: *Backend,
    image: vk.Image,
    desc: Texture.Desc,
    aspect: vk.ImageAspectFlags,
) Device.CreateError!vk.ImageView {
    const native_format = format.formatToVk(desc.format) orelse
        return error.InvalidDesc;

    return self.device_api.createImageView(
        self.vk_device,
        &.{
            .image = image,
            .view_type = format.imageViewTypeToVk(desc),
            .format = native_format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = aspect,
                .base_mip_level = 0,
                .level_count = @intCast(desc.mips),
                .base_array_layer = 0,
                .layer_count = @intCast(desc.layers),
            },
        },
        null,
    ) catch |err| return createError(err);
}

/// Reports whether a texture description is supported.
///
/// # Args
/// - `self`: owning backend.
/// - `desc`: texture description.
///
/// # Return
/// True when the format, usage, extent, mip count,
/// layer count and sample count are accepted.
pub fn textureSupported(
    self: *Backend,
    desc: Texture.Desc,
) bool {
    if (desc.extent.width == 0) return false;
    if (desc.extent.height == 0 or desc.extent.depth == 0) return false;
    if (desc.mips == 0 or desc.layers == 0) return false;
    if (format.textureUsageEmpty(desc.usage)) return false;

    switch (desc.kind) {
        .d1 => if (desc.extent.height != 1 or desc.extent.depth != 1) {
            return false;
        },
        .d2 => if (desc.extent.depth != 1) return false,
        .d3 => if (desc.layers != 1) return false,
    }

    if (!formatSupportsUsage(self, desc.format, desc.usage)) {
        return false;
    }

    const native_format = format.formatToVk(desc.format) orelse
        return false;

    const max_extent: u32 = switch (desc.kind) {
        .d1 => self.limits.max_texture_extent_1d,
        .d2 => self.limits.max_texture_extent_2d,
        .d3 => self.limits.max_texture_extent_3d,
    };
    if (desc.extent.width > max_extent) return false;
    if (desc.extent.height > max_extent) return false;
    if (desc.extent.depth > max_extent) return false;
    if (desc.layers > self.limits.max_texture_layers) return false;

    const largest = @max(
        desc.extent.width,
        @max(desc.extent.height, desc.extent.depth),
    );
    const max_mips = std.math.log2_int(u32, largest) + 1;
    if (desc.mips > max_mips) return false;

    const properties = self.instance_api.getPhysicalDeviceImageFormatProperties(
        self.physical_device,
        native_format,
        format.imageTypeToVk(desc.kind),
        .optimal,
        format.textureUsageToVk(desc.usage),
        .{},
    ) catch return false;

    if (desc.extent.width > properties.max_extent.width) return false;
    if (desc.extent.height > properties.max_extent.height) return false;
    if (desc.extent.depth > properties.max_extent.depth) return false;
    if (desc.mips > properties.max_mip_levels) return false;
    if (desc.layers > properties.max_array_layers) return false;

    return sampleCountSupported(properties.sample_counts, desc.samples);
}

fn sampleCountSupported(
    mask: vk.SampleCountFlags,
    samples: Texture.SampleCount,
) bool {
    return switch (samples) {
        .x1 => mask.@"1",
        .x2 => mask.@"2",
        .x4 => mask.@"4",
        .x8 => mask.@"8",
        .x16 => mask.@"16",
    };
}

/// Reports whether a format serves every requested usage.
///
/// # Args
/// - `self`: owning backend.
/// - `pixel_format`: RHI format to test.
/// - `usage`: requested usages.
///
/// # Return
/// True when the optimal tiling features contain every
/// required bit and the format kind fits the usage.
pub fn formatSupportsUsage(
    self: *Backend,
    pixel_format: Resource.Format,
    usage: Texture.Usage,
) bool {
    if (format.textureUsageEmpty(usage)) return false;

    const native_format = format.formatToVk(pixel_format) orelse
        return false;
    const properties = self.instance_api.getPhysicalDeviceFormatProperties(
        self.physical_device,
        native_format,
    );
    const features = properties.optimal_tiling_features;

    const depth = format.isDepthStencil(pixel_format);
    if (usage.color_attachment and depth) return false;
    if (usage.depth_stencil_attachment and !depth) return false;

    if (usage.sampled and !features.sampled_image) return false;
    if (usage.storage and !features.storage_image) return false;
    if (usage.color_attachment and !features.color_attachment) return false;
    if (usage.depth_stencil_attachment and
        !features.depth_stencil_attachment)
    {
        return false;
    }
    if (usage.transfer_src and !features.transfer_src) return false;
    if (usage.transfer_dst and !features.transfer_dst) return false;

    return true;
}

// -------------------------------------------------------------------
// Samplers and shaders
// -------------------------------------------------------------------

fn createSampler(
    self: *Backend,
    desc: Sampler.Desc,
) Device.CreateError!Resource {
    if (std.math.isNan(desc.lod_min) or
        std.math.isNan(desc.lod_max) or
        std.math.isNan(desc.max_anisotropy))
    {
        return error.InvalidDesc;
    }
    if (desc.lod_min < 0 or desc.lod_max < desc.lod_min) {
        return error.InvalidDesc;
    }
    if (desc.max_anisotropy < 1) return error.InvalidDesc;
    if (desc.max_anisotropy > 1 and
        self.features.sampler_anisotropy != .true)
    {
        return error.Unsupport;
    }

    const max_anisotropy = @min(
        desc.max_anisotropy,
        self.properties.limits.max_sampler_anisotropy,
    );

    const lod_min = @min(desc.lod_min, vk.LOD_CLAMP_NONE);
    const lod_max = if (std.math.isInf(desc.lod_max))
        vk.LOD_CLAMP_NONE
    else
        @min(desc.lod_max, vk.LOD_CLAMP_NONE);
    if (lod_max < lod_min) return error.InvalidDesc;

    const native = self.device_api.createSampler(
        self.vk_device,
        &.{
            .mag_filter = format.filterToVk(desc.mag_filter),
            .min_filter = format.filterToVk(desc.min_filter),
            .mipmap_mode = switch (desc.mip_filter) {
                .nearest => .nearest,
                .linear => .linear,
            },
            .address_mode_u = format.wrapToVk(desc.address.u),
            .address_mode_v = format.wrapToVk(desc.address.v),
            .address_mode_w = format.wrapToVk(desc.address.w),
            .mip_lod_bias = 0,
            .anisotropy_enable = vk.Bool32.fromBool(
                desc.max_anisotropy > 1,
            ),
            .max_anisotropy = max_anisotropy,
            .compare_enable = vk.Bool32.fromBool(desc.compare != null),
            .compare_op = if (desc.compare) |op|
                format.compareOpToVk(op)
            else
                .always,
            .min_lod = lod_min,
            .max_lod = lod_max,
            .border_color = .float_transparent_black,
            .unnormalized_coordinates = .false,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroySampler(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .sampler,
        .{ .sampler = .{ .sampler = native } },
        desc.label,
    );
}

fn createShader(
    self: *Backend,
    desc: Shader.Desc,
) Device.CreateError!Resource {
    if (desc.language != .spirv) return error.Unsupport;
    if (desc.entry.len == 0) return error.InvalidDesc;
    if (desc.code.len == 0 or desc.code.len % 4 != 0) {
        return error.InvalidDesc;
    }
    if (desc.code.len < @sizeOf(u32)) return error.InvalidDesc;
    const magic = std.mem.readInt(u32, desc.code[0..4], .little);
    if (magic != 0x07230203) return error.InvalidDesc;

    const words = try self.allocator.alloc(u32, desc.code.len / 4);
    defer self.allocator.free(words);
    @memcpy(std.mem.sliceAsBytes(words), desc.code);

    const module = self.device_api.createShaderModule(
        self.vk_device,
        &.{
            .code_size = std.mem.sliceAsBytes(words).len,
            .p_code = words.ptr,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyShaderModule(
        self.vk_device,
        module,
        null,
    );

    const entry = try self.allocator.allocSentinel(
        u8,
        desc.entry.len,
        0,
    );
    @memcpy(entry, desc.entry);
    errdefer self.allocator.free(entry);

    return self.newEntry(
        .shader,
        .{ .shader = .{
            .module = module,
            .stage = desc.stage,
            .entry = entry,
        } },
        desc.label,
    );
}

// -------------------------------------------------------------------
// Descriptors
// -------------------------------------------------------------------

fn createDescriptorSetLayout(
    self: *Backend,
    desc: DescriptorSetLayout.Desc,
) Device.CreateError!Resource {
    for (desc.bindings, 0..) |binding, i| {
        if (binding.count == 0) return error.InvalidDesc;
        if (format.stageFlagsEmpty(binding.stages)) {
            return error.InvalidDesc;
        }
        _ = format.descriptorTypeToVk(binding.kind) orelse
            return error.Unsupport;
        for (desc.bindings[0..i]) |previous| {
            if (previous.binding == binding.binding) {
                return error.InvalidDesc;
            }
        }
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const native_bindings = try scratch.alloc(
        vk.DescriptorSetLayoutBinding,
        desc.bindings.len,
    );
    for (desc.bindings, 0..) |binding, i| {
        native_bindings[i] = .{
            .binding = binding.binding,
            .descriptor_type = format.descriptorTypeToVk(binding.kind).?,
            .descriptor_count = binding.count,
            .stage_flags = format.stageFlagsToVk(binding.stages),
        };
    }

    const native = self.device_api.createDescriptorSetLayout(
        self.vk_device,
        &.{
            .binding_count = @intCast(native_bindings.len),
            .p_bindings = if (native_bindings.len == 0)
                null
            else
                native_bindings.ptr,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyDescriptorSetLayout(
        self.vk_device,
        native,
        null,
    );

    const bindings = try self.allocator.dupe(
        DescriptorSetLayout.Binding,
        desc.bindings,
    );
    errdefer self.allocator.free(bindings);

    return self.newEntry(
        .descriptor_set_layout,
        .{ .descriptor_set_layout = .{
            .layout = native,
            .bindings = bindings,
        } },
        desc.label,
    );
}

fn createDescriptorSet(
    self: *Backend,
    desc: DescriptorSet.Desc,
) Device.CreateError!Resource {
    const layout_entry = self.lookup(desc.layout) orelse
        return error.InvalidResource;
    if (layout_entry.kind != .descriptor_set_layout) {
        return error.InvalidResource;
    }
    const layout = &layout_entry.data.descriptor_set_layout;

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    var sizes: std.ArrayListUnmanaged(vk.DescriptorPoolSize) = .empty;
    for (layout.bindings) |binding| {
        const descriptor_type =
            format.descriptorTypeToVk(binding.kind) orelse
            return error.Unsupport;
        var merged = false;
        for (sizes.items) |*size| {
            if (size.type == descriptor_type) {
                size.descriptor_count += binding.count;
                merged = true;
                break;
            }
        }
        if (!merged) {
            try sizes.append(scratch, .{
                .type = descriptor_type,
                .descriptor_count = binding.count,
            });
        }
    }
    if (sizes.items.len == 0) {
        try sizes.append(scratch, .{
            .type = .uniform_buffer,
            .descriptor_count = 1,
        });
    }

    const pool = self.device_api.createDescriptorPool(
        self.vk_device,
        &.{
            .max_sets = 1,
            .pool_size_count = @intCast(sizes.items.len),
            .p_pool_sizes = sizes.items.ptr,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyDescriptorPool(
        self.vk_device,
        pool,
        null,
    );

    var native: vk.DescriptorSet = undefined;
    const layouts = [_]vk.DescriptorSetLayout{layout.layout};
    self.device_api.allocateDescriptorSets(
        self.vk_device,
        &.{
            .descriptor_pool = pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &layouts,
        },
        @ptrCast(&native),
    ) catch |err| return createError(err);

    return self.newEntry(
        .descriptor_set,
        .{ .descriptor_set = .{
            .set = native,
            .pool = pool,
            .layout = desc.layout.handle,
        } },
        desc.label,
    );
}

fn createPipelineLayout(
    self: *Backend,
    desc: PipelineLayout.Desc,
) Device.CreateError!Resource {
    if (desc.push_constants.len > 0) {
        for (desc.push_constants, 0..) |range, i| {
            if (range.size == 0) return error.InvalidDesc;
            if (format.stageFlagsEmpty(range.stages)) {
                return error.InvalidDesc;
            }
            const end = @as(u64, range.offset) + range.size;
            if (end > self.limits.max_push_constant_size) {
                return error.InvalidDesc;
            }
            for (desc.push_constants[0..i]) |previous| {
                const previous_end =
                    @as(u64, previous.offset) + previous.size;
                if (@as(u64, range.offset) < previous_end and
                    @as(u64, previous.offset) < end)
                {
                    return error.InvalidDesc;
                }
            }
        }
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const native_sets = try scratch.alloc(
        vk.DescriptorSetLayout,
        desc.sets.len,
    );
    for (desc.sets, 0..) |handle, i| {
        const entry = self.lookup(handle) orelse
            return error.InvalidResource;
        if (entry.kind != .descriptor_set_layout) {
            return error.InvalidResource;
        }
        native_sets[i] = entry.data.descriptor_set_layout.layout;
    }

    const native_ranges = try scratch.alloc(
        vk.PushConstantRange,
        desc.push_constants.len,
    );
    for (desc.push_constants, 0..) |range, i| {
        native_ranges[i] = .{
            .stage_flags = format.stageFlagsToVk(range.stages),
            .offset = range.offset,
            .size = range.size,
        };
    }

    const native = self.device_api.createPipelineLayout(
        self.vk_device,
        &.{
            .set_layout_count = @intCast(native_sets.len),
            .p_set_layouts = if (native_sets.len == 0)
                null
            else
                native_sets.ptr,
            .push_constant_range_count = @intCast(native_ranges.len),
            .p_push_constant_ranges = if (native_ranges.len == 0)
                null
            else
                native_ranges.ptr,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyPipelineLayout(
        self.vk_device,
        native,
        null,
    );

    const sets = try self.allocator.alloc(Resource, desc.sets.len);
    errdefer self.allocator.free(sets);
    for (desc.sets, 0..) |handle, i| {
        sets[i] = handle.handle;
    }
    const push_constants = try self.allocator.dupe(
        PipelineLayout.PushConstantRange,
        desc.push_constants,
    );
    errdefer self.allocator.free(push_constants);

    return self.newEntry(
        .pipeline_layout,
        .{ .pipeline_layout = .{
            .layout = native,
            .sets = sets,
            .push_constants = push_constants,
        } },
        desc.label,
    );
}

// -------------------------------------------------------------------
// Pipelines
// -------------------------------------------------------------------

fn createPipelineCompute(
    self: *Backend,
    desc: PipelineCompute.Desc,
) Device.CreateError!Resource {
    const layout_entry = self.lookup(desc.layout) orelse
        return error.InvalidResource;
    if (layout_entry.kind != .pipeline_layout) {
        return error.InvalidResource;
    }
    const layout = &layout_entry.data.pipeline_layout;

    const shader_entry = self.lookup(desc.shader) orelse
        return error.InvalidResource;
    if (shader_entry.kind != .shader) return error.InvalidResource;
    const shader = &shader_entry.data.shader;
    if (shader.stage != .compute) return error.InvalidDesc;

    const stage = vk.PipelineShaderStageCreateInfo{
        .stage = format.shaderStageToVk(shader.stage),
        .module = shader.module,
        .p_name = shader.entry.ptr,
    };

    const create_info = [_]vk.ComputePipelineCreateInfo{.{
        .stage = stage,
        .layout = layout.layout,
        .base_pipeline_index = -1,
    }};

    var native: vk.Pipeline = undefined;
    const result = self.device_api.createComputePipelines(
        self.vk_device,
        .null,
        &create_info,
        null,
        @ptrCast(&native),
    ) catch |err| return createError(err);
    if (result != .success) return error.InvalidDesc;
    errdefer self.device_api.destroyPipeline(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .pipeline_compute,
        .{ .pipeline_compute = .{
            .pipeline = native,
            .layout = desc.layout.handle,
            .bind_point = .compute,
        } },
        desc.label,
    );
}

fn createPipelineGraphics(
    self: *Backend,
    desc: PipelineGraphics.Desc,
) Device.CreateError!Resource {
    if (!self.dynamic_rendering_supported) return error.Unsupport;

    const layout_entry = self.lookup(desc.layout) orelse
        return error.InvalidResource;
    if (layout_entry.kind != .pipeline_layout) {
        return error.InvalidResource;
    }
    const layout = &layout_entry.data.pipeline_layout;

    const vertex_entry = self.lookup(desc.vertex) orelse
        return error.InvalidResource;
    if (vertex_entry.kind != .shader) return error.InvalidResource;
    const vertex = &vertex_entry.data.shader;
    if (vertex.stage != .vertex) return error.InvalidDesc;

    var fragment_entry: ?*state.ShaderState = null;
    if (desc.fragment) |handle| {
        const entry = self.lookup(handle) orelse
            return error.InvalidResource;
        if (entry.kind != .shader) return error.InvalidResource;
        if (entry.data.shader.stage != .fragment) {
            return error.InvalidDesc;
        }
        fragment_entry = &entry.data.shader;
    }

    if (desc.vertex_buffers.len > self.limits.max_vertex_buffers) {
        return error.InvalidDesc;
    }
    if (desc.color.targets.len > desc.color.formats.len) {
        return error.InvalidDesc;
    }

    if (std.math.isNan(desc.raster.line_width)) {
        return error.InvalidDesc;
    }
    if (desc.raster.line_width < 1 or
        desc.raster.line_width >
            self.properties.limits.line_width_range[1])
    {
        return error.Unsupport;
    }
    if (desc.raster.depth_clamp and self.features.depth_clamp != .true) {
        return error.Unsupport;
    }
    if (desc.raster.line_width > 1 and
        self.features.wide_lines != .true)
    {
        return error.Unsupport;
    }
    if (desc.primitive.polygon != .fill and
        self.features.fill_mode_non_solid != .true)
    {
        return error.Unsupport;
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    var stages: std.ArrayListUnmanaged(
        vk.PipelineShaderStageCreateInfo,
    ) = .empty;
    try stages.append(scratch, .{
        .stage = format.shaderStageToVk(vertex.stage),
        .module = vertex.module,
        .p_name = vertex.entry.ptr,
    });
    if (fragment_entry) |fragment| {
        try stages.append(scratch, .{
            .stage = format.shaderStageToVk(fragment.stage),
            .module = fragment.module,
            .p_name = fragment.entry.ptr,
        });
    }

    var bindings: std.ArrayListUnmanaged(
        vk.VertexInputBindingDescription,
    ) = .empty;
    var attributes: std.ArrayListUnmanaged(
        vk.VertexInputAttributeDescription,
    ) = .empty;
    for (desc.vertex_buffers, 0..) |vertex_buffer, binding| {
        if (vertex_buffer.stride > self.limits.max_vertex_stride) {
            return error.InvalidDesc;
        }
        try bindings.append(scratch, .{
            .binding = @intCast(binding),
            .stride = vertex_buffer.stride,
            .input_rate = .vertex,
        });
        for (vertex_buffer.attributes) |attribute| {
            if (attribute.location >=
                self.properties.limits.max_vertex_input_attributes)
            {
                return error.InvalidDesc;
            }
            if (attribute.offset >
                self.properties.limits.max_vertex_input_attribute_offset)
            {
                return error.InvalidDesc;
            }
            try attributes.append(scratch, .{
                .location = attribute.location,
                .binding = @intCast(binding),
                .format = format.vertexFormatToVk(attribute.format),
                .offset = attribute.offset,
            });
        }
    }

    var color_formats: std.ArrayListUnmanaged(vk.Format) = .empty;
    var color_targets: std.ArrayListUnmanaged(
        vk.PipelineColorBlendAttachmentState,
    ) = .empty;
    for (desc.color.formats) |pixel_format| {
        const native_format = format.formatToVk(pixel_format) orelse
            return error.InvalidDesc;
        if (format.isDepthStencil(pixel_format)) {
            return error.InvalidDesc;
        }
        if (!formatSupportsUsage(
            self,
            pixel_format,
            .{ .color_attachment = true },
        )) {
            return error.InvalidDesc;
        }
        try color_formats.append(scratch, native_format);
    }
    for (desc.color.formats, 0..) |_, i| {
        const target: PipelineGraphics.ColorTarget =
            if (i < desc.color.targets.len)
                desc.color.targets[i]
            else
                .{};
        try color_targets.append(scratch, .{
            .blend_enable = vk.Bool32.fromBool(target.blend),
            .src_color_blend_factor = format.blendFactorToVk(target.color.src),
            .dst_color_blend_factor = format.blendFactorToVk(target.color.dst),
            .color_blend_op = format.blendOpToVk(target.color.op),
            .src_alpha_blend_factor = format.blendFactorToVk(target.alpha.src),
            .dst_alpha_blend_factor = format.blendFactorToVk(target.alpha.dst),
            .alpha_blend_op = format.blendOpToVk(target.alpha.op),
            .color_write_mask = .{
                .r = target.write_mask.r,
                .g = target.write_mask.g,
                .b = target.write_mask.b,
                .a = target.write_mask.a,
            },
        });
    }

    var depth_format: vk.Format = .undefined;
    var stencil_format: vk.Format = .undefined;
    if (desc.depth_format) |pixel_format| {
        if (!format.isDepthStencil(pixel_format)) {
            return error.InvalidDesc;
        }
        if (!formatSupportsUsage(
            self,
            pixel_format,
            .{ .depth_stencil_attachment = true },
        )) {
            return error.InvalidDesc;
        }
        const native_format = format.formatToVk(pixel_format) orelse
            return error.InvalidDesc;
        depth_format = native_format;
        if (format.hasStencil(pixel_format)) {
            stencil_format = native_format;
        }
    }

    var rendering_info = vk.PipelineRenderingCreateInfo{
        .view_mask = 0,
        .color_attachment_count = @intCast(color_formats.items.len),
        .p_color_attachment_formats = if (color_formats.items.len == 0)
            null
        else
            color_formats.items.ptr,
        .depth_attachment_format = depth_format,
        .stencil_attachment_format = stencil_format,
    };

    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = @intCast(bindings.items.len),
        .p_vertex_binding_descriptions = if (bindings.items.len == 0)
            null
        else
            bindings.items.ptr,
        .vertex_attribute_description_count = @intCast(attributes.items.len),
        .p_vertex_attribute_descriptions = if (attributes.items.len == 0)
            null
        else
            attributes.items.ptr,
    };

    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = format.topologyToVk(desc.primitive.topology),
        .primitive_restart_enable = .false,
    };

    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .scissor_count = 1,
    };

    const depth_bias = desc.raster.depth_bias;
    const rasterization = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = vk.Bool32.fromBool(
            desc.raster.depth_clamp,
        ),
        .rasterizer_discard_enable = vk.Bool32.fromBool(
            desc.raster.discard,
        ),
        .polygon_mode = format.polygonToVk(desc.primitive.polygon),
        .cull_mode = format.cullToVk(desc.primitive.cull),
        .front_face = format.frontFaceToVk(desc.primitive.front_face),
        .depth_bias_enable = vk.Bool32.fromBool(
            depth_bias.constant != 0 or
                depth_bias.slope != 0 or
                depth_bias.clamp != 0,
        ),
        .depth_bias_constant_factor = depth_bias.constant,
        .depth_bias_clamp = depth_bias.clamp,
        .depth_bias_slope_factor = depth_bias.slope,
        .line_width = desc.raster.line_width,
    };

    const multisample = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = format.sampleCountToVk(desc.samples),
        .sample_shading_enable = .false,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.Bool32.fromBool(
            desc.alpha_to_coverage,
        ),
        .alpha_to_one_enable = .false,
    };

    if (desc.depth_stencil) |depth_state| {
        if (depth_state.stencil != null) {
            const depth_format_value = desc.depth_format orelse
                return error.InvalidDesc;
            if (!format.hasStencil(depth_format_value)) {
                return error.InvalidDesc;
            }
        }
    }

    var depth_stencil = vk.PipelineDepthStencilStateCreateInfo{
        .depth_test_enable = .false,
        .depth_write_enable = .false,
        .depth_compare_op = .always,
        .depth_bounds_test_enable = .false,
        .stencil_test_enable = .false,
        .front = .{
            .fail_op = .keep,
            .pass_op = .keep,
            .depth_fail_op = .keep,
            .compare_op = .always,
            .compare_mask = 0xff,
            .write_mask = 0xff,
            .reference = 0,
        },
        .back = .{
            .fail_op = .keep,
            .pass_op = .keep,
            .depth_fail_op = .keep,
            .compare_op = .always,
            .compare_mask = 0xff,
            .write_mask = 0xff,
            .reference = 0,
        },
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };
    if (desc.depth_format != null) {
        if (desc.depth_stencil) |state_desc| {
            depth_stencil.depth_test_enable = .true;
            depth_stencil.depth_write_enable = vk.Bool32.fromBool(
                state_desc.write,
            );
            depth_stencil.depth_compare_op =
                format.compareOpToVk(state_desc.compare);
            if (state_desc.stencil) |stencil| {
                depth_stencil.stencil_test_enable = .true;
                depth_stencil.front = .{
                    .fail_op = format.stencilOpToVk(
                        stencil.front.fail,
                    ),
                    .pass_op = format.stencilOpToVk(
                        stencil.front.pass,
                    ),
                    .depth_fail_op = format.stencilOpToVk(
                        stencil.front.depth_fail,
                    ),
                    .compare_op = format.compareOpToVk(
                        stencil.front.compare,
                    ),
                    .compare_mask = stencil.compare_mask,
                    .write_mask = stencil.write_mask,
                    .reference = stencil.reference,
                };
                depth_stencil.back = .{
                    .fail_op = format.stencilOpToVk(stencil.back.fail),
                    .pass_op = format.stencilOpToVk(stencil.back.pass),
                    .depth_fail_op = format.stencilOpToVk(
                        stencil.back.depth_fail,
                    ),
                    .compare_op = format.compareOpToVk(
                        stencil.back.compare,
                    ),
                    .compare_mask = stencil.compare_mask,
                    .write_mask = stencil.write_mask,
                    .reference = stencil.reference,
                };
            }
        }
    }

    const color_blend = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = @intCast(color_targets.items.len),
        .p_attachments = if (color_targets.items.len == 0)
            null
        else
            color_targets.items.ptr,
        .blend_constants = desc.color.constants,
    };

    const dynamic_states = [_]vk.DynamicState{
        .viewport,
        .scissor,
    };
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const create_info = [_]vk.GraphicsPipelineCreateInfo{.{
        .p_next = @ptrCast(&rendering_info),
        .stage_count = @intCast(stages.items.len),
        .p_stages = stages.items.ptr,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &input_assembly,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        .p_depth_stencil_state = &depth_stencil,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic_state,
        .layout = layout.layout,
        .render_pass = .null,
        .subpass = 0,
        .base_pipeline_index = -1,
    }};

    var native: vk.Pipeline = undefined;
    const result = self.device_api.createGraphicsPipelines(
        self.vk_device,
        .null,
        &create_info,
        null,
        @ptrCast(&native),
    ) catch |err| return createError(err);
    if (result != .success) return error.InvalidDesc;
    errdefer self.device_api.destroyPipeline(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .pipeline_graphics,
        .{ .pipeline_graphics = .{
            .pipeline = native,
            .layout = desc.layout.handle,
            .bind_point = .graphics,
        } },
        desc.label,
    );
}

// -------------------------------------------------------------------
// Synchronisation and command buffers
// -------------------------------------------------------------------

fn createCommandPool(
    self: *Backend,
    desc: CommandPool.Desc,
) Device.CreateError!Resource {
    if (@as(usize, desc.queue_family) >= self.families.len) {
        return error.InvalidDesc;
    }

    var flags = vk.CommandPoolCreateFlags{};
    flags.reset_command_buffer = desc.reset_individual;

    const native = self.device_api.createCommandPool(
        self.vk_device,
        &.{
            .flags = flags,
            .queue_family_index = desc.queue_family,
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyCommandPool(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .command_pool,
        .{ .command_pool = .{
            .pool = native,
            .queue_family = desc.queue_family,
            .reset_individual = desc.reset_individual,
        } },
        desc.label,
    );
}

fn createCommandBuffer(
    self: *Backend,
    desc: CommandBuffer.Desc,
) Device.CreateError!Resource {
    const pool_entry = self.lookup(desc.pool) orelse
        return error.InvalidResource;
    if (pool_entry.kind != .command_pool) {
        return error.InvalidResource;
    }
    const pool = &pool_entry.data.command_pool;

    const level: vk.CommandBufferLevel = switch (desc.level) {
        .primary => .primary,
        .secondary => .secondary,
    };

    const native_pool = pool.pool;
    var native: vk.CommandBuffer = undefined;
    self.device_api.allocateCommandBuffers(
        self.vk_device,
        &.{
            .command_pool = native_pool,
            .level = level,
            .command_buffer_count = 1,
        },
        @ptrCast(&native),
    ) catch |err| return createError(err);
    errdefer self.device_api.freeCommandBuffers(
        self.vk_device,
        native_pool,
        &.{native},
    );

    var buffer = state.CommandBufferState{
        .native = native,
        .pool = desc.pool.handle,
        .level = desc.level,
        .arena = std.heap.ArenaAllocator.init(self.allocator),
    };
    errdefer releaseCommandBufferHost(self, &buffer);

    const handle = try self.newEntry(
        .command_buffer,
        .{ .command_buffer = buffer },
        desc.label,
    );
    errdefer {
        const index = root.entryIndex(handle);
        self.recycleEntry(index);
    }

    const registered_pool = self.get(desc.pool, .command_pool) orelse
        return error.InvalidResource;
    try registered_pool.buffers.append(self.allocator, handle);
    return handle;
}

fn createFence(
    self: *Backend,
    desc: Fence.Desc,
) Device.CreateError!Resource {
    var flags = vk.FenceCreateFlags{};
    flags.signaled = desc.signaled;

    const native = self.device_api.createFence(
        self.vk_device,
        &.{ .flags = flags },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroyFence(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .fence,
        .{ .fence = .{ .fence = native } },
        desc.label,
    );
}

fn createSemaphore(
    self: *Backend,
    desc: Semaphore.Desc,
) Device.CreateError!Resource {
    if (desc.kind == .timeline and !self.timeline_supported) {
        return error.Unsupport;
    }

    var type_info = vk.SemaphoreTypeCreateInfo{
        .semaphore_type = switch (desc.kind) {
            .binary => .binary,
            .timeline => .timeline,
        },
        .initial_value = if (desc.kind == .timeline) desc.value else 0,
    };

    const native = self.device_api.createSemaphore(
        self.vk_device,
        &.{
            .p_next = @ptrCast(&type_info),
        },
        null,
    ) catch |err| return createError(err);
    errdefer self.device_api.destroySemaphore(
        self.vk_device,
        native,
        null,
    );

    return self.newEntry(
        .semaphore,
        .{ .semaphore = .{
            .semaphore = native,
            .kind = desc.kind,
        } },
        desc.label,
    );
}

// -------------------------------------------------------------------
// Surfaces and swapchains
// -------------------------------------------------------------------

fn createSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!Resource {
    const native = switch (desc.platform) {
        .win32 => try createWin32Surface(self, desc),
        .xlib => try createXlibSurface(self, desc),
        .xcb => try createXcbSurface(self, desc),
        .wayland => try createWaylandSurface(self, desc),
        .metal => try createMetalSurface(self, desc),
        .android => try createAndroidSurface(self, desc),
        .headless => try createHeadlessSurface(self, desc),
    };
    errdefer self.instance_api.destroySurfaceKHR(
        self.instance,
        native,
        null,
    );

    self.refreshPresentSupport(native);

    return self.newEntry(
        .surface,
        .{ .surface = .{
            .surface = native,
            .platform = desc.platform,
        } },
        desc.label,
    );
}

fn createWin32Surface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0) return error.InvalidDesc;
    if (self.instance_api.dispatch.vkCreateWin32SurfaceKHR == null) {
        return error.Unsupport;
    }
    const hinstance: vk.HINSTANCE = if (desc.display != 0)
        @ptrFromInt(desc.display)
    else if (builtin.os.tag == .windows)
        @ptrCast(Win32.GetModuleHandleW(null) orelse
            return error.DeviceFailure)
    else
        return error.InvalidDesc;
    const info = vk.Win32SurfaceCreateInfoKHR{
        .hinstance = hinstance,
        .hwnd = @ptrFromInt(desc.window),
    };
    return self.instance_api.createWin32SurfaceKHR(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createXlibSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0 or desc.display == 0) {
        return error.InvalidDesc;
    }
    if (self.instance_api.dispatch.vkCreateXlibSurfaceKHR == null) {
        return error.Unsupport;
    }
    const info = vk.XlibSurfaceCreateInfoKHR{
        .dpy = @ptrFromInt(desc.display),
        .window = @intCast(desc.window),
    };
    return self.instance_api.createXlibSurfaceKHR(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createXcbSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0 or desc.display == 0) {
        return error.InvalidDesc;
    }
    if (self.instance_api.dispatch.vkCreateXcbSurfaceKHR == null) {
        return error.Unsupport;
    }
    const info = vk.XcbSurfaceCreateInfoKHR{
        .connection = @ptrFromInt(desc.display),
        .window = @intCast(desc.window),
    };
    return self.instance_api.createXcbSurfaceKHR(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createWaylandSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0 or desc.display == 0) {
        return error.InvalidDesc;
    }
    if (self.instance_api.dispatch.vkCreateWaylandSurfaceKHR == null) {
        return error.Unsupport;
    }
    const info = vk.WaylandSurfaceCreateInfoKHR{
        .display = @ptrFromInt(desc.display),
        .surface = @ptrFromInt(desc.window),
    };
    return self.instance_api.createWaylandSurfaceKHR(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createMetalSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0) return error.InvalidDesc;
    if (self.instance_api.dispatch.vkCreateMetalSurfaceEXT == null) {
        return error.Unsupport;
    }
    const info = vk.MetalSurfaceCreateInfoEXT{
        .p_layer = @ptrFromInt(desc.window),
    };
    return self.instance_api.createMetalSurfaceEXT(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createAndroidSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    if (desc.window == 0) return error.InvalidDesc;
    if (self.instance_api.dispatch.vkCreateAndroidSurfaceKHR == null) {
        return error.Unsupport;
    }
    const info = vk.AndroidSurfaceCreateInfoKHR{
        .window = @ptrFromInt(desc.window),
    };
    return self.instance_api.createAndroidSurfaceKHR(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createHeadlessSurface(
    self: *Backend,
    desc: Surface.Desc,
) Device.CreateError!vk.SurfaceKHR {
    _ = desc;
    if (self.instance_api.dispatch.vkCreateHeadlessSurfaceEXT == null) {
        return error.Unsupport;
    }
    const info = vk.HeadlessSurfaceCreateInfoEXT{};
    return self.instance_api.createHeadlessSurfaceEXT(
        self.instance,
        &info,
        null,
    ) catch |err| return createError(err);
}

fn createSwapchain(
    self: *Backend,
    desc: Swapchain.Desc,
) Device.CreateError!Resource {
    if (desc.extent.width == 0 or desc.extent.height == 0) {
        return error.InvalidDesc;
    }
    if (desc.extent.depth != 1) return error.InvalidDesc;
    if (format.textureUsageEmpty(desc.usage)) return error.InvalidDesc;

    const surface_entry = self.lookup(desc.surface) orelse
        return error.InvalidResource;
    if (surface_entry.kind != .surface) {
        return error.InvalidResource;
    }
    const surface = &surface_entry.data.surface;

    const native_format = format.formatToVk(desc.format) orelse
        return error.InvalidDesc;

    const family = self.graphicsPresentFamily(surface.surface) orelse
        return error.Unsupport;

    const capabilities = self.instance_api
        .getPhysicalDeviceSurfaceCapabilitiesKHR(
        self.physical_device,
        surface.surface,
    ) catch |err| return createError(err);

    const surface_format = findSurfaceFormat(
        self,
        surface.surface,
        desc.format,
    ) orelse return error.InvalidDesc;
    if (!surfaceSupportsPresentMode(
        self,
        surface.surface,
        desc.mode,
    )) {
        return error.InvalidDesc;
    }

    const usage = format.textureUsageToVk(desc.usage);
    if (!capabilities.supported_usage_flags.contains(usage)) {
        return error.InvalidDesc;
    }

    var extent = vk.Extent2D{
        .width = desc.extent.width,
        .height = desc.extent.height,
    };
    if (capabilities.current_extent.width !=
        std.math.maxInt(u32))
    {
        extent = capabilities.current_extent;
    } else {
        extent.width = std.math.clamp(
            extent.width,
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );
        extent.height = std.math.clamp(
            extent.height,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );
    }

    var image_count: u32 = if (desc.images == 0)
        capabilities.min_image_count + 1
    else
        desc.images;
    image_count = @max(image_count, capabilities.min_image_count);
    if (capabilities.max_image_count != 0) {
        image_count = @min(image_count, capabilities.max_image_count);
    }

    const composite_alpha: vk.CompositeAlphaFlagsKHR =
        if (capabilities.supported_composite_alpha.opaque_khr)
            .{ .opaque_khr = true }
        else if (capabilities.supported_composite_alpha.pre_multiplied_khr)
            .{ .pre_multiplied_khr = true }
        else if (capabilities.supported_composite_alpha.post_multiplied_khr)
            .{ .post_multiplied_khr = true }
        else
            .{ .inherit_khr = true };

    const native = self.device_api.createSwapchainKHR(
        self.vk_device,
        &.{
            .surface = surface.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = usage,
            .image_sharing_mode = .exclusive,
            .pre_transform = capabilities.current_transform,
            .composite_alpha = composite_alpha,
            .present_mode = format.presentModeToVk(desc.mode),
            .clipped = .true,
            .old_swapchain = .null,
        },
        null,
    ) catch |err| return mapSwapchainCreateError(err);
    errdefer self.device_api.destroySwapchainKHR(
        self.vk_device,
        native,
        null,
    );

    const images = self.device_api.getSwapchainImagesAllocKHR(
        self.vk_device,
        native,
        self.allocator,
    ) catch |err| return createError(err);
    defer self.allocator.free(images);

    const acquired = try self.allocator.alloc(bool, images.len);
    errdefer self.allocator.free(acquired);
    @memset(acquired, false);

    const handles = try self.allocator.alloc(Resource, images.len);
    errdefer self.allocator.free(handles);

    const handle = try self.newEntry(
        .swapchain,
        .{ .swapchain = .{
            .swapchain = native,
            .surface = desc.surface.handle,
            .format = native_format,
            .extent = extent,
            .usage = desc.usage,
            .images = handles,
            .acquired = acquired,
            .queue_family = family,
        } },
        desc.label,
    );
    errdefer {
        const index = root.entryIndex(handle);
        const entry = &self.entries.items[index];
        entry.data.swapchain.images = handles[0..0];
        entry.data.swapchain.acquired = acquired[0..0];
        entry.data.swapchain.swapchain = .null;
        self.recycleEntry(index);
    }

    const texture_desc = Texture.Desc{
        .kind = .d2,
        .extent = .{
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .format = desc.format,
        .mips = 1,
        .layers = 1,
        .samples = .x1,
        .usage = desc.usage,
    };

    var created: usize = 0;
    errdefer {
        for (handles[0..created]) |texture| {
            const index = root.entryIndex(texture);
            const child = &self.entries.items[index];
            if (child.live) {
                self.device_api.destroyImageView(
                    self.vk_device,
                    child.data.texture.view,
                    null,
                );
                self.recycleEntry(index);
            }
        }
    }

    for (images, 0..) |image, i| {
        const view = try createImageView(
            self,
            image,
            texture_desc,
            .{ .color = true },
        );
        handles[i] = self.newEntry(
            .texture,
            .{ .texture = .{
                .image = image,
                .memory = null,
                .view = view,
                .desc = texture_desc,
                .owner = handle,
            } },
            desc.label,
        ) catch |err| {
            self.device_api.destroyImageView(
                self.vk_device,
                view,
                null,
            );
            return err;
        };
        created += 1;
    }

    return handle;
}

/// Finds a surface format matching an RHI format.
///
/// # Args
/// - `self`: owning backend.
/// - `surface`: native surface.
/// - `pixel_format`: RHI format to match.
///
/// # Return
/// The matching surface format and colour space,
/// or null when the surface does not list it.
pub fn findSurfaceFormat(
    self: *Backend,
    surface: vk.SurfaceKHR,
    pixel_format: Resource.Format,
) ?vk.SurfaceFormatKHR {
    const native_format = format.formatToVk(pixel_format) orelse
        return null;
    const formats = self.instance_api.getPhysicalDeviceSurfaceFormatsAllocKHR(
        self.physical_device,
        surface,
        self.allocator,
    ) catch return null;
    defer self.allocator.free(formats);

    if (formats.len == 1 and formats[0].format == .undefined) {
        return .{
            .format = native_format,
            .color_space = formats[0].color_space,
        };
    }
    for (formats) |candidate| {
        if (candidate.format == native_format) return candidate;
    }
    return null;
}

/// Reports whether a surface accepts a format.
///
/// # Args
/// - `self`: owning backend.
/// - `surface`: native surface.
/// - `pixel_format`: RHI format to test.
///
/// # Return
/// True when the surface lists the format, or lists
/// `undefined` to accept any format.
pub fn surfaceSupportsFormat(
    self: *Backend,
    surface: vk.SurfaceKHR,
    pixel_format: Resource.Format,
) bool {
    return findSurfaceFormat(self, surface, pixel_format) != null;
}

/// Reports whether a surface accepts a present mode.
///
/// # Args
/// - `self`: owning backend.
/// - `surface`: native surface.
/// - `mode`: RHI present mode.
///
/// # Return
/// True when the driver lists the mode.
pub fn surfaceSupportsPresentMode(
    self: *Backend,
    surface: vk.SurfaceKHR,
    mode: Swapchain.PresentMode,
) bool {
    const modes = self.instance_api
        .getPhysicalDeviceSurfacePresentModesAllocKHR(
        self.physical_device,
        surface,
        self.allocator,
    ) catch return false;
    defer self.allocator.free(modes);

    const wanted = format.presentModeToVk(mode);
    for (modes) |candidate| {
        if (candidate == wanted) return true;
    }
    return false;
}

const MemoryAllocation = struct {
    memory: vk.DeviceMemory,
    flags: vk.MemoryPropertyFlags,
};

fn allocateMemory(
    self: *Backend,
    requirements: vk.MemoryRequirements,
    required: vk.MemoryPropertyFlags,
    preferred: vk.MemoryPropertyFlags,
) Device.CreateError!MemoryAllocation {
    const type_index = self.findMemoryType(
        requirements.memory_type_bits,
        required,
        preferred,
    ) orelse return error.OutOfDeviceMemory;

    const memory = self.device_api.allocateMemory(
        self.vk_device,
        &.{
            .allocation_size = requirements.size,
            .memory_type_index = type_index,
        },
        null,
    ) catch |err| return createError(err);

    return .{
        .memory = memory,
        .flags = self.memory_properties.memory_types[type_index]
            .property_flags,
    };
}

/// Maps a Vulkan creation failure to an RHI error.
///
/// # Args
/// - `err`: error returned by the Vulkan wrapper.
///
/// # Return
/// The matching RHI creation error.
pub fn createError(err: anyerror) Device.CreateError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.OutOfDeviceMemory => error.OutOfDeviceMemory,
        error.DeviceLost => error.DeviceLost,
        error.TooManyObjects => error.TooManyObjects,
        error.InvalidShaderNV,
        error.CompressionExhaustedEXT,
        error.InvalidOpaqueCaptureAddressKHR,
        error.FormatNotSupported,
        => error.InvalidDesc,
        error.FragmentationEXT,
        error.OutOfPoolMemory,
        error.FragmentedPool,
        => error.OutOfDeviceMemory,
        error.ExtensionNotPresent, error.FeatureNotPresent => error.Unsupport,
        else => error.DeviceFailure,
    };
}

fn mapError(err: anyerror) Device.MapError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.DeviceLost => error.DeviceLost,
        error.MemoryMapFailed => error.NotHostVisible,
        else => error.DeviceFailure,
    };
}

fn mapSwapchainCreateError(
    err: anyerror,
) Device.CreateError {
    return switch (err) {
        error.DeviceLost => error.DeviceLost,
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.OutOfDeviceMemory => error.OutOfDeviceMemory,
        error.SurfaceLostKHR => error.InvalidDesc,
        error.NativeWindowInUseKHR => error.InvalidDesc,
        else => error.DeviceFailure,
    };
}
