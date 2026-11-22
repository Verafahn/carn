//! Feature queries and immediate device operations.
//!
//! Descriptor writes validate subresource ranges but
//! bind the texture's full image view; the backend does
//! not create narrower views for a range.

const std = @import("std");
const rhi = @import("rhi");
const vk = @import("vk");

const root = @import("root.zig");
const format = @import("format.zig");
const state = @import("state.zig");
const command = @import("command.zig");
const resource = @import("resource.zig");

const Backend = root.Backend;
const Device = rhi.Device;
const Resource = rhi.Resource;
const Queue = rhi.Queue;
const DescriptorSet = Resource.DescriptorSet;

/// Answers a capability or limit query.
///
/// # Args
/// - `self`: owning backend.
/// - `feature`: query to perform.
/// - `feature_index`: index inside a repeated result.
/// - `handle`: optional resource the query is scoped to.
/// - `out`: optional output buffer.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The support verdict.
///
/// # Error
/// - `error.IndexOutOfBounds`: the index selects a
///   missing entry.
/// - `error.InsufficientBuffer`: a required output is
///   missing or too small.
/// - `error.InvalidResource`: `handle` is not live or
///   has the wrong kind.
pub fn query(
    self: *Backend,
    feature: Device.Feature,
    feature_index: u32,
    handle: ?Resource,
    out: ?[]u8,
    ret_addr: usize,
) Device.QueryError!Device.Support {
    _ = ret_addr;

    return switch (feature) {
        .format => |query_desc| blk: {
            if (feature_index != 0) return error.IndexOutOfBounds;
            if (format.formatToVk(query_desc.format) == null) {
                break :blk .unsupported;
            }
            break :blk if (resource.formatSupportsUsage(
                self,
                query_desc.format,
                query_desc.usage,
            )) .supported else .unsupported;
        },
        .texture => |desc| blk: {
            if (feature_index != 0) return error.IndexOutOfBounds;
            break :blk if (resource.textureSupported(self, desc))
                .supported
            else
                .unsupported;
        },
        .swapchain => |query_desc| blk: {
            if (feature_index != 0) return error.IndexOutOfBounds;
            const surface = handle orelse
                return error.InvalidResource;
            const entry = self.lookup(surface) orelse
                return error.InvalidResource;
            if (entry.kind != .surface) {
                return error.InvalidResource;
            }
            const native = entry.data.surface.surface;
            if (!resource.surfaceSupportsFormat(
                self,
                native,
                query_desc.format,
            )) {
                break :blk .unsupported;
            }
            break :blk if (resource.surfaceSupportsPresentMode(
                self,
                native,
                query_desc.mode,
            )) .supported else .unsupported;
        },
        .queue_family => blk: {
            if (@as(usize, feature_index) >= self.families.len) {
                return error.IndexOutOfBounds;
            }
            const family = self.families[feature_index];
            try writeOut(
                Device.QueueFamily,
                Device.QueryError,
                out,
                .{
                    .count = 1,
                    .kinds = family.kinds,
                },
            );
            break :blk .supported;
        },
        .limits => blk: {
            if (feature_index != 0) return error.IndexOutOfBounds;
            try writeOut(
                Device.Limits,
                Device.QueryError,
                out,
                self.limits,
            );
            break :blk .supported;
        },
    };
}

/// Returns a queue by its flat index.
///
/// # Args
/// - `self`: owning backend.
/// - `index`: position in the flat queue list.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// A queue that identifies its family and index.
///
/// # Error
/// - `error.NoQueues`: the backend exposes no queues.
/// - `error.IndexOutOfBounds`: `index` is past the
///   queue list.
pub fn getQueue(
    self: *Backend,
    index: u32,
    ret_addr: usize,
) Device.GetQueueError!Queue {
    _ = ret_addr;
    if (self.queues.len == 0) return error.NoQueues;
    if (@as(usize, index) >= self.queues.len) {
        return error.IndexOutOfBounds;
    }
    const queue_state = &self.queues[index];
    return .{
        .vptr = queue_state,
        .vtable = &root.queue_vtable,
        .family_index = queue_state.family_index,
        .queue_index = queue_state.queue_index,
    };
}

/// Executes an immediate operation.
///
/// # Args
/// - `self`: owning backend.
/// - `op`: operation to execute.
/// - `out`: optional output buffer.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.InvalidDesc`: the payload is malformed.
/// - `error.InvalidResource`: a referenced resource is
///   not live.
/// - `error.IndexOutOfBounds`: an index selects a
///   missing entry.
/// - `error.InsufficientBuffer`: a required output is
///   missing or too small.
/// - `error.CommandBufferState`: a buffer is in the
///   wrong state.
/// - `error.Timeout`: a timed operation expired.
/// - `error.SwapchainOutOfDate`: the swapchain is
///   stale.
/// - `error.SurfaceLost`: the surface is gone.
/// - `error.Unsupport`: the operation is unavailable.
pub fn deviceOp(
    self: *Backend,
    op: Device.DeviceOp,
    out: ?[]u8,
    ret_addr: usize,
) Device.DeviceOpError!void {
    _ = ret_addr;

    switch (op) {
        .wait_idle => {
            self.device_api.deviceWaitIdle(self.vk_device) catch |err| {
                return opError(err);
            };
        },
        .acquire => |acquire| try acquireImage(self, acquire, out),
        .swapchain_image => |image| try swapchainImage(self, image, out),
        .reset_command_pool => |handle| try resetCommandPool(
            self,
            handle,
        ),
        .reset_command_buffer => |handle| try resetCommandBuffer(
            self,
            handle,
        ),
        .reset_fence => |handle| try resetFence(self, handle),
        .update_descriptor_set => |write| try updateDescriptorSet(
            self,
            write,
        ),
    }
}

fn acquireImage(
    self: *Backend,
    acquire: Device.Acquire,
    out: ?[]u8,
) Device.DeviceOpError!void {
    const out_bytes = out orelse return error.InsufficientBuffer;
    if (out_bytes.len < @sizeOf(u32)) return error.InsufficientBuffer;

    const entry = self.lookup(acquire.swapchain) orelse
        return error.InvalidResource;
    if (entry.kind != .swapchain) return error.InvalidResource;
    const chain = &entry.data.swapchain;

    var native_signal: vk.Semaphore = .null;
    if (acquire.signal) |handle| {
        const semaphore = self.get(handle, .semaphore) orelse
            return error.InvalidResource;
        if (semaphore.kind != .binary) return error.InvalidDesc;
        native_signal = semaphore.semaphore;
    }

    var native_fence: vk.Fence = .null;
    if (acquire.fence) |handle| {
        const fence = self.get(handle, .fence) orelse
            return error.InvalidResource;
        native_fence = fence.fence;
    }

    const timeout = acquire.timeout_ns orelse std.math.maxInt(u64);
    const result = self.device_api.acquireNextImageKHR(
        self.vk_device,
        chain.swapchain,
        timeout,
        native_signal,
        native_fence,
    ) catch |err| return switch (err) {
        error.OutOfDateKHR => error.SwapchainOutOfDate,
        error.SurfaceLostKHR => error.SurfaceLost,
        error.DeviceLost => error.DeviceLost,
        error.OutOfHostMemory => error.OutOfMemory,
        else => error.DeviceFailure,
    };

    switch (result.result) {
        .success, .suboptimal_khr => {},
        .timeout, .not_ready => return error.Timeout,
        else => return error.DeviceFailure,
    }

    try writeOut(
        u32,
        Device.DeviceOpError,
        out,
        result.image_index,
    );
    if (@as(usize, result.image_index) < chain.acquired.len) {
        chain.acquired[result.image_index] = true;
    }
}

fn swapchainImage(
    self: *Backend,
    image: Device.SwapchainImage,
    out: ?[]u8,
) Device.DeviceOpError!void {
    const entry = self.lookup(image.swapchain) orelse
        return error.InvalidResource;
    if (entry.kind != .swapchain) return error.InvalidResource;
    const chain = &entry.data.swapchain;
    if (@as(usize, image.index) >= chain.images.len) {
        return error.IndexOutOfBounds;
    }
    try writeOut(
        Resource,
        Device.DeviceOpError,
        out,
        chain.images[image.index],
    );
}

fn resetCommandPool(
    self: *Backend,
    handle: Resource.CommandPool,
) Device.DeviceOpError!void {
    const entry = self.lookup(handle) orelse
        return error.InvalidResource;
    if (entry.kind != .command_pool) return error.InvalidResource;
    const pool = &entry.data.command_pool;

    self.device_api.resetCommandPool(
        self.vk_device,
        pool.pool,
        .{},
    ) catch |err| return opError(err);

    for (pool.buffers.items) |child_handle| {
        const child_entry = self.lookup(child_handle) orelse continue;
        if (child_entry.kind != .command_buffer) continue;
        command.resetState(&child_entry.data.command_buffer);
    }
}

fn resetCommandBuffer(
    self: *Backend,
    handle: Resource.CommandBuffer,
) Device.DeviceOpError!void {
    const entry = self.lookup(handle) orelse
        return error.InvalidResource;
    if (entry.kind != .command_buffer) return error.InvalidResource;
    const buffer = &entry.data.command_buffer;

    const pool_entry = self.get(buffer.pool, .command_pool) orelse
        return error.InvalidResource;
    if (!pool_entry.reset_individual) return error.InvalidState;

    self.device_api.resetCommandBuffer(
        buffer.native,
        .{},
    ) catch |err| return opError(err);

    command.resetState(buffer);
}

fn resetFence(
    self: *Backend,
    handle: Resource.Fence,
) Device.DeviceOpError!void {
    const fence = self.get(handle, .fence) orelse
        return error.InvalidResource;
    self.device_api.resetFences(
        self.vk_device,
        &.{fence.fence},
    ) catch |err| return opError(err);
}

fn updateDescriptorSet(
    self: *Backend,
    write: Device.DescriptorWrite,
) Device.DeviceOpError!void {
    const set_entry = self.lookup(write.set) orelse
        return error.InvalidResource;
    if (set_entry.kind != .descriptor_set) {
        return error.InvalidResource;
    }
    const set = &set_entry.data.descriptor_set;

    const layout_entry = self.lookup(set.layout) orelse
        return error.InvalidResource;
    if (layout_entry.kind != .descriptor_set_layout) {
        return error.InvalidResource;
    }
    const layout = &layout_entry.data.descriptor_set_layout;

    if (write.writes.len == 0) return;

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const native_writes = try scratch.alloc(
        vk.WriteDescriptorSet,
        write.writes.len,
    );

    for (write.writes, 0..) |item, i| {
        const binding_index = writeBinding(item);
        const array_index = writeIndex(item);
        const binding = findBinding(
            layout.bindings,
            binding_index,
        ) orelse return error.InvalidDesc;

        native_writes[i] = .{
            .dst_set = set.set,
            .dst_binding = binding.binding,
            .dst_array_element = array_index,
            .descriptor_count = 1,
            .descriptor_type = format.descriptorTypeToVk(binding.kind) orelse
                return error.Unsupport,
        };

        if (array_index >= binding.count) {
            return error.InvalidDesc;
        }

        switch (item) {
            .buffer => |buffer_write| {
                if (binding.kind != .uniform_buffer and
                    binding.kind != .storage_buffer)
                {
                    return error.InvalidDesc;
                }
                const buffer = self.get(buffer_write.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (binding.kind == .uniform_buffer and
                    !buffer.usage.uniform)
                {
                    return error.InvalidDesc;
                }
                if (binding.kind == .storage_buffer and
                    !buffer.usage.storage)
                {
                    return error.InvalidDesc;
                }
                const size: u64 = if (buffer_write.size == 0)
                    buffer.size -| buffer_write.offset
                else
                    buffer_write.size;
                if (!root.fits(
                    buffer_write.offset,
                    size,
                    buffer.size,
                )) {
                    return error.InvalidDesc;
                }
                const info = try scratch.create(vk.DescriptorBufferInfo);
                info.* = .{
                    .buffer = buffer.buffer,
                    .offset = buffer_write.offset,
                    .range = if (buffer_write.size == 0)
                        vk.WHOLE_SIZE
                    else
                        buffer_write.size,
                };
                native_writes[i].p_buffer_info = @ptrCast(info);
            },
            .texture => |texture_write| {
                if (binding.kind != .sampled_texture and
                    binding.kind != .storage_texture)
                {
                    return error.InvalidDesc;
                }
                const texture = self.get(texture_write.texture, .texture) orelse
                    return error.InvalidResource;
                switch (binding.kind) {
                    .sampled_texture => if (!texture.desc.usage.sampled) {
                        return error.InvalidDesc;
                    },
                    .storage_texture => if (!texture.desc.usage.storage) {
                        return error.InvalidDesc;
                    },
                    else => return error.InvalidDesc,
                }
                try validateRange(texture.desc, texture_write.range);
                const info = try scratch.create(vk.DescriptorImageInfo);
                info.* = .{
                    .sampler = .null,
                    .image_view = texture.view,
                    .image_layout = if (binding.kind == .storage_texture)
                        .general
                    else
                        .shader_read_only_optimal,
                };
                native_writes[i].p_image_info = @ptrCast(info);
            },
            .sampler => |sampler_write| {
                if (binding.kind != .sampler) {
                    return error.InvalidDesc;
                }
                const sampler = self.get(sampler_write.sampler, .sampler) orelse
                    return error.InvalidResource;
                const info = try scratch.create(vk.DescriptorImageInfo);
                info.* = .{
                    .sampler = sampler.sampler,
                    .image_view = .null,
                    .image_layout = .undefined,
                };
                native_writes[i].p_image_info = @ptrCast(info);
            },
            .combined => |combined| {
                if (binding.kind != .combined_sampler) {
                    return error.InvalidDesc;
                }
                const texture = self.get(combined.texture, .texture) orelse
                    return error.InvalidResource;
                if (!texture.desc.usage.sampled) return error.InvalidDesc;
                try validateRange(texture.desc, combined.range);
                const sampler = self.get(combined.sampler, .sampler) orelse
                    return error.InvalidResource;
                const info = try scratch.create(vk.DescriptorImageInfo);
                info.* = .{
                    .sampler = sampler.sampler,
                    .image_view = texture.view,
                    .image_layout = .shader_read_only_optimal,
                };
                native_writes[i].p_image_info = @ptrCast(info);
            },
        }
    }

    self.device_api.updateDescriptorSets(
        self.vk_device,
        native_writes,
        null,
    );
}

fn validateRange(
    desc: rhi.Resource.Texture.Desc,
    range: rhi.Resource.Texture.SubresourceRange,
) Device.DeviceOpError!void {
    if (range.mip >= desc.mips) return error.InvalidDesc;
    if (range.layer >= desc.layers) return error.InvalidDesc;

    const mip_count: u32 = if (range.mip_count == 0)
        @as(u32, desc.mips) - @as(u32, range.mip)
    else
        range.mip_count;
    const layer_count: u32 = if (range.layer_count == 0)
        @as(u32, desc.layers) - @as(u32, range.layer)
    else
        range.layer_count;

    if (@as(u32, range.mip) + mip_count > desc.mips) {
        return error.InvalidDesc;
    }
    if (@as(u32, range.layer) + layer_count > desc.layers) {
        return error.InvalidDesc;
    }
}

fn findBinding(
    bindings: []const rhi.Resource.DescriptorSetLayout.Binding,
    index: u32,
) ?rhi.Resource.DescriptorSetLayout.Binding {
    for (bindings) |binding| {
        if (binding.binding == index) return binding;
    }
    return null;
}

fn writeBinding(write: DescriptorSet.Write) u32 {
    return switch (write) {
        .buffer => |item| item.binding,
        .texture => |item| item.binding,
        .sampler => |item| item.binding,
        .combined => |item| item.binding,
    };
}

fn writeIndex(write: DescriptorSet.Write) u32 {
    return switch (write) {
        .buffer => |item| item.index,
        .texture => |item| item.index,
        .sampler => |item| item.index,
        .combined => |item| item.index,
    };
}

fn writeOut(
    comptime T: type,
    comptime E: type,
    out: ?[]u8,
    value: T,
) E!void {
    const bytes = out orelse return error.InsufficientBuffer;
    if (bytes.len < @sizeOf(T)) return error.InsufficientBuffer;
    @memcpy(bytes[0..@sizeOf(T)], std.mem.asBytes(&value));
}

fn opError(err: anyerror) Device.DeviceOpError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.DeviceLost => error.DeviceLost,
        error.OutOfDeviceMemory => error.DeviceFailure,
        error.Timeout => error.Timeout,
        else => error.DeviceFailure,
    };
}
