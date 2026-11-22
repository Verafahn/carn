//! Recording of RHI commands into Vulkan command buffers.
//!
//! Primary buffers record straight into their native
//! `VkCommandBuffer`. Secondary buffers store deep copies
//! of the commands and replay them into the primary when
//! `execute_commands` is recorded.

const std = @import("std");
const rhi = @import("rhi");
const vk = @import("vk");

const root = @import("root.zig");
const format = @import("format.zig");
const state = @import("state.zig");

const Backend = root.Backend;
const Device = rhi.Device;
const Resource = rhi.Resource;
const Texture = Resource.Texture;
const CommandBuffer = Resource.CommandBuffer;
const PipelineLayout = Resource.PipelineLayout;

/// Records one command into a command buffer.
///
/// # Args
/// - `self`: owning backend.
/// - `handle`: command buffer receiving the command.
/// - `cmd`: command to append.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.InvalidDesc`: the command payload is
///   malformed.
/// - `error.InvalidResource`: a referenced resource is
///   not live.
/// - `error.CommandBufferState`: the buffer is not
///   recording.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub fn record(
    self: *Backend,
    handle: Resource,
    cmd: *const Device.Command,
    ret_addr: usize,
) Device.RecordError!void {
    try recordInternal(self, handle, cmd, ret_addr, false);
}

fn recordInternal(
    self: *Backend,
    handle: Resource,
    cmd: *const Device.Command,
    ret_addr: usize,
    replay: bool,
) Device.RecordError!void {
    const entry = self.lookup(handle) orelse return error.InvalidResource;
    if (entry.kind != .command_buffer) return error.InvalidResource;
    const buffer = &entry.data.command_buffer;

    switch (cmd.*) {
        .begin => |begin| {
            if (replay) return error.InvalidDesc;
            return beginRecording(self, buffer, begin);
        },
        .end => {
            if (replay) return error.InvalidDesc;
            return endRecording(self, buffer);
        },
        else => {},
    }

    try recordBody(self, handle, buffer, cmd, ret_addr);

    if (!replay and buffer.level == .secondary) {
        const cloned = try cloneCommand(
            buffer.arena.allocator(),
            cmd.*,
        );
        try buffer.stored.append(self.allocator, .{ .cmd = cloned });
    }
}

fn beginRecording(
    self: *Backend,
    buffer: *state.CommandBufferState,
    begin: Device.Command.Begin,
) Device.RecordError!void {
    if (buffer.phase != .initial) return error.CommandBufferState;
    resetState(buffer);
    buffer.phase = .recording;
    buffer.one_time = begin.one_time;

    if (buffer.level == .primary) {
        var flags = vk.CommandBufferUsageFlags{};
        flags.one_time_submit = begin.one_time;
        self.device_api.beginCommandBuffer(
            buffer.native,
            &.{ .flags = flags },
        ) catch |err| return recordError(err);
    }
}

fn endRecording(
    self: *Backend,
    buffer: *state.CommandBufferState,
) Device.RecordError!void {
    if (buffer.phase != .recording) return error.CommandBufferState;
    if (buffer.in_render_pass) return error.CommandBufferState;

    if (buffer.level == .primary) {
        self.device_api.endCommandBuffer(buffer.native) catch |err|
            return recordError(err);
    }
    buffer.phase = .ended;
}

fn resetCommandBufferState(buffer: *state.CommandBufferState) void {
    buffer.in_render_pass = false;
    buffer.pipeline = null;
    buffer.pipeline_layout = null;
    buffer.pipeline_kind = null;
    buffer.viewport_set = false;
    buffer.scissor_set = false;
    buffer.index_buffer = false;
    buffer.kinds = .{};
}

pub fn resetState(buffer: *state.CommandBufferState) void {
    buffer.phase = .initial;
    buffer.one_time = false;
    buffer.in_render_pass = false;
    buffer.pipeline = null;
    buffer.pipeline_layout = null;
    buffer.pipeline_kind = null;
    buffer.viewport_set = false;
    buffer.scissor_set = false;
    buffer.index_buffer = false;
    buffer.kinds = .{};
    buffer.stored.clearRetainingCapacity();
    _ = buffer.arena.reset(.retain_capacity);
}

fn requireRecording(
    buffer: *const state.CommandBufferState,
) Device.RecordError!void {
    if (buffer.phase != .recording) return error.CommandBufferState;
}

fn requireGraphics(
    buffer: *const state.CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind != .graphics) {
        return error.CommandBufferState;
    }
}

fn requireCompute(
    buffer: *const state.CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind != .compute) {
        return error.CommandBufferState;
    }
}

fn requireGraphicsOrCompute(
    buffer: *const state.CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind == null) {
        return error.CommandBufferState;
    }
}

fn requireDraw(
    buffer: *const state.CommandBufferState,
) Device.RecordError!void {
    try requireGraphics(buffer);
    if (!buffer.in_render_pass) return error.CommandBufferState;
    if (!buffer.viewport_set or !buffer.scissor_set) {
        return error.CommandBufferState;
    }
}

fn recordBody(
    self: *Backend,
    handle: Resource,
    buffer: *state.CommandBufferState,
    cmd: *const Device.Command,
    ret_addr: usize,
) Device.RecordError!void {
    const primary = buffer.level == .primary;

    switch (cmd.*) {
        .begin, .end => unreachable,

        .begin_render_pass => |pass| {
            try requireRecording(buffer);
            if (!self.dynamic_rendering_supported) {
                return error.Unsupport;
            }
            if (buffer.in_render_pass) return error.CommandBufferState;
            try validateRenderPass(self, pass);
            buffer.in_render_pass = true;
            if (primary) try beginRenderPass(self, buffer, pass);
        },

        .end_render_pass => {
            try requireRecording(buffer);
            if (!buffer.in_render_pass) return error.CommandBufferState;
            buffer.in_render_pass = false;
            if (primary) {
                if (self.core_dynamic_rendering) {
                    self.device_api.cmdEndRendering(buffer.native);
                } else {
                    self.device_api.cmdEndRenderingKHR(buffer.native);
                }
            }
        },

        .bind_pipeline => |bind| {
            try requireRecording(buffer);
            switch (bind) {
                .graphics => |pipeline| {
                    const pipeline_state = self.get(
                        pipeline,
                        .pipeline_graphics,
                    ) orelse return error.InvalidResource;
                    buffer.pipeline = pipeline.handle;
                    buffer.pipeline_layout = pipeline_state.layout;
                    buffer.pipeline_kind = .graphics;
                    if (primary) {
                        self.device_api.cmdBindPipeline(
                            buffer.native,
                            .graphics,
                            pipeline_state.pipeline,
                        );
                    }
                },
                .compute => |pipeline| {
                    if (buffer.in_render_pass) {
                        return error.CommandBufferState;
                    }
                    const pipeline_state = self.get(
                        pipeline,
                        .pipeline_compute,
                    ) orelse return error.InvalidResource;
                    buffer.pipeline = pipeline.handle;
                    buffer.pipeline_layout = pipeline_state.layout;
                    buffer.pipeline_kind = .compute;
                    if (primary) {
                        self.device_api.cmdBindPipeline(
                            buffer.native,
                            .compute,
                            pipeline_state.pipeline,
                        );
                    }
                },
            }
        },

        .bind_descriptor_set => |bind| {
            try requireGraphicsOrCompute(buffer);
            const layout_handle = buffer.pipeline_layout orelse
                return error.CommandBufferState;
            const layout = self.get(layout_handle, .pipeline_layout) orelse
                return error.InvalidResource;
            if (@as(usize, bind.index) >= layout.sets.len) {
                return error.InvalidDesc;
            }
            const set = self.get(bind.set, .descriptor_set) orelse
                return error.InvalidResource;
            if (!root.resourceEql(set.layout, layout.sets[bind.index])) {
                return error.InvalidDesc;
            }
            if (primary) {
                const native_layout = layout.layout;
                const native_sets = [_]vk.DescriptorSet{set.set};
                self.device_api.cmdBindDescriptorSets(
                    buffer.native,
                    buffer.pipeline_kind orelse
                        return error.CommandBufferState,
                    native_layout,
                    bind.index,
                    &native_sets,
                    null,
                );
            }
        },

        .bind_vertex_buffers => |bind| {
            try requireGraphics(buffer);
            if (bind.buffers.len > 32) return error.InvalidDesc;
            if (bind.first > self.limits.max_vertex_buffers or
                bind.buffers.len >
                    self.limits.max_vertex_buffers - bind.first)
            {
                return error.InvalidDesc;
            }

            var natives: [32]vk.Buffer = undefined;
            var offsets: [32]vk.DeviceSize = undefined;
            for (bind.buffers, 0..) |binding, i| {
                const vertex = self.get(binding.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (!vertex.usage.vertex) return error.InvalidDesc;
                if (!root.fits(binding.offset, 1, vertex.size)) {
                    return error.InvalidDesc;
                }
                natives[i] = vertex.buffer;
                offsets[i] = binding.offset;
            }
            if (primary) {
                self.device_api.cmdBindVertexBuffers(
                    buffer.native,
                    bind.first,
                    natives[0..bind.buffers.len],
                    offsets[0..bind.buffers.len],
                );
            }
        },

        .bind_index_buffer => |bind| {
            try requireGraphics(buffer);
            const index = self.get(bind.buffer, .buffer) orelse
                return error.InvalidResource;
            if (!index.usage.index) return error.InvalidDesc;
            const index_size: u64 = switch (bind.format) {
                .u16 => 2,
                .u32 => 4,
            };
            if (bind.offset % index_size != 0) {
                return error.InvalidDesc;
            }
            if (!root.fits(bind.offset, 1, index.size)) {
                return error.InvalidDesc;
            }
            buffer.index_buffer = true;
            if (primary) {
                self.device_api.cmdBindIndexBuffer(
                    buffer.native,
                    index.buffer,
                    bind.offset,
                    switch (bind.format) {
                        .u16 => .uint16,
                        .u32 => .uint32,
                    },
                );
            }
        },

        .push_constants => |push| {
            try requireGraphicsOrCompute(buffer);
            if (format.stageFlagsEmpty(push.stages)) {
                return error.InvalidDesc;
            }
            if (push.data.len == 0) return error.InvalidDesc;
            const layout_handle = buffer.pipeline_layout orelse
                return error.CommandBufferState;
            const layout = self.get(layout_handle, .pipeline_layout) orelse
                return error.InvalidResource;
            if (!pushConstantsCover(layout.push_constants, push)) {
                return error.InvalidDesc;
            }
            if (primary) {
                self.device_api.cmdPushConstants(
                    buffer.native,
                    layout.layout,
                    format.stageFlagsToVk(push.stages),
                    push.offset,
                    @intCast(push.data.len),
                    @ptrCast(push.data.ptr),
                );
            }
        },

        .set_viewport => |viewport| {
            try requireGraphics(buffer);
            if (viewport.width <= 0 or viewport.height <= 0) {
                return error.InvalidDesc;
            }
            buffer.viewport_set = true;
            if (primary) {
                const native = vk.Viewport{
                    .x = viewport.x,
                    .y = viewport.y,
                    .width = viewport.width,
                    .height = viewport.height,
                    .min_depth = viewport.depth_min,
                    .max_depth = viewport.depth_max,
                };
                self.device_api.cmdSetViewport(
                    buffer.native,
                    0,
                    &.{native},
                );
            }
        },

        .set_scissor => |scissor| {
            try requireGraphics(buffer);
            buffer.scissor_set = true;
            if (primary) {
                const native = vk.Rect2D{
                    .offset = .{ .x = scissor.x, .y = scissor.y },
                    .extent = .{
                        .width = scissor.width,
                        .height = scissor.height,
                    },
                };
                self.device_api.cmdSetScissor(
                    buffer.native,
                    0,
                    &.{native},
                );
            }
        },

        .draw => |draw| {
            try requireDraw(buffer);
            buffer.kinds.graphics = true;
            if (primary) {
                self.device_api.cmdDraw(
                    buffer.native,
                    draw.vertices.count,
                    draw.instances.count,
                    draw.vertices.first,
                    draw.instances.first,
                );
            }
        },

        .draw_indexed => |draw| {
            try requireDraw(buffer);
            if (!buffer.index_buffer) return error.InvalidDesc;
            buffer.kinds.graphics = true;
            if (primary) {
                self.device_api.cmdDrawIndexed(
                    buffer.native,
                    draw.indices.count,
                    draw.instances.count,
                    draw.indices.first,
                    draw.base_vertex,
                    draw.instances.first,
                );
            }
        },

        .draw_indirect => |draw| {
            try requireDraw(buffer);
            const indirect = self.get(draw.buffer, .buffer) orelse
                return error.InvalidResource;
            if (!indirectUsage(indirect)) return error.InvalidDesc;
            const stride = if (draw.stride == 0) 16 else draw.stride;
            if (stride < 16) return error.InvalidDesc;
            const total = @as(u64, draw.count) * stride;
            if (!root.fits(draw.offset, total, indirect.size)) {
                return error.InvalidDesc;
            }
            buffer.kinds.graphics = true;
            if (primary and draw.count != 0) {
                self.device_api.cmdDrawIndirect(
                    buffer.native,
                    indirect.buffer,
                    draw.offset,
                    draw.count,
                    stride,
                );
            }
        },

        .draw_indexed_indirect => |draw| {
            try requireDraw(buffer);
            if (!buffer.index_buffer) return error.InvalidDesc;
            const indirect = self.get(draw.buffer, .buffer) orelse
                return error.InvalidResource;
            if (!indirectUsage(indirect)) return error.InvalidDesc;
            const stride = if (draw.stride == 0) 20 else draw.stride;
            if (stride < 20) return error.InvalidDesc;
            const total = @as(u64, draw.count) * stride;
            if (!root.fits(draw.offset, total, indirect.size)) {
                return error.InvalidDesc;
            }
            buffer.kinds.graphics = true;
            if (primary and draw.count != 0) {
                self.device_api.cmdDrawIndexedIndirect(
                    buffer.native,
                    indirect.buffer,
                    draw.offset,
                    draw.count,
                    stride,
                );
            }
        },

        .execute_commands => |execute| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            if (buffer.level != .primary) return error.InvalidDesc;
            for (execute.commands) |secondary| {
                const entry = self.lookup(secondary) orelse
                    return error.InvalidResource;
                if (entry.kind != .command_buffer) {
                    return error.InvalidResource;
                }
                const child = &entry.data.command_buffer;
                if (child.level != .secondary) return error.InvalidDesc;
                if (child.phase != .ended) {
                    return error.CommandBufferState;
                }
                buffer.kinds.graphics =
                    buffer.kinds.graphics or child.kinds.graphics;
                buffer.kinds.compute =
                    buffer.kinds.compute or child.kinds.compute;
                buffer.kinds.transfer =
                    buffer.kinds.transfer or child.kinds.transfer;

                for (child.stored.items) |stored| {
                    try recordInternal(
                        self,
                        handle,
                        &stored.cmd,
                        ret_addr,
                        true,
                    );
                }
            }
        },

        .dispatch => |dispatch| {
            try requireCompute(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            buffer.kinds.compute = true;
            if (primary) {
                self.device_api.cmdDispatch(
                    buffer.native,
                    dispatch.groups[0],
                    dispatch.groups[1],
                    dispatch.groups[2],
                );
            }
        },

        .dispatch_indirect => |dispatch| {
            try requireCompute(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const indirect = self.get(dispatch.buffer, .buffer) orelse
                return error.InvalidResource;
            if (!indirectUsage(indirect)) return error.InvalidDesc;
            if (!root.fits(dispatch.offset, 12, indirect.size)) {
                return error.InvalidDesc;
            }
            buffer.kinds.compute = true;
            if (primary) {
                self.device_api.cmdDispatchIndirect(
                    buffer.native,
                    indirect.buffer,
                    dispatch.offset,
                );
            }
        },

        .copy_buffer => |copy| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(copy.src, .buffer) orelse
                return error.InvalidResource;
            const dst = self.get(copy.dst, .buffer) orelse
                return error.InvalidResource;
            if (!src.usage.transfer_src or !dst.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (copy.regions.len == 0) return error.InvalidDesc;
            for (copy.regions) |region| {
                if (region.size == 0) return error.InvalidDesc;
                if (region.src_offset % 4 != 0 or
                    region.dst_offset % 4 != 0 or
                    region.size % 4 != 0)
                {
                    return error.InvalidDesc;
                }
                if (!root.fits(region.src_offset, region.size, src.size)) {
                    return error.InvalidDesc;
                }
                if (!root.fits(region.dst_offset, region.size, dst.size)) {
                    return error.InvalidDesc;
                }
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try copyBuffer(self, buffer, copy, src, dst);
            }
        },

        .copy_buffer_to_texture => |copy| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(copy.src, .buffer) orelse
                return error.InvalidResource;
            const dst = self.get(copy.dst, .texture) orelse
                return error.InvalidResource;
            if (!src.usage.transfer_src or !dst.desc.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (dst.desc.samples != .x1) return error.InvalidDesc;
            if (copy.regions.len == 0) return error.InvalidDesc;
            for (copy.regions) |region| {
                try validateTextureRegion(dst.desc, region.region);
                const texel = format.bytesPerTexel(dst.desc.format);
                if (region.buffer_offset % @max(texel, 4) != 0) {
                    return error.InvalidDesc;
                }
                if (region.buffer_offset >= src.size) {
                    return error.InvalidDesc;
                }
                if (region.buffer_stride != 0 and
                    region.buffer_stride %
                        format.bytesPerTexel(dst.desc.format) != 0)
                {
                    return error.InvalidDesc;
                }
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try copyBufferToTexture(self, buffer, copy, src, dst);
            }
        },

        .copy_texture_to_buffer => |copy| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(copy.src, .texture) orelse
                return error.InvalidResource;
            const dst = self.get(copy.dst, .buffer) orelse
                return error.InvalidResource;
            if (!src.desc.usage.transfer_src or !dst.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (src.desc.samples != .x1) return error.InvalidDesc;
            if (copy.regions.len == 0) return error.InvalidDesc;
            for (copy.regions) |region| {
                try validateTextureRegion(src.desc, region.region);
                const texel = format.bytesPerTexel(src.desc.format);
                if (region.buffer_offset % @max(texel, 4) != 0) {
                    return error.InvalidDesc;
                }
                if (region.buffer_offset >= dst.size) {
                    return error.InvalidDesc;
                }
                if (region.buffer_stride != 0 and
                    region.buffer_stride %
                        format.bytesPerTexel(src.desc.format) != 0)
                {
                    return error.InvalidDesc;
                }
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try copyTextureToBuffer(self, buffer, copy, src, dst);
            }
        },

        .copy_texture => |copy| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(copy.src, .texture) orelse
                return error.InvalidResource;
            const dst = self.get(copy.dst, .texture) orelse
                return error.InvalidResource;
            if (!src.desc.usage.transfer_src or !dst.desc.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (src.desc.format != dst.desc.format) {
                return error.InvalidDesc;
            }
            if (src.desc.samples != .x1 or dst.desc.samples != .x1) {
                return error.InvalidDesc;
            }
            if (copy.regions.len == 0) return error.InvalidDesc;
            for (copy.regions) |region| {
                try validateTextureRegion(src.desc, region.src);
                try validateTextureRegion(dst.desc, region.dst);
                if (!std.meta.eql(region.src.extent, region.dst.extent)) {
                    return error.InvalidDesc;
                }
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try copyTexture(self, buffer, copy, src, dst);
            }
        },

        .blit_texture => |blit| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(blit.src, .texture) orelse
                return error.InvalidResource;
            const dst = self.get(blit.dst, .texture) orelse
                return error.InvalidResource;
            if (!src.desc.usage.transfer_src or !dst.desc.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (src.desc.samples != .x1 or dst.desc.samples != .x1) {
                return error.InvalidDesc;
            }
            if (blit.regions.len == 0) return error.InvalidDesc;
            for (blit.regions) |region| {
                try validateTextureRegion(src.desc, region.src);
                try validateTextureRegion(dst.desc, region.dst);
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try blitTexture(self, buffer, blit, src, dst);
            }
        },

        .resolve_texture => |resolve| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const src = self.get(resolve.src, .texture) orelse
                return error.InvalidResource;
            const dst = self.get(resolve.dst, .texture) orelse
                return error.InvalidResource;
            if (!src.desc.usage.transfer_src or !dst.desc.usage.transfer_dst) {
                return error.InvalidDesc;
            }
            if (src.desc.samples == .x1) return error.InvalidDesc;
            if (dst.desc.samples != .x1) return error.InvalidDesc;
            if (src.desc.format != dst.desc.format) {
                return error.InvalidDesc;
            }
            if (resolve.regions.len == 0) return error.InvalidDesc;
            for (resolve.regions) |region| {
                try validateTextureRegion(src.desc, region.src);
                try validateTextureRegion(dst.desc, region.dst);
            }
            buffer.kinds.transfer = true;
            if (primary) {
                try resolveTexture(self, buffer, resolve, src, dst);
            }
        },

        .fill_buffer => |fill| {
            try requireRecording(buffer);
            if (buffer.in_render_pass) return error.CommandBufferState;
            const target = self.get(fill.buffer, .buffer) orelse
                return error.InvalidResource;
            if (!target.usage.transfer_dst) return error.InvalidDesc;
            const size: u64 = if (fill.size == 0)
                target.size -| fill.offset
            else
                fill.size;
            if (fill.offset % 4 != 0) return error.InvalidDesc;
            if (fill.size != 0 and fill.size % 4 != 0) {
                return error.InvalidDesc;
            }
            if (!root.fits(fill.offset, size, target.size)) {
                return error.InvalidDesc;
            }
            buffer.kinds.transfer = true;
            if (primary) {
                self.device_api.cmdFillBuffer(
                    buffer.native,
                    target.buffer,
                    fill.offset,
                    if (fill.size == 0) vk.WHOLE_SIZE else fill.size,
                    fill.value,
                );
            }
        },

        .barrier => |barrier| {
            try requireRecording(buffer);
            try recordBarrier(self, buffer, barrier, primary);
        },
    }
}

fn indirectUsage(buffer: *state.BufferState) bool {
    return buffer.usage.indirect;
}

fn stageFlagsIntersect(
    a: rhi.Resource.Shader.StageFlags,
    b: rhi.Resource.Shader.StageFlags,
) bool {
    return (a.vertex and b.vertex) or
        (a.fragment and b.fragment) or
        (a.compute and b.compute);
}

fn pushConstantsCover(
    ranges: []const PipelineLayout.PushConstantRange,
    push: Device.Command.PushConstants,
) bool {
    const end = @as(u64, push.offset) + push.data.len;
    for (ranges) |range| {
        const range_end = @as(u64, range.offset) + range.size;
        if (push.offset >= range.offset and
            end <= range_end and
            stageFlagsIntersect(range.stages, push.stages))
        {
            return true;
        }
    }
    return false;
}

fn validateTextureRegion(
    desc: Texture.Desc,
    region: Device.Command.TextureRegion,
) Device.RecordError!void {
    if (region.extent[0] == 0 or region.extent[1] == 0 or
        region.extent[2] == 0)
    {
        return error.InvalidDesc;
    }
    if (region.mip >= desc.mips) return error.InvalidDesc;
    if (region.layer >= desc.layers) return error.InvalidDesc;

    const shift: u5 = @intCast(@min(region.mip, 31));
    const mip_width = @max(@as(u32, 1), desc.extent.width >> shift);
    const mip_height = @max(@as(u32, 1), desc.extent.height >> shift);
    const mip_depth = @max(@as(u32, 1), desc.extent.depth >> shift);

    if (!root.fits(region.offset[0], region.extent[0], mip_width)) {
        return error.InvalidDesc;
    }
    if (!root.fits(region.offset[1], region.extent[1], mip_height)) {
        return error.InvalidDesc;
    }
    if (!root.fits(region.offset[2], region.extent[2], mip_depth)) {
        return error.InvalidDesc;
    }
}

fn validateSubresourceRange(
    desc: Texture.Desc,
    range: Texture.SubresourceRange,
) Device.RecordError!void {
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

fn validateRenderPass(
    self: *Backend,
    pass: Device.Command.RenderPass,
) Device.RecordError!void {
    if (pass.area.width == 0 or pass.area.height == 0) {
        return error.InvalidDesc;
    }
    for (pass.color) |attachment| {
        const texture = self.get(attachment.texture, .texture) orelse
            return error.InvalidResource;
        if (!texture.desc.usage.color_attachment) {
            return error.InvalidDesc;
        }
        if (!areaFits(pass.area, texture.desc.extent)) {
            return error.InvalidDesc;
        }
        if (attachment.resolve) |resolve| {
            const resolved = self.get(resolve, .texture) orelse
                return error.InvalidResource;
            if (!resolved.desc.usage.color_attachment) {
                return error.InvalidDesc;
            }
            if (!areaFits(pass.area, resolved.desc.extent)) {
                return error.InvalidDesc;
            }
        }
    }
    if (pass.depth_stencil) |attachment| {
        const texture = self.get(attachment.texture, .texture) orelse
            return error.InvalidResource;
        if (!texture.desc.usage.depth_stencil_attachment) {
            return error.InvalidDesc;
        }
        if (!areaFits(pass.area, texture.desc.extent)) {
            return error.InvalidDesc;
        }
    }
}

fn transition(
    self: *Backend,
    buffer: *state.CommandBufferState,
    texture: *state.TextureState,
    layout: vk.ImageLayout,
    stage: vk.PipelineStageFlags,
    access: vk.AccessFlags,
) void {
    if (texture.layout == layout) return;

    const old_layout = texture.layout;
    const needs_content = old_layout != .undefined and
        old_layout != .preinitialized;

    const barrier = vk.ImageMemoryBarrier{
        .src_access_mask = if (needs_content)
            .{ .memory_read = true, .memory_write = true }
        else
            .{},
        .dst_access_mask = access,
        .old_layout = old_layout,
        .new_layout = layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresource_range = format.fullSubresourceRange(texture.desc),
    };

    self.device_api.cmdPipelineBarrier(
        buffer.native,
        .{ .all_commands = true },
        stage,
        .{},
        null,
        null,
        &.{barrier},
    );
    texture.layout = layout;
}

fn areaFits(
    area: Device.Command.Rect,
    extent: Resource.Extent,
) bool {
    if (area.x < 0 or area.y < 0) return false;
    const right = @as(u64, @intCast(area.x)) + area.width;
    const bottom = @as(u64, @intCast(area.y)) + area.height;
    return right <= extent.width and bottom <= extent.height;
}

fn beginRenderPass(
    self: *Backend,
    buffer: *state.CommandBufferState,
    pass: Device.Command.RenderPass,
) Device.RecordError!void {
    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const colors = try scratch.alloc(
        vk.RenderingAttachmentInfo,
        pass.color.len,
    );

    for (pass.color, 0..) |attachment, i| {
        const texture = self.get(attachment.texture, .texture) orelse
            return error.InvalidResource;
        if (!texture.desc.usage.color_attachment) {
            return error.InvalidDesc;
        }
        if (!areaFits(pass.area, texture.desc.extent)) {
            return error.InvalidDesc;
        }

        transition(
            self,
            buffer,
            texture,
            .color_attachment_optimal,
            .{ .color_attachment_output = true },
            .{
                .color_attachment_read = true,
                .color_attachment_write = true,
            },
        );

        colors[i] = .{
            .image_view = texture.view,
            .image_layout = .color_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_view = .null,
            .resolve_image_layout = .undefined,
            .load_op = loadOp(attachment.load),
            .store_op = storeOp(attachment.store),
            .clear_value = clearColor(attachment.clear),
        };

        if (attachment.resolve) |resolve| {
            const resolved = self.get(resolve, .texture) orelse
                return error.InvalidResource;
            if (!resolved.desc.usage.color_attachment) {
                return error.InvalidDesc;
            }
            if (!areaFits(pass.area, resolved.desc.extent)) {
                return error.InvalidDesc;
            }
            transition(
                self,
                buffer,
                resolved,
                .color_attachment_optimal,
                .{ .color_attachment_output = true },
                .{
                    .color_attachment_read = true,
                    .color_attachment_write = true,
                },
            );
            colors[i].resolve_mode = if (format.isIntegerFormat(
                texture.desc.format,
            ))
                .{ .sample_zero = true }
            else
                .{ .average = true };
            colors[i].resolve_image_view = resolved.view;
            colors[i].resolve_image_layout =
                .color_attachment_optimal;
        }
    }

    var depth_info = emptyAttachment();
    var stencil_info = emptyAttachment();
    var has_depth = false;
    var has_stencil = false;
    if (pass.depth_stencil) |attachment| {
        const texture = self.get(attachment.texture, .texture) orelse
            return error.InvalidResource;
        if (!texture.desc.usage.depth_stencil_attachment) {
            return error.InvalidDesc;
        }
        if (!areaFits(pass.area, texture.desc.extent)) {
            return error.InvalidDesc;
        }

        transition(
            self,
            buffer,
            texture,
            .depth_stencil_attachment_optimal,
            .{
                .early_fragment_tests = true,
                .late_fragment_tests = true,
            },
            .{
                .depth_stencil_attachment_read = true,
                .depth_stencil_attachment_write = true,
            },
        );

        const clear_value = vk.ClearValue{
            .depth_stencil = .{
                .depth = attachment.depth_clear,
                .stencil = attachment.stencil_clear,
            },
        };
        const aspect = format.formatAspect(texture.desc.format);
        if (aspect.depth) {
            has_depth = true;
            depth_info = emptyAttachment();
            depth_info.image_view = texture.view;
            depth_info.image_layout =
                .depth_stencil_attachment_optimal;
            depth_info.load_op = loadOp(attachment.load);
            depth_info.store_op = storeOp(attachment.store);
            depth_info.clear_value = clear_value;
        }
        if (aspect.stencil) {
            has_stencil = true;
            stencil_info = emptyAttachment();
            stencil_info.image_view = texture.stencil_view orelse
                texture.view;
            stencil_info.image_layout =
                .depth_stencil_attachment_optimal;
            stencil_info.load_op = loadOp(attachment.load);
            stencil_info.store_op = storeOp(attachment.store);
            stencil_info.clear_value = clear_value;
        }
    }

    const info = vk.RenderingInfo{
        .render_area = .{
            .offset = .{ .x = pass.area.x, .y = pass.area.y },
            .extent = .{
                .width = pass.area.width,
                .height = pass.area.height,
            },
        },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = @intCast(colors.len),
        .p_color_attachments = if (colors.len == 0) null else colors.ptr,
        .p_depth_attachment = if (has_depth) &depth_info else null,
        .p_stencil_attachment = if (has_stencil) &stencil_info else null,
    };

    if (self.core_dynamic_rendering) {
        self.device_api.cmdBeginRendering(buffer.native, &info);
    } else {
        self.device_api.cmdBeginRenderingKHR(buffer.native, &info);
    }
}

fn emptyAttachment() vk.RenderingAttachmentInfo {
    return .{
        .image_view = .null,
        .image_layout = .undefined,
        .resolve_mode = .{},
        .resolve_image_view = .null,
        .resolve_image_layout = .undefined,
        .load_op = .dont_care,
        .store_op = .dont_care,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 0 } } },
    };
}

fn loadOp(load: Device.Command.Load) vk.AttachmentLoadOp {
    return switch (load) {
        .load => .load,
        .clear => .clear,
        .dont_care => .dont_care,
    };
}

fn storeOp(store: Device.Command.Store) vk.AttachmentStoreOp {
    return switch (store) {
        .store => .store,
        .dont_care => .dont_care,
    };
}

fn clearColor(value: Device.Command.ClearColor) vk.ClearValue {
    return switch (value) {
        .float => |channels| .{
            .color = .{ .float_32 = channels },
        },
        .uint => |channels| .{
            .color = .{ .uint_32 = channels },
        },
        .sint => |channels| .{
            .color = .{ .int_32 = channels },
        },
    };
}

fn copyBuffer(
    self: *Backend,
    buffer: *state.CommandBufferState,
    copy: Device.Command.CopyBuffer,
    src: *state.BufferState,
    dst: *state.BufferState,
) Device.RecordError!void {
    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(vk.BufferCopy, copy.regions.len);
    for (copy.regions, 0..) |region, i| {
        regions[i] = .{
            .src_offset = region.src_offset,
            .dst_offset = region.dst_offset,
            .size = region.size,
        };
    }

    self.device_api.cmdCopyBuffer(
        buffer.native,
        src.buffer,
        dst.buffer,
        regions,
    );
}

fn copyBufferToTexture(
    self: *Backend,
    buffer: *state.CommandBufferState,
    copy: Device.Command.CopyBufferTexture,
    src: *state.BufferState,
    dst: *state.TextureState,
) Device.RecordError!void {
    transition(
        self,
        buffer,
        dst,
        .transfer_dst_optimal,
        .{ .transfer = true },
        .{ .transfer_write = true },
    );

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(
        vk.BufferImageCopy,
        copy.regions.len,
    );
    const texel = format.bytesPerTexel(dst.desc.format);
    for (copy.regions, 0..) |region, i| {
        regions[i] = .{
            .buffer_offset = region.buffer_offset,
            .buffer_row_length = if (region.buffer_stride == 0)
                0
            else
                region.buffer_stride / texel,
            .buffer_image_height = 0,
            .image_subresource = .{
                .aspect_mask = format.primaryAspect(dst.desc.format),
                .mip_level = region.region.mip,
                .base_array_layer = region.region.layer,
                .layer_count = 1,
            },
            .image_offset = .{
                .x = @intCast(region.region.offset[0]),
                .y = @intCast(region.region.offset[1]),
                .z = @intCast(region.region.offset[2]),
            },
            .image_extent = .{
                .width = region.region.extent[0],
                .height = region.region.extent[1],
                .depth = region.region.extent[2],
            },
        };
    }

    self.device_api.cmdCopyBufferToImage(
        buffer.native,
        src.buffer,
        dst.image,
        .transfer_dst_optimal,
        regions,
    );
}

fn copyTextureToBuffer(
    self: *Backend,
    buffer: *state.CommandBufferState,
    copy: Device.Command.CopyTextureBuffer,
    src: *state.TextureState,
    dst: *state.BufferState,
) Device.RecordError!void {
    transition(
        self,
        buffer,
        src,
        .transfer_src_optimal,
        .{ .transfer = true },
        .{ .transfer_read = true },
    );

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(
        vk.BufferImageCopy,
        copy.regions.len,
    );
    const texel = format.bytesPerTexel(src.desc.format);
    for (copy.regions, 0..) |region, i| {
        regions[i] = .{
            .buffer_offset = region.buffer_offset,
            .buffer_row_length = if (region.buffer_stride == 0)
                0
            else
                region.buffer_stride / texel,
            .buffer_image_height = 0,
            .image_subresource = .{
                .aspect_mask = format.primaryAspect(src.desc.format),
                .mip_level = region.region.mip,
                .base_array_layer = region.region.layer,
                .layer_count = 1,
            },
            .image_offset = .{
                .x = @intCast(region.region.offset[0]),
                .y = @intCast(region.region.offset[1]),
                .z = @intCast(region.region.offset[2]),
            },
            .image_extent = .{
                .width = region.region.extent[0],
                .height = region.region.extent[1],
                .depth = region.region.extent[2],
            },
        };
    }

    self.device_api.cmdCopyImageToBuffer(
        buffer.native,
        src.image,
        .transfer_src_optimal,
        dst.buffer,
        regions,
    );
}

fn copyTexture(
    self: *Backend,
    buffer: *state.CommandBufferState,
    copy: Device.Command.CopyTexture,
    src: *state.TextureState,
    dst: *state.TextureState,
) Device.RecordError!void {
    transition(
        self,
        buffer,
        src,
        .transfer_src_optimal,
        .{ .transfer = true },
        .{ .transfer_read = true },
    );
    transition(
        self,
        buffer,
        dst,
        .transfer_dst_optimal,
        .{ .transfer = true },
        .{ .transfer_write = true },
    );

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(vk.ImageCopy, copy.regions.len);
    for (copy.regions, 0..) |region, i| {
        regions[i] = .{
            .src_subresource = .{
                .aspect_mask = format.primaryAspect(src.desc.format),
                .mip_level = region.src.mip,
                .base_array_layer = region.src.layer,
                .layer_count = 1,
            },
            .src_offset = .{
                .x = @intCast(region.src.offset[0]),
                .y = @intCast(region.src.offset[1]),
                .z = @intCast(region.src.offset[2]),
            },
            .dst_subresource = .{
                .aspect_mask = format.primaryAspect(dst.desc.format),
                .mip_level = region.dst.mip,
                .base_array_layer = region.dst.layer,
                .layer_count = 1,
            },
            .dst_offset = .{
                .x = @intCast(region.dst.offset[0]),
                .y = @intCast(region.dst.offset[1]),
                .z = @intCast(region.dst.offset[2]),
            },
            .extent = .{
                .width = region.src.extent[0],
                .height = region.src.extent[1],
                .depth = region.src.extent[2],
            },
        };
    }

    self.device_api.cmdCopyImage(
        buffer.native,
        src.image,
        .transfer_src_optimal,
        dst.image,
        .transfer_dst_optimal,
        regions,
    );
}

fn blitTexture(
    self: *Backend,
    buffer: *state.CommandBufferState,
    blit: Device.Command.BlitTexture,
    src: *state.TextureState,
    dst: *state.TextureState,
) Device.RecordError!void {
    transition(
        self,
        buffer,
        src,
        .transfer_src_optimal,
        .{ .transfer = true },
        .{ .transfer_read = true },
    );
    transition(
        self,
        buffer,
        dst,
        .transfer_dst_optimal,
        .{ .transfer = true },
        .{ .transfer_write = true },
    );

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(vk.ImageBlit, blit.regions.len);
    for (blit.regions, 0..) |region, i| {
        regions[i] = .{
            .src_subresource = .{
                .aspect_mask = format.primaryAspect(src.desc.format),
                .mip_level = region.src.mip,
                .base_array_layer = region.src.layer,
                .layer_count = 1,
            },
            .src_offsets = .{
                .{
                    .x = @intCast(region.src.offset[0]),
                    .y = @intCast(region.src.offset[1]),
                    .z = @intCast(region.src.offset[2]),
                },
                .{
                    .x = @intCast(region.src.offset[0] +
                        region.src.extent[0]),
                    .y = @intCast(region.src.offset[1] +
                        region.src.extent[1]),
                    .z = @intCast(region.src.offset[2] +
                        region.src.extent[2]),
                },
            },
            .dst_subresource = .{
                .aspect_mask = format.primaryAspect(dst.desc.format),
                .mip_level = region.dst.mip,
                .base_array_layer = region.dst.layer,
                .layer_count = 1,
            },
            .dst_offsets = .{
                .{
                    .x = @intCast(region.dst.offset[0]),
                    .y = @intCast(region.dst.offset[1]),
                    .z = @intCast(region.dst.offset[2]),
                },
                .{
                    .x = @intCast(region.dst.offset[0] +
                        region.dst.extent[0]),
                    .y = @intCast(region.dst.offset[1] +
                        region.dst.extent[1]),
                    .z = @intCast(region.dst.offset[2] +
                        region.dst.extent[2]),
                },
            },
        };
    }

    self.device_api.cmdBlitImage(
        buffer.native,
        src.image,
        .transfer_src_optimal,
        dst.image,
        .transfer_dst_optimal,
        regions,
        format.filterToVk(blit.filter),
    );
}

fn resolveTexture(
    self: *Backend,
    buffer: *state.CommandBufferState,
    resolve: Device.Command.ResolveTexture,
    src: *state.TextureState,
    dst: *state.TextureState,
) Device.RecordError!void {
    transition(
        self,
        buffer,
        src,
        .transfer_src_optimal,
        .{ .transfer = true },
        .{ .transfer_read = true },
    );
    transition(
        self,
        buffer,
        dst,
        .transfer_dst_optimal,
        .{ .transfer = true },
        .{ .transfer_write = true },
    );

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const regions = try scratch.alloc(
        vk.ImageResolve,
        resolve.regions.len,
    );
    for (resolve.regions, 0..) |region, i| {
        regions[i] = .{
            .src_subresource = .{
                .aspect_mask = format.primaryAspect(src.desc.format),
                .mip_level = region.src.mip,
                .base_array_layer = region.src.layer,
                .layer_count = 1,
            },
            .src_offset = .{
                .x = @intCast(region.src.offset[0]),
                .y = @intCast(region.src.offset[1]),
                .z = @intCast(region.src.offset[2]),
            },
            .dst_subresource = .{
                .aspect_mask = format.primaryAspect(dst.desc.format),
                .mip_level = region.dst.mip,
                .base_array_layer = region.dst.layer,
                .layer_count = 1,
            },
            .dst_offset = .{
                .x = @intCast(region.dst.offset[0]),
                .y = @intCast(region.dst.offset[1]),
                .z = @intCast(region.dst.offset[2]),
            },
            .extent = .{
                .width = region.src.extent[0],
                .height = region.src.extent[1],
                .depth = region.src.extent[2],
            },
        };
    }

    self.device_api.cmdResolveImage(
        buffer.native,
        src.image,
        .transfer_src_optimal,
        dst.image,
        .transfer_dst_optimal,
        regions,
    );
}

fn recordBarrier(
    self: *Backend,
    buffer: *state.CommandBufferState,
    barrier: Device.Command.Barrier,
    primary: bool,
) Device.RecordError!void {
    for (barrier.buffers) |item| {
        const target = self.get(item.buffer, .buffer) orelse
            return error.InvalidResource;
        const size: u64 = if (item.size == 0)
            target.size -| item.offset
        else
            item.size;
        if (!root.fits(item.offset, size, target.size)) {
            return error.InvalidDesc;
        }
    }
    for (barrier.textures) |item| {
        const target = self.get(item.texture, .texture) orelse
            return error.InvalidResource;
        try validateSubresourceRange(target.desc, item.range);
    }

    if (!primary) return;

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const buffer_barriers = try scratch.alloc(
        vk.BufferMemoryBarrier,
        barrier.buffers.len,
    );
    for (barrier.buffers, 0..) |item, i| {
        const target = self.get(item.buffer, .buffer).?;
        buffer_barriers[i] = .{
            .src_access_mask = format.accessToVk(item.src.access),
            .dst_access_mask = format.accessToVk(item.dst.access),
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = target.buffer,
            .offset = item.offset,
            .size = if (item.size == 0) vk.WHOLE_SIZE else item.size,
        };
    }

    const image_barriers = try scratch.alloc(
        vk.ImageMemoryBarrier,
        barrier.textures.len,
    );
    for (barrier.textures, 0..) |item, i| {
        const target = self.get(item.texture, .texture).?;
        const old_layout = target.layout;
        const needs_content = old_layout != .undefined and
            old_layout != .preinitialized;
        const new_layout = format.layoutToVk(item.layout);
        image_barriers[i] = .{
            .src_access_mask = if (needs_content)
                format.accessToVk(item.src.access)
            else
                .{},
            .dst_access_mask = if (new_layout == .undefined)
                .{}
            else
                format.accessToVk(item.dst.access),
            .old_layout = old_layout,
            .new_layout = new_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = target.image,
            .subresource_range = format.subresourceRangeToVk(
                target.desc,
                item.range,
            ),
        };
        target.layout = new_layout;
    }

    if (buffer_barriers.len == 0 and image_barriers.len == 0) {
        return;
    }

    var src_stage = format.stagesToVk(barrierStageUnion(
        barrier.buffers,
        barrier.textures,
        true,
    ));
    var dst_stage = format.stagesToVk(barrierStageUnion(
        barrier.buffers,
        barrier.textures,
        false,
    ));
    if (src_stage.toInt() == 0) src_stage = .{ .all_commands = true };
    if (dst_stage.toInt() == 0) dst_stage = .{ .all_commands = true };

    self.device_api.cmdPipelineBarrier(
        buffer.native,
        src_stage,
        dst_stage,
        .{},
        null,
        buffer_barriers,
        image_barriers,
    );
}

fn barrierStageUnion(
    buffers: []const Device.Command.BufferBarrier,
    textures: []const Device.Command.TextureBarrier,
    source: bool,
) Device.Command.Stage {
    var result = Device.Command.Stage{};
    for (buffers) |item| {
        const sync = if (source) item.src else item.dst;
        mergeStage(&result, sync.stages);
    }
    for (textures) |item| {
        const sync = if (source) item.src else item.dst;
        mergeStage(&result, sync.stages);
    }
    return result;
}

fn mergeStage(
    into: *Device.Command.Stage,
    from: Device.Command.Stage,
) void {
    into.all = into.all or from.all;
    into.vertex = into.vertex or from.vertex;
    into.fragment = into.fragment or from.fragment;
    into.compute = into.compute or from.compute;
    into.transfer = into.transfer or from.transfer;
    into.host = into.host or from.host;
    into.color_attachment =
        into.color_attachment or from.color_attachment;
    into.depth_stencil = into.depth_stencil or from.depth_stencil;
}

fn cloneCommand(
    allocator: std.mem.Allocator,
    cmd: Device.Command,
) std.mem.Allocator.Error!Device.Command {
    var copy = cmd;
    switch (copy) {
        inline else => |*payload| {
            payload.* = try clonePayload(
                allocator,
                @TypeOf(payload.*),
                payload.*,
            );
        },
    }
    return copy;
}

fn clonePayload(
    allocator: std.mem.Allocator,
    comptime T: type,
    value: T,
) std.mem.Allocator.Error!T {
    var copy = value;
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, 0..) |name, i| {
                const Field = info.field_types[i];
                switch (@typeInfo(Field)) {
                    .pointer => |pointer| {
                        if (pointer.size == .slice) {
                            @field(copy, name) = try allocator.dupe(
                                pointer.child,
                                @field(value, name),
                            );
                        }
                    },
                    else => {},
                }
            }
        },
        else => {},
    }
    return copy;
}

fn recordError(err: anyerror) Device.RecordError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.DeviceLost => error.DeviceLost,
        else => error.DeviceFailure,
    };
}
