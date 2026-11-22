//! Render Hardware Interface (RHI).
//!
//! The RHI is a closed, backend-agnostic description of
//! the GPU operations this engine needs. A backend owns
//! all native state; the rest of the engine only sees
//! values and tagged handles.
//!
//! The interface has four layers:
//!
//! - `Resource.*` wrappers create and destroy typed
//!   handles. Each wrapper stores the owning `Device`.
//! - `Device` dispatches creation, mapping, recording,
//!   queries and immediate operations through a vtable.
//! - `Queue` submits recorded work and synchronises with
//!   the device.
//! - `Resource.Kind`, `Command`, `Feature` and
//!   `DeviceOp` enumerate every supported resource,
//!   command, query and immediate operation.
//!
//! The RHI is exhaustive on purpose. A feature that is
//! not listed is unsupported rather than silently
//! emulated. Extending the interface is a breaking
//! change, so the closed sets only grow when every
//! backend can implement the addition.
//!
//! Handles are values, not pointers. `Resource.handle`
//! is opaque to callers; only the backend that created
//! it may interpret it. Operations pass `ret_addr` so
//! the backend can attribute diagnostics to the public
//! call site.
//!
//! Public functions follow four forms:
//!
//! - `rawX` (or `rawXyz`): a vtable entry taking
//!   `ret_addr` explicitly, marked `inline`.
//! - `x`: a vtable entry capturing `@returnAddress`,
//!   marked `inline`.
//! - `xWithAddr`: a typed wrapper taking `ret_addr`.
//! - `x`: a typed wrapper capturing
//!   `@returnAddress` and forwarding to `xWithAddr`.
//!
//! Prefer the forms without `ret_addr`; use the
//! explicit forms only when forwarding an address from
//! another wrapper. The typed wrappers check handle
//! kinds in safety builds, and a validation layer owns
//! the full checks.

const std = @import("std");

pub const Device = @import("Device.zig");
pub const Queue = @import("Queue.zig");
pub const Resource = @import("Resource.zig");

test {
    _ = Device;
    _ = Queue;
    _ = Resource;
    std.testing.refAllDecls(Device);
    std.testing.refAllDecls(Queue);
    std.testing.refAllDecls(Resource);
}

test "public descriptions and operations type-check" {
    const device: Device = undefined;
    const raw = Resource{ .kind = .buffer, .handle = 1 };

    const buffer = Resource.Buffer{ .handle = raw, .device = device };
    const command_buffer = Resource.CommandBuffer{
        .handle = raw,
        .device = device,
    };
    const command_pool = Resource.CommandPool{
        .handle = raw,
        .device = device,
    };
    const descriptor_set = Resource.DescriptorSet{
        .handle = raw,
        .device = device,
    };
    const descriptor_set_layout = Resource.DescriptorSetLayout{
        .handle = raw,
        .device = device,
    };
    const fence = Resource.Fence{ .handle = raw, .device = device };
    const pipeline_compute = Resource.PipelineCompute{
        .handle = raw,
        .device = device,
    };
    const pipeline_graphics = Resource.PipelineGraphics{
        .handle = raw,
        .device = device,
    };
    const pipeline_layout = Resource.PipelineLayout{
        .handle = raw,
        .device = device,
    };
    const sampler = Resource.Sampler{ .handle = raw, .device = device };
    const semaphore = Resource.Semaphore{
        .handle = raw,
        .device = device,
    };
    const shader = Resource.Shader{ .handle = raw, .device = device };
    const surface = Resource.Surface{ .handle = raw, .device = device };
    const swapchain = Resource.Swapchain{
        .handle = raw,
        .device = device,
    };
    const texture = Resource.Texture{ .handle = raw, .device = device };

    const descs = [_]Resource.Desc{
        .{ .buffer = .{ .size = 16 } },
        .{ .command_buffer = .{ .pool = command_pool } },
        .{ .command_pool = .{} },
        .{ .descriptor_set = .{ .layout = descriptor_set_layout } },
        .{ .descriptor_set_layout = .{} },
        .{ .fence = .{} },
        .{ .pipeline_compute = .{
            .layout = pipeline_layout,
            .shader = shader,
        } },
        .{ .pipeline_graphics = .{
            .layout = pipeline_layout,
            .vertex = shader,
        } },
        .{ .pipeline_layout = .{
            .sets = &.{descriptor_set_layout},
        } },
        .{ .sampler = .{} },
        .{ .semaphore = .{} },
        .{ .shader = .{ .code = &.{}, .stage = .vertex } },
        .{ .surface = .{ .window = 1, .platform = .win32 } },
        .{ .swapchain = .{
            .surface = surface,
            .format = .r8g8b8a8_unorm,
            .extent = .{ .width = 8, .height = 8 },
        } },
        .{ .texture = .{
            .extent = .{ .width = 8 },
            .format = .r8g8b8a8_unorm,
        } },
    };
    _ = descs;

    const region: Device.Command.TextureRegion = .{
        .extent = .{ 1, 1, 1 },
    };
    const copy: Device.Command.TextureCopy = .{
        .src = region,
        .dst = region,
    };

    const commands = [_]Device.Command{
        .{ .begin = .{} },
        .end,
        .{ .begin_render_pass = .{
            .area = .{ .width = 8, .height = 8 },
            .color = &.{.{ .texture = texture }},
            .depth_stencil = .{ .texture = texture },
        } },
        .end_render_pass,
        .{ .bind_pipeline = .{ .graphics = pipeline_graphics } },
        .{ .bind_pipeline = .{ .compute = pipeline_compute } },
        .{ .bind_descriptor_set = .{
            .index = 0,
            .set = descriptor_set,
        } },
        .{ .bind_vertex_buffers = .{
            .buffers = &.{.{ .buffer = buffer }},
        } },
        .{ .bind_index_buffer = .{ .buffer = buffer } },
        .{ .push_constants = .{
            .stages = .{ .vertex = true },
            .data = &.{},
        } },
        .{ .set_viewport = .{ .width = 8, .height = 8 } },
        .{ .set_scissor = .{ .width = 8, .height = 8 } },
        .{ .draw = .{ .vertices = .{ .count = 3 } } },
        .{ .draw_indexed = .{ .indices = .{ .count = 3 } } },
        .{ .draw_indirect = .{ .buffer = buffer, .count = 1 } },
        .{ .draw_indexed_indirect = .{
            .buffer = buffer,
            .count = 1,
        } },
        .{ .execute_commands = .{
            .commands = &.{command_buffer},
        } },
        .{ .dispatch = .{ .groups = .{ 1, 1, 1 } } },
        .{ .dispatch_indirect = .{ .buffer = buffer } },
        .{ .copy_buffer = .{
            .src = buffer,
            .dst = buffer,
            .regions = &.{.{ .size = 4 }},
        } },
        .{ .copy_buffer_to_texture = .{
            .src = buffer,
            .dst = texture,
            .regions = &.{.{ .region = region }},
        } },
        .{ .copy_texture_to_buffer = .{
            .src = texture,
            .dst = buffer,
            .regions = &.{.{ .region = region }},
        } },
        .{ .copy_texture = .{
            .src = texture,
            .dst = texture,
            .regions = &.{copy},
        } },
        .{ .blit_texture = .{
            .src = texture,
            .dst = texture,
            .regions = &.{copy},
        } },
        .{ .resolve_texture = .{
            .src = texture,
            .dst = texture,
            .regions = &.{copy},
        } },
        .{ .fill_buffer = .{ .buffer = buffer } },
        .{ .barrier = .{
            .buffers = &.{.{
                .buffer = buffer,
                .src = .{},
                .dst = .{},
            }},
            .textures = &.{.{
                .texture = texture,
                .src = .{},
                .dst = .{},
            }},
        } },
    };
    _ = commands;

    const features = [_]Device.Feature{
        .{ .format = .{
            .format = .r8g8b8a8_unorm,
            .usage = .{ .sampled = true },
        } },
        .{ .texture = .{
            .extent = .{ .width = 8 },
            .format = .r8g8b8a8_unorm,
        } },
        .{ .swapchain = .{
            .format = .r8g8b8a8_unorm,
            .mode = .fifo,
        } },
        .queue_family,
        .limits,
    };
    _ = features;

    const ops = [_]Device.DeviceOp{
        .wait_idle,
        .{ .acquire = .{ .swapchain = swapchain } },
        .{ .swapchain_image = .{
            .swapchain = swapchain,
            .index = 0,
        } },
        .{ .reset_command_pool = command_pool },
        .{ .reset_command_buffer = command_buffer },
        .{ .reset_fence = fence },
        .{ .update_descriptor_set = .{
            .set = descriptor_set,
            .writes = &.{.{
                .buffer = .{ .binding = 0, .buffer = buffer },
            }},
        } },
    };
    _ = ops;

    const syncs = [_]Queue.SyncOp{
        .{ .semaphore = semaphore },
        .{ .timeline = .{ .semaphore = semaphore, .value = 1 } },
        .{ .fence = fence },
    };
    const submit: Queue.SubmitInfo = .{
        .commands = &.{command_buffer},
        .waits = &syncs,
        .signals = &syncs,
        .fence = fence,
    };
    _ = submit;
    _ = sampler;
}
