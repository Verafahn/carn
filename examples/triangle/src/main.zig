//! Colored triangle example.
//!
//! Opens a GLFW window, creates a Vulkan surface for
//! it and draws an RGB triangle with the Vulkan
//! backend of `carn`'s render hardware interface.

const std = @import("std");
const builtin = @import("builtin");
const carn = @import("carn");
const glfw = @import("glfw");

const rhi = carn.render.rhi;
const vulkan = carn.render.vulkan;
const shader = carn.utils.shader;
const shared = @import("shared.zig");

const VertexStream = shader.Vertex(shared.VertexData);
const UniformStream = shader.Uniform(shared.UniformData);

const triangle_vertices = [_]shared.VertexData{
    .{
        .position = .{ 0.0, -0.5, 0.0, 1.0 },
        .color = .{ 1.0, 0.0, 0.0, 1.0 },
    },
    .{
        .position = .{ 0.5, 0.5, 0.0, 1.0 },
        .color = .{ 0.0, 1.0, 0.0, 1.0 },
    },
    .{
        .position = .{ -0.5, 0.5, 0.0, 1.0 },
        .color = .{ 0.0, 0.0, 1.0, 1.0 },
    },
};

const triangle_uniform = shared.UniformData{
    .tint = .{ 1.0, 1.0, 1.0, 1.0 },
};

const window_width = 800;
const window_height = 600;
const frame_timeout_ns = 5 * std.time.ns_per_s;

const clear_color = rhi.Device.Command.ClearColor{
    .float = .{ 0.02, 0.02, 0.03, 1.0 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    _ = glfw.glfwSetErrorCallback(glfwErrorCallback);
    if (glfw.glfwInit() != glfw.GLFW_TRUE) return error.GlfwInitFailed;
    defer glfw.glfwTerminate();

    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_NO_API);
    glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, glfw.GLFW_TRUE);

    const window = glfw.glfwCreateWindow(
        window_width,
        window_height,
        "Carn - colored triangle",
        null,
        null,
    ) orelse return error.WindowCreationFailed;
    defer glfw.glfwDestroyWindow(window);

    var triangle = try Triangle.init(gpa, window);
    defer triangle.deinit();

    while (glfw.glfwWindowShouldClose(window) == glfw.GLFW_FALSE) {
        glfw.glfwPollEvents();

        if (glfw.glfwGetKey(window, glfw.GLFW_KEY_ESCAPE) == glfw.GLFW_PRESS) {
            glfw.glfwSetWindowShouldClose(window, glfw.GLFW_TRUE);
        }

        var width: c_int = 0;
        var height: c_int = 0;
        glfw.glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 or height <= 0) {
            // Minimized windows cannot own a swapchain.
            triangle.dropSwapchain();
            glfw.glfwWaitEvents();
            continue;
        }

        try triangle.ensureSwapchain(.{
            .width = @intCast(width),
            .height = @intCast(height),
        });

        switch (try triangle.drawFrame()) {
            .presented => {},
            .out_of_date => triangle.dropSwapchain(),
        }
    }
}

fn glfwErrorCallback(code: c_int, description: [*c]const u8) callconv(.c) void {
    const text: []const u8 = if (description == null)
        "unknown"
    else
        std.mem.sliceTo(description, 0);
    std.log.err("GLFW error 0x{x}: {s}", .{
        @as(u32, @intCast(code)),
        text,
    });
}

const Triangle = struct {
    backend: *vulkan.Backend,
    device: rhi.Device,
    queue: rhi.Queue,
    surface: rhi.Resource.Surface,
    swapchain: ?rhi.Resource.Swapchain,
    format: rhi.Resource.Format,
    extent: rhi.Resource.Extent,
    vertex_shader: rhi.Resource.Shader,
    fragment_shader: rhi.Resource.Shader,
    vertex_buffer: rhi.Resource.Buffer,
    uniform_buffer: rhi.Resource.Buffer,
    descriptor_set_layout: rhi.Resource.DescriptorSetLayout,
    descriptor_set: rhi.Resource.DescriptorSet,
    pipeline_layout: rhi.Resource.PipelineLayout,
    pipeline: ?rhi.Resource.PipelineGraphics,
    command_pool: rhi.Resource.CommandPool,
    command_buffer: rhi.Resource.CommandBuffer,
    image_available: rhi.Resource.Semaphore,
    render_finished: rhi.Resource.Semaphore,
    in_flight: rhi.Resource.Fence,

    fn init(
        allocator: std.mem.Allocator,
        window: *glfw.GLFWwindow,
    ) !Triangle {
        const backend = try vulkan.init(allocator, .{
            .app_name = "carn triangle",
            .engine_name = "carn",
            .validation = builtin.mode == .debug,
            .verbose = builtin.mode == .debug,
        });
        errdefer _ = backend.deinit();

        const device = backend.device();
        const surface = try device.createSurface(
            try nativeSurfaceDesc(window),
        );
        errdefer surface.destroy();

        const queue = try findGraphicsQueue(device);

        const image_available = try device.createSemaphore(.{});
        errdefer image_available.destroy();
        const render_finished = try device.createSemaphore(.{});
        errdefer render_finished.destroy();
        const in_flight = try device.createFence(.{ .signaled = true });
        errdefer in_flight.destroy();

        const command_pool = try device.createCommandPool(.{
            .queue_family = queue.family_index,
        });
        errdefer command_pool.destroy();
        const command_buffer = try device.createCommandBuffer(.{
            .pool = command_pool,
        });
        errdefer command_buffer.destroy();

        const vertex_shader = try device.createShader(.{
            .label = "triangle vertex",
            .code = @embedFile("vertex"),
            .stage = .vertex,
        });
        errdefer vertex_shader.destroy();
        const fragment_shader = try device.createShader(.{
            .label = "triangle fragment",
            .code = @embedFile("fragment"),
            .stage = .fragment,
        });
        errdefer fragment_shader.destroy();

        const vertex_stride: usize = VertexStream.vertex_layout.stride;
        const vertex_buffer = try device.createBuffer(.{
            .label = "triangle vertices",
            .size = triangle_vertices.len * vertex_stride,
            .usage = .{ .vertex = true },
            .memory = .host,
        });
        errdefer vertex_buffer.destroy();
        try VertexStream.submit(&triangle_vertices, vertex_buffer);

        const uniform_buffer = try device.createBuffer(.{
            .label = "triangle uniform",
            .size = UniformStream.buffer_size,
            .usage = .{ .uniform = true },
            .memory = .host,
        });
        errdefer uniform_buffer.destroy();
        try UniformStream.write(triangle_uniform, uniform_buffer);

        const descriptor_set_layout = try device.createDescriptorSetLayout(.{
            .label = "triangle set layout",
            .bindings = &.{.{
                .binding = UniformStream.binding,
                .kind = .uniform_buffer,
                .stages = .{ .fragment = true },
            }},
        });
        errdefer descriptor_set_layout.destroy();

        const descriptor_set = try device.createDescriptorSet(.{
            .label = "triangle set",
            .layout = descriptor_set_layout,
        });
        errdefer descriptor_set.destroy();
        try descriptor_set.update(&.{.{ .buffer = .{
            .binding = UniformStream.binding,
            .buffer = uniform_buffer,
        } }});

        const pipeline_layout = try device.createPipelineLayout(.{
            .sets = &.{descriptor_set_layout},
        });
        errdefer pipeline_layout.destroy();

        return .{
            .backend = backend,
            .device = device,
            .queue = queue,
            .surface = surface,
            .swapchain = null,
            .format = undefined,
            .extent = .{ .width = 0, .height = 0 },
            .vertex_shader = vertex_shader,
            .fragment_shader = fragment_shader,
            .vertex_buffer = vertex_buffer,
            .uniform_buffer = uniform_buffer,
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_set = descriptor_set,
            .pipeline_layout = pipeline_layout,
            .pipeline = null,
            .command_pool = command_pool,
            .command_buffer = command_buffer,
            .image_available = image_available,
            .render_finished = render_finished,
            .in_flight = in_flight,
        };
    }

    fn deinit(self: *Triangle) void {
        self.dropSwapchain();
        if (self.pipeline) |pipeline| pipeline.destroy();
        self.pipeline_layout.destroy();
        self.descriptor_set.destroy();
        self.descriptor_set_layout.destroy();
        self.uniform_buffer.destroy();
        self.vertex_buffer.destroy();
        self.fragment_shader.destroy();
        self.vertex_shader.destroy();
        self.command_buffer.destroy();
        self.command_pool.destroy();
        self.in_flight.destroy();
        self.render_finished.destroy();
        self.image_available.destroy();
        self.surface.destroy();

        const status = self.backend.deinit();
        if (!status.ok()) {
            std.log.err(
                "vulkan shutdown was not clean: {d} leak(s), {d} violation(s)",
                .{ status.leaks, status.violations },
            );
        }
    }

    fn ensureSwapchain(
        self: *Triangle,
        extent: rhi.Resource.Extent,
    ) !void {
        if (self.swapchain != null and
            self.extent.width == extent.width and
            self.extent.height == extent.height)
        {
            return;
        }

        self.device.waitIdle() catch |err| switch (err) {
            error.DeviceLost, error.DeviceFailure => return err,
            else => {},
        };

        if (self.swapchain) |swapchain| {
            swapchain.destroy();
            self.swapchain = null;
        }

        const format = try pickSwapchainFormat(self.device, self.surface);
        if (self.pipeline == null or self.format != format) {
            if (self.pipeline) |pipeline| {
                pipeline.destroy();
                self.pipeline = null;
            }
            self.format = format;
            self.pipeline = try createGraphicsPipeline(
                self.device,
                self.pipeline_layout,
                self.vertex_shader,
                self.fragment_shader,
                format,
            );
        }

        const swapchain = try self.device.createSwapchain(.{
            .label = "triangle swapchain",
            .surface = self.surface,
            .format = format,
            .extent = extent,
            .mode = .fifo,
            .usage = .{ .color_attachment = true },
        });
        errdefer swapchain.destroy();

        // Presentation consumes binary semaphores. Recreate them together
        // with the swapchain so a failed present cannot leave one signalled.
        const image_available = try self.device.createSemaphore(.{});
        errdefer image_available.destroy();
        const render_finished = try self.device.createSemaphore(.{});
        errdefer render_finished.destroy();

        self.image_available.destroy();
        self.render_finished.destroy();
        self.image_available = image_available;
        self.render_finished = render_finished;
        self.swapchain = swapchain;
        self.extent = extent;
    }

    fn dropSwapchain(self: *Triangle) void {
        self.device.waitIdle() catch {};
        if (self.swapchain) |swapchain| {
            swapchain.destroy();
            self.swapchain = null;
        }
    }

    const FrameOutcome = enum {
        presented,
        out_of_date,
    };

    fn drawFrame(self: *Triangle) !FrameOutcome {
        const swapchain = self.swapchain orelse return error.NoSwapchain;
        const pipeline = self.pipeline orelse return error.NoPipeline;

        // Wait until the previous frame no longer uses the command buffer.
        _ = try self.queue.wait(
            &.{.{ .fence = self.in_flight }},
            false,
            frame_timeout_ns,
        );

        const image_index = swapchain.acquire(
            self.image_available,
            null,
            frame_timeout_ns,
        ) catch |err| switch (err) {
            error.SwapchainOutOfDate, error.SurfaceLost => return .out_of_date,
            else => return err,
        };

        try self.in_flight.reset();
        try self.command_pool.reset();

        const image = try swapchain.image(image_index);

        try self.record(.{ .begin = .{} });
        try self.record(.{ .begin_render_pass = .{
            .area = .{
                .width = self.extent.width,
                .height = self.extent.height,
            },
            .color = &.{.{
                .texture = image,
                .load = .clear,
                .store = .store,
                .clear = clear_color,
            }},
        } });
        try self.record(.{ .bind_pipeline = .{ .graphics = pipeline } });
        try self.record(.{ .bind_vertex_buffers = .{
            .buffers = &.{.{ .buffer = self.vertex_buffer }},
        } });
        try self.record(.{ .bind_descriptor_set = .{
            .index = 0,
            .set = self.descriptor_set,
        } });
        try self.record(.{ .set_viewport = .{
            .width = @floatFromInt(self.extent.width),
            .height = @floatFromInt(self.extent.height),
        } });
        try self.record(.{ .set_scissor = .{
            .width = self.extent.width,
            .height = self.extent.height,
        } });
        try self.record(.{ .draw = .{
            .vertices = .{ .count = triangle_vertices.len },
        } });
        try self.record(.{ .end_render_pass = {} });

        // Hand the image to the presentation engine.
        try self.record(.{ .barrier = .{
            .textures = &.{.{
                .texture = image,
                .src = .{
                    .stages = .{ .color_attachment = true },
                    .access = .{ .color_write = true },
                },
                .dst = .{
                    .stages = .{ .all = true },
                },
                .layout = .present,
            }},
        } });
        try self.record(.{ .end = {} });

        const waits = [_]rhi.Queue.SyncOp{
            .{ .semaphore = self.image_available },
        };
        const signals = [_]rhi.Queue.SyncOp{
            .{ .semaphore = self.render_finished },
        };
        const submits = [_]rhi.Queue.SubmitInfo{.{
            .commands = &.{self.command_buffer},
            .waits = &waits,
            .signals = &signals,
            .fence = self.in_flight,
        }};
        try self.queue.submit(&submits);

        const present_result = self.queue.present(
            swapchain,
            image_index,
            &.{.{ .semaphore = self.render_finished }},
        ) catch |err| switch (err) {
            error.SwapchainOutOfDate, error.SurfaceLost => return .out_of_date,
            else => return err,
        };

        return switch (present_result) {
            .success => .presented,
            .suboptimal => .out_of_date,
        };
    }

    fn record(
        self: *Triangle,
        command: rhi.Device.Command,
    ) rhi.Device.RecordError!void {
        try self.command_buffer.record(&command);
    }
};

fn createGraphicsPipeline(
    device: rhi.Device,
    layout: rhi.Resource.PipelineLayout,
    vertex: rhi.Resource.Shader,
    fragment: rhi.Resource.Shader,
    format: rhi.Resource.Format,
) !rhi.Resource.PipelineGraphics {
    const formats = [_]rhi.Resource.Format{format};
    return device.createPipelineGraphics(.{
        .label = "triangle pipeline",
        .layout = layout,
        .vertex = vertex,
        .fragment = fragment,
        .vertex_buffers = &.{VertexStream.vertex_layout},
        .primitive = .{ .cull = .none },
        .color = .{ .formats = &formats },
    });
}

fn pickSwapchainFormat(
    device: rhi.Device,
    surface: rhi.Resource.Surface,
) !rhi.Resource.Format {
    const candidates = [_]rhi.Resource.Format{
        .b8g8r8a8_srgb,
        .b8g8r8a8_unorm,
        .r8g8b8a8_srgb,
        .r8g8b8a8_unorm,
    };

    for (candidates) |format| {
        const support = try device.query(
            .{ .swapchain = .{ .format = format, .mode = .fifo } },
            0,
            surface.handle,
            null,
        );
        if (support == .supported) return format;
    }
    return error.NoSupportedSwapchainFormat;
}

fn findGraphicsQueue(device: rhi.Device) !rhi.Queue {
    var index: u32 = 0;
    while (true) : (index += 1) {
        var family: rhi.Device.QueueFamily = undefined;
        const support = device.query(
            .queue_family,
            index,
            null,
            std.mem.asBytes(&family),
        ) catch |err| switch (err) {
            error.IndexOutOfBounds => return error.NoGraphicsQueue,
            else => return err,
        };
        if (support != .supported) continue;
        if (!family.kinds.graphics or !family.kinds.present) continue;
        return device.getQueue(index);
    }
}

fn nativeSurfaceDesc(
    window: *glfw.GLFWwindow,
) !rhi.Resource.Surface.Desc {
    switch (builtin.os.tag) {
        .windows => {
            const hwnd = glfw.glfwGetWin32Window(window) orelse
                return error.NoNativeWindow;
            return .{
                .label = "triangle window",
                .window = @intFromPtr(hwnd),
                .platform = .win32,
            };
        },
        .linux => {
            const platform = glfw.glfwGetPlatform();
            if (platform == glfw.GLFW_PLATFORM_X11) {
                const display = glfw.glfwGetX11Display() orelse
                    return error.NoNativeDisplay;
                return .{
                    .label = "triangle window",
                    .display = @intFromPtr(display),
                    .window = @intCast(glfw.glfwGetX11Window(window)),
                    .platform = .xlib,
                };
            }
            if (platform == glfw.GLFW_PLATFORM_WAYLAND) {
                const display = glfw.glfwGetWaylandDisplay() orelse
                    return error.NoNativeDisplay;
                const surface = glfw.glfwGetWaylandWindow(window) orelse
                    return error.NoNativeWindow;
                return .{
                    .label = "triangle window",
                    .display = @intFromPtr(display),
                    .window = @intFromPtr(surface),
                    .platform = .wayland,
                };
            }
            return error.UnsupportedPlatform;
        },
        else => return error.UnsupportedPlatform,
    }
}
