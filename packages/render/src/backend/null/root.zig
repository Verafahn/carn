//! A deterministic, validating RHI backend for tests.
//!
//! The null backend owns no GPU state. It keeps an
//! explicit registry of every live resource and checks
//! handles, descriptions, command ordering and
//! synchronisation the way a real backend must.
//!
//! Queue family kinds and limits are validated against
//! the simulated configuration, so work sent to a queue
//! that cannot execute it is rejected.
//!
//! Contract violations are reported when they happen:
//! fallible operations return the matching RHI error,
//! void operations log and increment the violation
//! count, and `deinit` reports resources that were
//! never destroyed. This makes the backend suitable for
//! CI, where a real driver is not available.

const std = @import("std");
const rhi = @import("rhi");

const Device = rhi.Device;
const Queue = rhi.Queue;
const Resource = rhi.Resource;

const Buffer = Resource.Buffer;
const CommandBuffer = Resource.CommandBuffer;
const CommandPool = Resource.CommandPool;
const DescriptorSet = Resource.DescriptorSet;
const DescriptorSetLayout = Resource.DescriptorSetLayout;
const Fence = Resource.Fence;
const PipelineCompute = Resource.PipelineCompute;
const PipelineGraphics = Resource.PipelineGraphics;
const PipelineLayout = Resource.PipelineLayout;
const Sampler = Resource.Sampler;
const Semaphore = Resource.Semaphore;
const Shader = Resource.Shader;
const Surface = Resource.Surface;
const Swapchain = Resource.Swapchain;
const Texture = Resource.Texture;

const log = std.log.scoped(.null_rhi);

const no_index = std.math.maxInt(u32);

/// Errors returned by `init`.
pub const InitError = std.mem.Allocator.Error || error{InvalidDesc};

/// Simulated queue family.
pub const QueueFamily = struct {
    /// Number of queues exposed by the family.
    count: u32 = 1,
    /// Work kinds accepted by the family.
    kinds: Device.QueueKind = .{
        .graphics = true,
        .compute = true,
        .transfer = true,
        .present = true,
    },
};

/// Backend configuration.
pub const Options = struct {
    /// Simulated queue families, in index order.
    ///
    /// An empty slice exposes one family with one
    /// all-purpose queue.
    queue_families: []const QueueFamily = &.{},
    /// Limits returned by `Feature.limits`.
    limits: Device.Limits = default_limits,
    /// Log contract violations and leak reports.
    verbose: bool = true,
};

/// Limits reported by a default null device.
pub const default_limits: Device.Limits = .{
    .max_texture_extent_1d = 16384,
    .max_texture_extent_2d = 16384,
    .max_texture_extent_3d = 2048,
    .max_texture_layers = 2048,
    .max_texture_mips = 15,
    .max_vertex_stride = 2048,
    .max_vertex_buffers = 16,
    .max_descriptor_sets = 8,
    .max_push_constant_size = 128,
    .max_storage_buffers = 16,
    .max_sampled_textures = 32,
    .max_workgroup_size = .{ 1024, 1024, 64 },
    .max_workgroup_memory = 32768,
};

/// Result of `Backend.deinit`.
pub const Status = struct {
    /// Resources that were still alive.
    leaks: usize = 0,
    /// Contract violations observed at runtime.
    violations: usize = 0,

    /// Returns true when the backend shut down cleanly.
    ///
    /// # Return
    /// True when both `leaks` and `violations` are zero.
    pub fn ok(self: Status) bool {
        return self.leaks == 0 and self.violations == 0;
    }
};

/// Allocates and initialises a null backend.
///
/// # Args
/// - `allocator`: owns the backend and its bookkeeping.
/// - `options`: simulated queues, limits and logging.
///
/// # Return
/// A backend whose address is stable. Release it with
/// `deinit`.
///
/// # Error
/// - `error.OutOfMemory`: bookkeeping allocation fails.
/// - `error.InvalidDesc`: a queue family has a count
///   that overflows the flat queue list.
///
/// # Note
/// The device returned by `device` borrows the backend
/// storage. Destroy every resource before `deinit` for
/// a clean shutdown; live resources are reported as
/// leaks.
pub fn init(
    allocator: std.mem.Allocator,
    options: Options,
) InitError!*Backend {
    const self = try allocator.create(Backend);
    errdefer allocator.destroy(self);

    self.* = .{
        .allocator = allocator,
        .options = options,
        .limits = options.limits,
        .families = &.{},
        .queues = &.{},
    };

    const default_families = [_]QueueFamily{.{}};
    const families: []const QueueFamily = if (options.queue_families.len == 0)
        &default_families
    else
        options.queue_families;

    self.families = try allocator.alloc(FamilyState, families.len);
    errdefer allocator.free(self.families);

    var queue_count: usize = 0;
    for (families, 0..) |family, i| {
        if (family.count > std.math.maxInt(u32)) {
            return error.InvalidDesc;
        }
        const count: usize = family.count;
        queue_count = std.math.add(usize, queue_count, count) catch
            return error.InvalidDesc;
        self.families[i] = .{
            .count = family.count,
            .kinds = family.kinds,
        };
    }

    self.queues = try allocator.alloc(QueueState, queue_count);
    errdefer allocator.free(self.queues);

    var flat: usize = 0;
    for (self.families, 0..) |family, family_index| {
        var queue_index: u32 = 0;
        while (queue_index < family.count) : (queue_index += 1) {
            self.queues[flat] = .{
                .backend = self,
                .family_index = @intCast(family_index),
                .queue_index = queue_index,
            };
            flat += 1;
        }
    }

    return self;
}

/// A null backend instance.
pub const Backend = struct {
    allocator: std.mem.Allocator,
    options: Options,
    limits: Device.Limits,
    families: []FamilyState,
    queues: []QueueState,
    entries: std.ArrayListUnmanaged(Entry) = .empty,
    free_head: u32 = no_index,
    violations: usize = 0,

    /// Destroys the backend and reports leaked resources.
    ///
    /// # Return
    /// Leak and violation counts. Every live resource is
    /// released; when `options.verbose` is set, each leak
    /// is also logged before release.
    ///
    /// # Note
    /// The backend storage is freed. Device and queue
    /// values obtained from it are invalid afterwards.
    pub fn deinit(self: *Backend) Status {
        var leaks: usize = 0;
        for (self.entries.items) |*entry| {
            if (!entry.live) continue;
            leaks += 1;
            if (self.options.verbose) {
                log.err(
                    "leaked {s} id={d} label=\"{s}\"",
                    .{ @tagName(entry.kind), entry.generation, entry.label orelse "" },
                );
            }
            self.deinitData(&entry.data);
            if (entry.label) |label| self.allocator.free(label);
        }

        const status = Status{
            .leaks = leaks,
            .violations = self.violations,
        };

        self.entries.deinit(self.allocator);
        self.allocator.free(self.families);
        self.allocator.free(self.queues);

        const allocator = self.allocator;
        allocator.destroy(self);
        return status;
    }

    /// Returns the RHI device implemented by this backend.
    ///
    /// # Return
    /// A device value that borrows this backend.
    pub fn device(self: *Backend) Device {
        return .{
            .vptr = self,
            .vtable = &device_vtable,
        };
    }

    /// Returns the number of observed contract violations.
    ///
    /// # Return
    /// Every violation counted since `init`.
    pub fn violationCount(self: *const Backend) usize {
        return self.violations;
    }

    /// Returns the number of live resources.
    ///
    /// # Return
    /// Handles created through the device that have not
    /// been destroyed yet.
    pub fn liveResourceCount(self: *const Backend) usize {
        var count: usize = 0;
        for (self.entries.items) |entry| {
            if (entry.live) count += 1;
        }
        return count;
    }

    fn report(self: *Backend, comptime fmt: []const u8, args: anytype) void {
        self.violations += 1;
        if (self.options.verbose) log.err(fmt, args);
    }

    fn lookup(self: *Backend, value: anytype) ?*Entry {
        const resource = asResource(value);
        const low: u32 = @truncate(resource.handle);
        if (low == 0) return null;
        const index = low - 1;
        if (@as(usize, index) >= self.entries.items.len) return null;
        const entry = &self.entries.items[index];
        if (!entry.live) return null;
        if (entry.kind != resource.kind) return null;
        if (entry.generation != @as(u32, @truncate(resource.handle >> 32))) {
            return null;
        }
        return entry;
    }

    fn get(
        self: *Backend,
        value: anytype,
        comptime kind: Resource.Kind,
    ) ?*@FieldType(Data, @tagName(kind)) {
        const entry = self.lookup(value) orelse return null;
        if (entry.kind != kind) return null;
        return &@field(entry.data, @tagName(kind));
    }

    fn entryIndex(resource: Resource) u32 {
        return @as(u32, @truncate(resource.handle)) - 1;
    }

    fn encode(index: u32, generation: u32) u64 {
        return (@as(u64, generation) << 32) | (@as(u64, index) + 1);
    }

    fn newEntry(
        self: *Backend,
        kind: Resource.Kind,
        data: Data,
        label: []const u8,
    ) std.mem.Allocator.Error!Resource {
        const owned_label: ?[]u8 = if (label.len == 0)
            null
        else
            try self.allocator.dupe(u8, label);
        errdefer if (owned_label) |owned| self.allocator.free(owned);

        var index: u32 = undefined;
        var generation: u32 = 1;

        if (self.free_head != no_index) {
            index = self.free_head;
            const entry = &self.entries.items[index];
            self.free_head = entry.next_free;
            generation = entry.generation;
            entry.* = .{
                .kind = kind,
                .generation = generation,
                .live = true,
                .label = owned_label,
                .data = data,
            };
        } else {
            if (self.entries.items.len >= std.math.maxInt(u32) - 1) {
                return error.OutOfMemory;
            }
            index = @intCast(self.entries.items.len);
            try self.entries.append(self.allocator, .{
                .kind = kind,
                .generation = generation,
                .live = true,
                .label = owned_label,
                .data = data,
            });
        }

        return .{ .kind = kind, .handle = encode(index, generation) };
    }

    fn destroyEntryAt(self: *Backend, index: u32) void {
        if (@as(usize, index) >= self.entries.items.len) return;
        const entry = &self.entries.items[index];
        if (!entry.live) return;

        self.destroyData(&entry.data);
        if (entry.label) |label| self.allocator.free(label);

        entry.live = false;
        entry.generation +%= 1;
        if (entry.generation == 0) entry.generation = 1;
        entry.label = null;
        entry.next_free = self.free_head;
        self.free_head = index;
    }

    fn destroyEntry(self: *Backend, resource: Resource) void {
        if (self.lookup(resource) == null) {
            self.report(
                "destroy of invalid {s} handle",
                .{@tagName(resource.kind)},
            );
            return;
        }
        self.destroyEntryAt(entryIndex(resource));
    }

    fn deinitData(self: *Backend, data: *Data) void {
        switch (data.*) {
            .buffer => |*buffer| {
                if (buffer.host) |host| self.allocator.free(host);
            },
            .command_buffer => |*buffer| buffer.refs.deinit(self.allocator),
            .command_pool => |*pool| pool.buffers.deinit(self.allocator),
            .descriptor_set_layout => |*layout| {
                self.allocator.free(layout.bindings);
            },
            .pipeline_layout => |*layout| {
                self.allocator.free(layout.sets);
                self.allocator.free(layout.push_constants);
            },
            .shader => |*shader| self.allocator.free(shader.entry),
            .swapchain => |*chain| {
                self.allocator.free(chain.images);
                self.allocator.free(chain.acquired);
            },
            else => {},
        }
    }

    fn destroyData(self: *Backend, data: *Data) void {
        switch (data.*) {
            .command_pool => |*pool| {
                for (pool.buffers.items) |handle| {
                    const index = entryIndex(handle);
                    self.destroyEntryAt(index);
                }
                pool.buffers.deinit(self.allocator);
            },
            .swapchain => |*chain| {
                for (chain.images) |handle| {
                    self.destroyEntryAt(entryIndex(handle));
                }
                self.allocator.free(chain.images);
                self.allocator.free(chain.acquired);
            },
            else => self.deinitData(data),
        }
    }

    fn addRef(
        self: *Backend,
        buffer: *CommandBufferState,
        value: anytype,
    ) std.mem.Allocator.Error!void {
        try buffer.refs.append(self.allocator, asResource(value));
    }

    fn pipelineLayout(
        self: *Backend,
        buffer: *CommandBufferState,
    ) Device.RecordError!Resource {
        const pipeline = buffer.pipeline orelse
            return error.CommandBufferState;
        const kind = buffer.pipeline_kind orelse
            return error.CommandBufferState;
        return switch (kind) {
            .graphics => blk: {
                const state = self.get(pipeline, .pipeline_graphics) orelse
                    return error.InvalidResource;
                break :blk state.layout;
            },
            .compute => blk: {
                const state = self.get(pipeline, .pipeline_compute) orelse
                    return error.InvalidResource;
                break :blk state.layout;
            },
        };
    }

    fn removeChild(self: *Backend, handle: Resource) void {
        const entry = self.lookup(handle) orelse return;
        if (entry.kind != .command_buffer) return;
        const pool_handle = entry.data.command_buffer.pool;
        const pool = self.get(pool_handle, .command_pool) orelse return;
        for (pool.buffers.items, 0..) |child, i| {
            if (resourceEql(child, handle)) {
                _ = pool.buffers.swapRemove(i);
                return;
            }
        }
    }

    // ---------------------------------------------------------------
    // Resource creation
    // ---------------------------------------------------------------

    fn createResource(
        self: *Backend,
        desc: *const Resource.Desc,
        ret_addr: usize,
    ) Device.CreateError!Resource {
        _ = ret_addr;
        return switch (desc.*) {
            .buffer => |value| self.createBuffer(value),
            .command_buffer => |value| self.createCommandBuffer(value),
            .command_pool => |value| self.createCommandPool(value),
            .descriptor_set => |value| self.createDescriptorSet(value),
            .descriptor_set_layout => |value| self.createDescriptorSetLayout(value),
            .fence => |value| self.createFence(value),
            .pipeline_compute => |value| self.createPipelineCompute(value),
            .pipeline_graphics => |value| self.createPipelineGraphics(value),
            .pipeline_layout => |value| self.createPipelineLayout(value),
            .sampler => |value| self.createSampler(value),
            .semaphore => |value| self.createSemaphore(value),
            .shader => |value| self.createShader(value),
            .surface => |value| self.createSurface(value),
            .swapchain => |value| self.createSwapchain(value),
            .texture => |value| self.createTexture(value),
        };
    }

    fn createBuffer(
        self: *Backend,
        desc: Buffer.Desc,
    ) Device.CreateError!Resource {
        if (desc.size == 0) return error.InvalidDesc;
        if (std.meta.eql(desc.usage, Buffer.Usage{})) {
            return error.InvalidDesc;
        }

        var host: ?[]u8 = null;
        errdefer if (host) |memory| self.allocator.free(memory);

        if (desc.memory != .device) {
            if (desc.size > @as(u64, std.math.maxInt(usize))) {
                return error.OutOfMemory;
            }
            host = try self.allocator.alloc(u8, @intCast(desc.size));
            @memset(host.?, 0);
        }

        return try self.newEntry(.buffer, .{ .buffer = .{
            .size = desc.size,
            .usage = desc.usage,
            .memory = desc.memory,
            .host = host,
        } }, desc.label);
    }

    fn createTexture(
        self: *Backend,
        desc: Texture.Desc,
    ) Device.CreateError!Resource {
        if (!self.textureDescSupported(desc)) return error.InvalidDesc;
        var stored = desc;
        stored.label = "";
        return try self.newEntry(.texture, .{ .texture = .{
            .desc = stored,
        } }, desc.label);
    }

    fn createSampler(
        self: *Backend,
        desc: Sampler.Desc,
    ) Device.CreateError!Resource {
        if (!std.math.isFinite(desc.lod_min)) return error.InvalidDesc;
        if (!std.math.isFinite(desc.lod_max)) return error.InvalidDesc;
        if (desc.lod_max < desc.lod_min) return error.InvalidDesc;
        if (!std.math.isFinite(desc.max_anisotropy)) {
            return error.InvalidDesc;
        }
        if (desc.max_anisotropy < 1) return error.InvalidDesc;

        var stored = desc;
        stored.label = "";
        return try self.newEntry(.sampler, .{ .sampler = .{
            .desc = stored,
        } }, desc.label);
    }

    fn createShader(
        self: *Backend,
        desc: Shader.Desc,
    ) Device.CreateError!Resource {
        if (desc.code.len == 0 or desc.entry.len == 0) {
            return error.InvalidDesc;
        }
        const entry = try self.allocator.dupe(u8, desc.entry);
        errdefer self.allocator.free(entry);

        return try self.newEntry(.shader, .{ .shader = .{
            .stage = desc.stage,
            .language = desc.language,
            .entry = entry,
        } }, desc.label);
    }

    fn createDescriptorSetLayout(
        self: *Backend,
        desc: DescriptorSetLayout.Desc,
    ) Device.CreateError!Resource {
        for (desc.bindings, 0..) |binding, i| {
            if (binding.count == 0) return error.InvalidDesc;
            if (!stageFlagsAny(binding.stages)) return error.InvalidDesc;
            if (binding.read_only and
                binding.kind != .storage_buffer and
                binding.kind != .storage_texture)
            {
                return error.InvalidDesc;
            }
            for (desc.bindings[0..i]) |earlier| {
                if (earlier.binding == binding.binding) {
                    return error.InvalidDesc;
                }
            }
        }

        const bindings = try self.allocator.dupe(
            DescriptorSetLayout.Binding,
            desc.bindings,
        );
        errdefer self.allocator.free(bindings);

        return try self.newEntry(
            .descriptor_set_layout,
            .{ .descriptor_set_layout = .{ .bindings = bindings } },
            desc.label,
        );
    }

    fn createDescriptorSet(
        self: *Backend,
        desc: DescriptorSet.Desc,
    ) Device.CreateError!Resource {
        _ = self.get(desc.layout, .descriptor_set_layout) orelse
            return error.InvalidResource;

        return try self.newEntry(.descriptor_set, .{ .descriptor_set = .{
            .layout = asResource(desc.layout),
        } }, desc.label);
    }

    fn createPipelineLayout(
        self: *Backend,
        desc: PipelineLayout.Desc,
    ) Device.CreateError!Resource {
        for (desc.sets) |set_layout| {
            _ = self.get(set_layout, .descriptor_set_layout) orelse
                return error.InvalidResource;
        }

        for (desc.push_constants, 0..) |range, i| {
            if (range.size == 0) return error.InvalidDesc;
            if (!stageFlagsAny(range.stages)) return error.InvalidDesc;
            const end = @as(u64, range.offset) + range.size;
            if (end > self.limits.max_push_constant_size) {
                return error.InvalidDesc;
            }
            for (desc.push_constants[0..i]) |earlier| {
                if (pushConstantOverlap(earlier, range) and
                    stageFlagsIntersect(earlier.stages, range.stages))
                {
                    return error.InvalidDesc;
                }
            }
        }

        const sets = try self.allocator.alloc(Resource, desc.sets.len);
        errdefer self.allocator.free(sets);
        for (desc.sets, sets) |set, *out| out.* = asResource(set);
        const ranges = try self.allocator.dupe(
            PipelineLayout.PushConstantRange,
            desc.push_constants,
        );
        errdefer self.allocator.free(ranges);

        return try self.newEntry(.pipeline_layout, .{ .pipeline_layout = .{
            .sets = sets,
            .push_constants = ranges,
        } }, desc.label);
    }

    fn createPipelineCompute(
        self: *Backend,
        desc: PipelineCompute.Desc,
    ) Device.CreateError!Resource {
        _ = self.get(desc.layout, .pipeline_layout) orelse
            return error.InvalidResource;
        const shader = self.get(desc.shader, .shader) orelse
            return error.InvalidResource;
        if (shader.stage != .compute) return error.InvalidDesc;

        return try self.newEntry(
            .pipeline_compute,
            .{ .pipeline_compute = .{
                .layout = asResource(desc.layout),
                .kind = .compute,
            } },
            desc.label,
        );
    }

    fn createPipelineGraphics(
        self: *Backend,
        desc: PipelineGraphics.Desc,
    ) Device.CreateError!Resource {
        _ = self.get(desc.layout, .pipeline_layout) orelse
            return error.InvalidResource;
        const vertex = self.get(desc.vertex, .shader) orelse
            return error.InvalidResource;
        if (vertex.stage != .vertex) return error.InvalidDesc;
        if (desc.fragment) |fragment| {
            const shader = self.get(fragment, .shader) orelse
                return error.InvalidResource;
            if (shader.stage != .fragment) return error.InvalidDesc;
        }

        for (desc.vertex_buffers) |vertex_buffer| {
            if (vertex_buffer.stride == 0) return error.InvalidDesc;
            for (vertex_buffer.attributes) |attribute| {
                const size = vertexFormatSize(attribute.format);
                const end = @as(u64, attribute.offset) + size;
                if (end > vertex_buffer.stride) return error.InvalidDesc;
            }
        }

        if (desc.color.targets.len != 0 and
            desc.color.targets.len != desc.color.formats.len)
        {
            return error.InvalidDesc;
        }

        return try self.newEntry(
            .pipeline_graphics,
            .{ .pipeline_graphics = .{
                .layout = asResource(desc.layout),
                .kind = .graphics,
            } },
            desc.label,
        );
    }

    fn createCommandPool(
        self: *Backend,
        desc: CommandPool.Desc,
    ) Device.CreateError!Resource {
        if (@as(usize, desc.queue_family) >= self.families.len) {
            return error.InvalidDesc;
        }
        return try self.newEntry(.command_pool, .{ .command_pool = .{
            .queue_family = desc.queue_family,
            .reset_individual = desc.reset_individual,
        } }, desc.label);
    }

    fn createCommandBuffer(
        self: *Backend,
        desc: CommandBuffer.Desc,
    ) Device.CreateError!Resource {
        const pool = self.get(desc.pool, .command_pool) orelse
            return error.InvalidResource;

        const handle = try self.newEntry(
            .command_buffer,
            .{ .command_buffer = .{
                .pool = asResource(desc.pool),
                .level = desc.level,
            } },
            desc.label,
        );
        errdefer self.destroyEntryAt(entryIndex(handle));

        try pool.buffers.append(self.allocator, handle);
        return handle;
    }

    fn createFence(
        self: *Backend,
        desc: Fence.Desc,
    ) Device.CreateError!Resource {
        return try self.newEntry(.fence, .{ .fence = .{
            .signaled = desc.signaled,
        } }, desc.label);
    }

    fn createSemaphore(
        self: *Backend,
        desc: Semaphore.Desc,
    ) Device.CreateError!Resource {
        return try self.newEntry(.semaphore, .{ .semaphore = .{
            .kind = desc.kind,
            .value = desc.value,
        } }, desc.label);
    }

    fn createSurface(
        self: *Backend,
        desc: Surface.Desc,
    ) Device.CreateError!Resource {
        if (desc.platform != .headless and desc.window == 0) {
            return error.InvalidDesc;
        }
        if (needsDisplay(desc.platform) and desc.display == 0) {
            return error.InvalidDesc;
        }

        var stored = desc;
        stored.label = "";
        return try self.newEntry(.surface, .{ .surface = .{
            .desc = stored,
        } }, desc.label);
    }

    fn createSwapchain(
        self: *Backend,
        desc: Swapchain.Desc,
    ) Device.CreateError!Resource {
        const surface = self.get(desc.surface, .surface) orelse
            return error.InvalidResource;
        if (surface.desc.platform == .headless) return error.Unsupport;
        if (desc.extent.width == 0 or desc.extent.height == 0) {
            return error.InvalidDesc;
        }
        if (desc.extent.depth != 1) return error.InvalidDesc;
        if (desc.extent.width > self.limits.max_texture_extent_2d or
            desc.extent.height > self.limits.max_texture_extent_2d)
        {
            return error.InvalidDesc;
        }
        if (!swapchainFormatSupported(desc.format)) {
            return error.InvalidDesc;
        }
        if (!desc.usage.color_attachment) return error.InvalidDesc;
        if (desc.images > 64) return error.InvalidDesc;

        const count: u32 = if (desc.images == 0) 3 else desc.images;
        const count_usize: usize = @intCast(count);

        const swapchain_handle = try self.newEntry(
            .swapchain,
            .{ .swapchain = .{
                .surface = asResource(desc.surface),
                .format = desc.format,
                .extent = desc.extent,
                .usage = desc.usage,
                .images = &.{},
                .acquired = &.{},
            } },
            desc.label,
        );
        errdefer self.destroyEntryAt(entryIndex(swapchain_handle));

        const images = try self.allocator.alloc(Resource, count_usize);
        errdefer self.allocator.free(images);
        const acquired = try self.allocator.alloc(bool, count_usize);
        errdefer self.allocator.free(acquired);
        @memset(acquired, false);

        var created: usize = 0;
        errdefer for (images[0..created]) |handle| {
            self.destroyEntryAt(entryIndex(handle));
        };

        for (images, 0..) |*handle, i| {
            handle.* = try self.newEntry(
                .texture,
                .{ .texture = .{
                    .desc = .{
                        .extent = desc.extent,
                        .format = desc.format,
                        .usage = desc.usage,
                    },
                    .owner = swapchain_handle,
                    .layout = .present,
                } },
                desc.label,
            );
            created = i + 1;
        }

        const chain = self.get(swapchain_handle, .swapchain) orelse
            unreachable;
        chain.images = images;
        chain.acquired = acquired;
        return swapchain_handle;
    }

    // ---------------------------------------------------------------
    // Description validation
    // ---------------------------------------------------------------

    fn textureDescSupported(self: *Backend, desc: Texture.Desc) bool {
        if (desc.extent.width == 0) return false;
        if (desc.mips == 0 or desc.layers == 0) return false;
        if (std.meta.eql(desc.usage, Texture.Usage{})) return false;

        switch (desc.kind) {
            .d1 => if (desc.extent.height != 1 or desc.extent.depth != 1) {
                return false;
            },
            .d2 => if (desc.extent.depth != 1) return false,
            .d3 => {},
        }

        const max_extent = switch (desc.kind) {
            .d1 => self.limits.max_texture_extent_1d,
            .d2 => self.limits.max_texture_extent_2d,
            .d3 => self.limits.max_texture_extent_3d,
        };
        if (desc.extent.width > max_extent) return false;
        if (desc.kind != .d1 and desc.extent.height > max_extent) {
            return false;
        }
        if (desc.kind == .d3 and desc.extent.depth > max_extent) {
            return false;
        }
        if (@as(u32, desc.layers) > self.limits.max_texture_layers) {
            return false;
        }
        if (desc.mips > self.limits.max_texture_mips) return false;
        if (!formatSupports(desc.format, desc.usage)) return false;

        if (desc.samples == .x1) return true;
        return desc.usage.color_attachment or
            desc.usage.depth_stencil_attachment;
    }

    fn surfaceInUse(self: *Backend, handle: Resource) bool {
        for (self.entries.items) |*entry| {
            if (!entry.live or entry.kind != .swapchain) continue;
            if (entry.data.swapchain.surface.handle == handle.handle and
                entry.data.swapchain.surface.kind == handle.kind)
            {
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------
    // Mapping
    // ---------------------------------------------------------------

    fn mapResource(
        self: *Backend,
        handle: Resource,
        offset: u64,
        size: u64,
        flags: Device.MapFlags,
        ret_addr: usize,
    ) Device.MapError![]u8 {
        _ = flags;
        _ = ret_addr;
        const entry = self.lookup(handle) orelse return error.InvalidResource;
        if (entry.kind != .buffer) return error.InvalidResource;
        const buffer = &entry.data.buffer;

        if (buffer.memory == .device) return error.NotHostVisible;
        if (buffer.mapped) return error.AlreadyMapped;
        if (size == 0) return error.InvalidDesc;
        if (!fits(offset, size, buffer.size)) return error.OutOfRange;

        const host = buffer.host orelse return error.NotHostVisible;
        buffer.mapped = true;
        buffer.map_offset = offset;
        buffer.map_size = size;

        const start: usize = @intCast(offset);
        const end: usize = @intCast(offset + size);
        return host[start..end];
    }

    fn unmapResource(
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

        for (dirty) |range| {
            if (range.size == 0 or !fits(range.offset, range.size, buffer.size)) {
                self.report("unmap dirty range is out of bounds", .{});
                break;
            }
        }

        buffer.mapped = false;
        buffer.map_offset = 0;
        buffer.map_size = 0;
    }

    // ---------------------------------------------------------------
    // Command recording
    // ---------------------------------------------------------------

    fn recordCommand(
        self: *Backend,
        handle: Resource,
        cmd: *const Device.Command,
        ret_addr: usize,
    ) Device.RecordError!void {
        _ = ret_addr;
        const entry = self.lookup(handle) orelse return error.InvalidResource;
        if (entry.kind != .command_buffer) return error.InvalidResource;
        const buffer = &entry.data.command_buffer;

        switch (cmd.*) {
            .begin => |begin| {
                if (buffer.phase != .initial) {
                    return error.CommandBufferState;
                }
                buffer.phase = .recording;
                buffer.one_time = begin.one_time;
                buffer.in_render_pass = false;
                buffer.pipeline = null;
                buffer.pipeline_kind = null;
                buffer.viewport_set = false;
                buffer.scissor_set = false;
                buffer.index_buffer = false;
                buffer.kinds = .{};
                buffer.refs.clearRetainingCapacity();
            },
            .end => {
                if (buffer.phase != .recording) {
                    return error.CommandBufferState;
                }
                if (buffer.in_render_pass) {
                    return error.CommandBufferState;
                }
                buffer.phase = .ended;
            },
            .begin_render_pass => |pass| try self.recordBeginRenderPass(
                buffer,
                pass,
            ),
            .end_render_pass => {
                try requireRecording(buffer);
                if (!buffer.in_render_pass) {
                    return error.CommandBufferState;
                }
                buffer.in_render_pass = false;
            },
            .bind_pipeline => |bind| try self.recordBindPipeline(
                buffer,
                bind,
            ),
            .bind_descriptor_set => |bind| try self.recordBindDescriptorSet(
                buffer,
                bind,
            ),
            .bind_vertex_buffers => |bind| {
                try requireGraphics(buffer);
                for (bind.buffers) |vertex| {
                    const vb = self.get(vertex.buffer, .buffer) orelse
                        return error.InvalidResource;
                    if (!vb.usage.vertex) return error.InvalidDesc;
                    if (!fits(vertex.offset, 1, vb.size)) return error.InvalidDesc;
                    try self.addRef(buffer, vertex.buffer);
                }
            },
            .bind_index_buffer => |bind| {
                try requireGraphics(buffer);
                const index = self.get(bind.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (!index.usage.index) return error.InvalidDesc;
                if (!fits(bind.offset, 1, index.size)) return error.InvalidDesc;
                buffer.index_buffer = true;
                try self.addRef(buffer, bind.buffer);
            },
            .push_constants => |push| {
                try requireGraphicsOrCompute(buffer);
                if (!stageFlagsAny(push.stages)) return error.InvalidDesc;
                if (push.data.len == 0) return error.InvalidDesc;
                const layout_handle = try self.pipelineLayout(buffer);
                const layout = self.get(layout_handle, .pipeline_layout) orelse
                    return error.InvalidResource;
                if (!pushConstantsCover(layout.push_constants, push)) {
                    return error.InvalidDesc;
                }
            },
            .set_viewport => |viewport| {
                try requireGraphics(buffer);
                if (viewport.width <= 0 or viewport.height <= 0) {
                    return error.InvalidDesc;
                }
                buffer.viewport_set = true;
            },
            .set_scissor => |scissor| {
                try requireGraphics(buffer);
                _ = scissor;
                buffer.scissor_set = true;
            },
            .draw => |draw| {
                try requireDraw(buffer);
                buffer.kinds.graphics = true;
                _ = draw;
            },
            .draw_indexed => |draw| {
                try requireDraw(buffer);
                if (!buffer.index_buffer) return error.InvalidDesc;
                buffer.kinds.graphics = true;
                _ = draw;
            },
            .draw_indirect => |draw| {
                try self.recordIndirect(buffer, draw, false);
                buffer.kinds.graphics = true;
            },
            .draw_indexed_indirect => |draw| {
                try self.recordIndirect(buffer, draw, true);
                buffer.kinds.graphics = true;
            },
            .execute_commands => |execute| {
                try requireRecording(buffer);
                if (buffer.in_render_pass) return error.CommandBufferState;
                if (buffer.level != .primary) return error.InvalidDesc;
                for (execute.commands) |secondary| {
                    const cb = self.get(secondary, .command_buffer) orelse
                        return error.InvalidResource;
                    if (cb.level != .secondary) return error.InvalidDesc;
                    if (cb.phase != .ended) return error.CommandBufferState;
                    buffer.kinds.graphics =
                        buffer.kinds.graphics or cb.kinds.graphics;
                    buffer.kinds.compute =
                        buffer.kinds.compute or cb.kinds.compute;
                    buffer.kinds.transfer =
                        buffer.kinds.transfer or cb.kinds.transfer;
                    try self.addRef(buffer, secondary);
                }
            },
            .dispatch => |dispatch| {
                try requireCompute(buffer);
                buffer.kinds.compute = true;
                _ = dispatch;
            },
            .dispatch_indirect => |dispatch| {
                try requireCompute(buffer);
                if (buffer.in_render_pass) return error.CommandBufferState;
                const indirect = self.get(dispatch.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (!indirect.usage.indirect) return error.InvalidDesc;
                if (!fits(dispatch.offset, 12, indirect.size)) {
                    return error.InvalidDesc;
                }
                buffer.kinds.compute = true;
                try self.addRef(buffer, dispatch.buffer);
            },
            .copy_buffer => |copy| {
                try self.recordCopyBuffer(buffer, copy);
                buffer.kinds.transfer = true;
            },
            .copy_buffer_to_texture => |copy| {
                try self.recordCopyBufferTexture(buffer, copy);
                buffer.kinds.transfer = true;
            },
            .copy_texture_to_buffer => |copy| {
                try self.recordCopyTextureBuffer(buffer, copy);
                buffer.kinds.transfer = true;
            },
            .copy_texture => |copy| {
                try self.recordCopyTexture(buffer, copy);
                buffer.kinds.transfer = true;
            },
            .blit_texture => |blit| {
                try self.recordBlit(buffer, blit);
                buffer.kinds.transfer = true;
            },
            .resolve_texture => |resolve| {
                try self.recordCopyTexture(buffer, .{
                    .src = resolve.src,
                    .dst = resolve.dst,
                    .regions = resolve.regions,
                });
                const src = self.get(resolve.src, .texture) orelse
                    return error.InvalidResource;
                const dst = self.get(resolve.dst, .texture) orelse
                    return error.InvalidResource;
                if (src.desc.samples == .x1) return error.InvalidDesc;
                if (dst.desc.samples != .x1) return error.InvalidDesc;
                buffer.kinds.transfer = true;
            },
            .fill_buffer => |fill| {
                try requireRecording(buffer);
                if (buffer.in_render_pass) return error.CommandBufferState;
                const target = self.get(fill.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (!target.usage.transfer_dst) return error.InvalidDesc;
                if (fill.size == 0) return error.InvalidDesc;
                if (!fits(fill.offset, fill.size, target.size)) {
                    return error.InvalidDesc;
                }
                buffer.kinds.transfer = true;
                try self.addRef(buffer, fill.buffer);
            },
            .barrier => |barrier| try self.recordBarrier(buffer, barrier),
        }
    }

    fn recordBeginRenderPass(
        self: *Backend,
        buffer: *CommandBufferState,
        pass: Device.Command.RenderPass,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        if (pass.area.width == 0 or pass.area.height == 0) {
            return error.InvalidDesc;
        }

        for (pass.color) |attachment| {
            const color = self.get(attachment.texture, .texture) orelse
                return error.InvalidResource;
            if (!color.desc.usage.color_attachment) return error.InvalidDesc;
            if (attachment.resolve) |resolve| {
                const resolved = self.get(resolve, .texture) orelse
                    return error.InvalidResource;
                if (!resolved.desc.usage.color_attachment) {
                    return error.InvalidDesc;
                }
                try self.addRef(buffer, resolve);
            }
            try self.addRef(buffer, attachment.texture);
        }

        if (pass.depth_stencil) |attachment| {
            const depth = self.get(attachment.texture, .texture) orelse
                return error.InvalidResource;
            if (!depth.desc.usage.depth_stencil_attachment) {
                return error.InvalidDesc;
            }
            try self.addRef(buffer, attachment.texture);
        }

        buffer.in_render_pass = true;
    }

    fn recordBindPipeline(
        self: *Backend,
        buffer: *CommandBufferState,
        bind: Device.Command.BindPipeline,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        switch (bind) {
            .graphics => |pipeline| {
                const state = self.get(pipeline, .pipeline_graphics) orelse
                    return error.InvalidResource;
                _ = state;
                buffer.pipeline = asResource(pipeline);
                buffer.pipeline_kind = .graphics;
                try self.addRef(buffer, pipeline);
            },
            .compute => |pipeline| {
                const state = self.get(pipeline, .pipeline_compute) orelse
                    return error.InvalidResource;
                _ = state;
                buffer.pipeline = asResource(pipeline);
                buffer.pipeline_kind = .compute;
                try self.addRef(buffer, pipeline);
            },
        }
    }

    fn recordBindDescriptorSet(
        self: *Backend,
        buffer: *CommandBufferState,
        bind: Device.Command.BindDescriptorSet,
    ) Device.RecordError!void {
        try requireGraphicsOrCompute(buffer);
        const layout_handle = try self.pipelineLayout(buffer);
        const layout = self.get(layout_handle, .pipeline_layout) orelse
            return error.InvalidResource;
        if (@as(usize, bind.index) >= layout.sets.len) {
            return error.InvalidDesc;
        }

        const set = self.get(bind.set, .descriptor_set) orelse
            return error.InvalidResource;
        if (!resourceEql(set.layout, layout.sets[bind.index])) {
            return error.InvalidDesc;
        }
        try self.addRef(buffer, bind.set);
    }

    fn recordIndirect(
        self: *Backend,
        buffer: *CommandBufferState,
        draw: Device.Command.DrawIndirect,
        indexed: bool,
    ) Device.RecordError!void {
        try requireDraw(buffer);
        if (indexed and !buffer.index_buffer) return error.InvalidDesc;
        const indirect = self.get(draw.buffer, .buffer) orelse
            return error.InvalidResource;
        if (!indirect.usage.indirect) return error.InvalidDesc;
        if (draw.count == 0) return;

        const natural: u64 = if (indexed) 20 else 16;
        const stride = if (draw.stride == 0) natural else draw.stride;
        if (stride < natural) return error.InvalidDesc;
        const total = @as(u64, draw.count) * stride;
        if (!fits(draw.offset, total, indirect.size)) {
            return error.InvalidDesc;
        }
        try self.addRef(buffer, draw.buffer);
    }

    fn recordCopyBuffer(
        self: *Backend,
        buffer: *CommandBufferState,
        copy: Device.Command.CopyBuffer,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        const src = self.get(copy.src, .buffer) orelse
            return error.InvalidResource;
        const dst = self.get(copy.dst, .buffer) orelse
            return error.InvalidResource;
        if (!src.usage.transfer_src or !dst.usage.transfer_dst) {
            return error.InvalidDesc;
        }
        for (copy.regions) |region| {
            if (region.size == 0) return error.InvalidDesc;
            if (!fits(region.src_offset, region.size, src.size)) {
                return error.InvalidDesc;
            }
            if (!fits(region.dst_offset, region.size, dst.size)) {
                return error.InvalidDesc;
            }
        }
        try self.addRef(buffer, copy.src);
        try self.addRef(buffer, copy.dst);
    }

    fn recordCopyBufferTexture(
        self: *Backend,
        buffer: *CommandBufferState,
        copy: Device.Command.CopyBufferTexture,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        const src = self.get(copy.src, .buffer) orelse
            return error.InvalidResource;
        const dst = self.get(copy.dst, .texture) orelse
            return error.InvalidResource;
        if (!src.usage.transfer_src or !dst.desc.usage.transfer_dst) {
            return error.InvalidDesc;
        }
        for (copy.regions) |region| {
            if (region.buffer_offset >= src.size) return error.InvalidDesc;
            try validateTextureRegion(dst.desc, region.region);
        }
        try self.addRef(buffer, copy.src);
        try self.addRef(buffer, copy.dst);
    }

    fn recordCopyTextureBuffer(
        self: *Backend,
        buffer: *CommandBufferState,
        copy: Device.Command.CopyTextureBuffer,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        const src = self.get(copy.src, .texture) orelse
            return error.InvalidResource;
        const dst = self.get(copy.dst, .buffer) orelse
            return error.InvalidResource;
        if (!src.desc.usage.transfer_src or !dst.usage.transfer_dst) {
            return error.InvalidDesc;
        }
        for (copy.regions) |region| {
            if (region.buffer_offset >= dst.size) return error.InvalidDesc;
            try validateTextureRegion(src.desc, region.region);
        }
        try self.addRef(buffer, copy.src);
        try self.addRef(buffer, copy.dst);
    }

    fn recordCopyTexture(
        self: *Backend,
        buffer: *CommandBufferState,
        copy: Device.Command.CopyTexture,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        const src = self.get(copy.src, .texture) orelse
            return error.InvalidResource;
        const dst = self.get(copy.dst, .texture) orelse
            return error.InvalidResource;
        if (!src.desc.usage.transfer_src or !dst.desc.usage.transfer_dst) {
            return error.InvalidDesc;
        }
        if (src.desc.format != dst.desc.format) return error.InvalidDesc;
        for (copy.regions) |region| {
            try validateTextureRegion(src.desc, region.src);
            try validateTextureRegion(dst.desc, region.dst);
            if (!std.meta.eql(region.src.extent, region.dst.extent)) {
                return error.InvalidDesc;
            }
        }
        try self.addRef(buffer, copy.src);
        try self.addRef(buffer, copy.dst);
    }

    fn recordBlit(
        self: *Backend,
        buffer: *CommandBufferState,
        blit: Device.Command.BlitTexture,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        if (buffer.in_render_pass) return error.CommandBufferState;
        if (blit.filter != .nearest and blit.filter != .linear) {
            return error.InvalidDesc;
        }
        const src = self.get(blit.src, .texture) orelse
            return error.InvalidResource;
        const dst = self.get(blit.dst, .texture) orelse
            return error.InvalidResource;
        if (!src.desc.usage.transfer_src or !dst.desc.usage.transfer_dst) {
            return error.InvalidDesc;
        }
        for (blit.regions) |region| {
            try validateTextureRegion(src.desc, region.src);
            try validateTextureRegion(dst.desc, region.dst);
        }
        try self.addRef(buffer, blit.src);
        try self.addRef(buffer, blit.dst);
    }

    fn recordBarrier(
        self: *Backend,
        buffer: *CommandBufferState,
        barrier: Device.Command.Barrier,
    ) Device.RecordError!void {
        try requireRecording(buffer);
        for (barrier.buffers) |item| {
            const target = self.get(item.buffer, .buffer) orelse
                return error.InvalidResource;
            if (item.offset > target.size) return error.InvalidDesc;
            const size = if (item.size == 0)
                target.size - item.offset
            else
                item.size;
            if (!fits(item.offset, size, target.size)) {
                return error.InvalidDesc;
            }
            try self.addRef(buffer, item.buffer);
        }
        for (barrier.textures) |item| {
            const target = self.get(item.texture, .texture) orelse
                return error.InvalidResource;
            try validateSubresourceRange(target.desc, item.range);
            target.layout = item.layout;
            try self.addRef(buffer, item.texture);
        }
    }

    // ---------------------------------------------------------------
    // Queue operations
    // ---------------------------------------------------------------

    fn queueSubmit(
        self: *Backend,
        queue: *QueueState,
        submits: []const Queue.SubmitInfo,
        ret_addr: usize,
    ) Queue.SubmitError!void {
        _ = ret_addr;
        const family = self.families[queue.family_index];
        for (submits) |submit| {
            for (submit.commands) |handle| {
                const entry = self.lookup(handle) orelse
                    return error.InvalidResource;
                if (entry.kind != .command_buffer) return error.InvalidResource;
                const buffer = &entry.data.command_buffer;
                if (buffer.level != .primary) return error.InvalidDesc;
                if (buffer.phase != .ended) return error.InvalidDesc;

                if (buffer.kinds.graphics and !family.kinds.graphics) {
                    return error.InvalidDesc;
                }
                if (buffer.kinds.compute and
                    !family.kinds.graphics and !family.kinds.compute)
                {
                    return error.InvalidDesc;
                }
                if (buffer.kinds.transfer and
                    !family.kinds.graphics and
                    !family.kinds.compute and
                    !family.kinds.transfer)
                {
                    return error.InvalidDesc;
                }

                for (buffer.refs.items) |referenced| {
                    if (self.lookup(referenced) == null) {
                        return error.InvalidResource;
                    }
                }
            }

            for (submit.waits) |wait| try self.validateSyncOp(wait, true);
            for (submit.signals) |signal| {
                try self.validateSyncOp(signal, false);
            }
            if (submit.fence) |handle| {
                _ = self.get(handle, .fence) orelse
                    return error.InvalidResource;
            }
        }

        for (submits) |submit| {
            for (submit.signals) |signal| try self.signalSyncOp(signal);
            if (submit.fence) |handle| {
                const fence = self.get(handle, .fence) orelse
                    return error.InvalidResource;
                fence.signaled = true;
            }
        }
    }

    fn queuePresent(
        self: *Backend,
        queue: *QueueState,
        swapchain: Resource,
        image_index: u32,
        wait_ops: []const Queue.SyncOp,
        ret_addr: usize,
    ) Queue.PresentError!Queue.PresentResult {
        _ = ret_addr;
        const family = self.families[queue.family_index];
        if (!family.kinds.present) return error.InvalidDesc;
        const chain = self.get(swapchain, .swapchain) orelse
            return error.InvalidResource;
        _ = self.get(chain.surface, .surface) orelse
            return error.InvalidResource;
        if (@as(usize, image_index) >= chain.images.len) {
            return error.InvalidDesc;
        }
        if (!chain.acquired[image_index]) {
            self.report("present of an image that was not acquired", .{});
            return error.InvalidState;
        }

        for (wait_ops) |wait| {
            switch (wait) {
                .fence => return error.InvalidDesc,
                .semaphore => |handle| {
                    const semaphore = self.get(handle, .semaphore) orelse
                        return error.InvalidResource;
                    if (semaphore.kind != .binary) return error.InvalidDesc;
                },
                .timeline => |timeline| {
                    const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                        return error.InvalidResource;
                    if (semaphore.kind != .timeline) return error.InvalidDesc;
                    if (semaphore.value < timeline.value) {
                        return error.InvalidState;
                    }
                },
            }
        }

        chain.acquired[image_index] = false;
        return .success;
    }

    fn queueWait(
        self: *Backend,
        syncs: []const Queue.SyncOp,
        wait_any: bool,
        timeout_ns: ?u64,
        ret_addr: usize,
    ) Queue.WaitError!Queue.WaitResult {
        _ = ret_addr;

        var result = Queue.WaitResult{};

        if (syncs.len == 0) return result;

        var ready_count: usize = 0;
        var first_ready: ?usize = null;
        var first_value: u64 = 0;

        for (syncs, 0..) |sync, i| {
            const ready, const value = self.syncReady(sync) catch |err| {
                return err;
            };
            if (ready) {
                ready_count += 1;
                if (first_ready == null) {
                    first_ready = i;
                    first_value = value;
                }
            }
        }

        const satisfied = if (wait_any)
            ready_count > 0
        else
            ready_count == syncs.len;

        if (!satisfied) {
            if (timeout_ns == null) {
                self.report("wait cannot make progress", .{});
            }
            return error.Timeout;
        }

        if (first_ready) |index| {
            result.index = index;
            result.value = first_value;
        }
        return result;
    }

    fn validateSyncOp(
        self: *Backend,
        sync: Queue.SyncOp,
        is_wait: bool,
    ) Queue.SubmitError!void {
        switch (sync) {
            .fence => return error.InvalidDesc,
            .semaphore => |handle| {
                const semaphore = self.get(handle, .semaphore) orelse
                    return error.InvalidResource;
                if (semaphore.kind != .binary) return error.InvalidResource;
            },
            .timeline => |timeline| {
                const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                    return error.InvalidResource;
                if (semaphore.kind != .timeline) return error.InvalidResource;
                if (!is_wait and timeline.value <= semaphore.value) {
                    return error.InvalidDesc;
                }
            },
        }
    }

    fn signalSyncOp(
        self: *Backend,
        sync: Queue.SyncOp,
    ) Queue.SubmitError!void {
        switch (sync) {
            .fence => unreachable,
            .semaphore => |handle| {
                const semaphore = self.get(handle, .semaphore) orelse
                    return error.InvalidResource;
                semaphore.signaled = true;
            },
            .timeline => |timeline| {
                const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                    return error.InvalidResource;
                semaphore.value = @max(semaphore.value, timeline.value);
            },
        }
    }

    fn syncReady(
        self: *Backend,
        sync: Queue.SyncOp,
    ) Queue.WaitError!struct { bool, u64 } {
        switch (sync) {
            .fence => |handle| {
                const fence = self.get(handle, .fence) orelse
                    return error.InvalidResource;
                return .{ fence.signaled, 0 };
            },
            .semaphore => |handle| {
                const semaphore = self.get(handle, .semaphore) orelse
                    return error.InvalidResource;
                if (semaphore.kind != .binary) return error.InvalidResource;
                return .{ semaphore.signaled, 0 };
            },
            .timeline => |timeline| {
                const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                    return error.InvalidResource;
                if (semaphore.kind != .timeline) return error.InvalidResource;
                return .{
                    semaphore.value >= timeline.value,
                    semaphore.value,
                };
            },
        }
    }

    // ---------------------------------------------------------------
    // Immediate operations
    // ---------------------------------------------------------------

    fn deviceOp(
        self: *Backend,
        op: Device.DeviceOp,
        out: ?[]u8,
        ret_addr: usize,
    ) Device.DeviceOpError!void {
        _ = ret_addr;
        switch (op) {
            .wait_idle => {},
            .acquire => |acquire| try self.acquireImage(acquire, out),
            .swapchain_image => |image| try self.swapchainImage(image, out),
            .reset_command_pool => |handle| {
                const pool = self.get(handle, .command_pool) orelse
                    return error.InvalidResource;
                for (pool.buffers.items) |buffer_handle| {
                    const entry = self.lookup(buffer_handle) orelse continue;
                    if (entry.kind != .command_buffer) continue;
                    resetCommandBuffer(&entry.data.command_buffer);
                }
            },
            .reset_command_buffer => |handle| {
                const entry = self.lookup(handle) orelse
                    return error.InvalidResource;
                if (entry.kind != .command_buffer) return error.InvalidResource;
                resetCommandBuffer(&entry.data.command_buffer);
            },
            .reset_fence => |handle| {
                const fence = self.get(handle, .fence) orelse
                    return error.InvalidResource;
                fence.signaled = false;
            },
            .update_descriptor_set => |write| try self.updateDescriptorSet(write),
        }
    }

    fn acquireImage(
        self: *Backend,
        acquire: Device.Acquire,
        out: ?[]u8,
    ) Device.DeviceOpError!void {
        const chain = self.get(acquire.swapchain, .swapchain) orelse
            return error.InvalidResource;
        _ = self.get(chain.surface, .surface) orelse
            return error.InvalidResource;

        const signal = if (acquire.signal) |handle|
            self.get(handle, .semaphore) orelse
                return error.InvalidResource
        else
            null;
        const fence = if (acquire.fence) |handle|
            self.get(handle, .fence) orelse
                return error.InvalidResource
        else
            null;

        var index: ?u32 = null;
        for (chain.acquired, 0..) |acquired, i| {
            if (!acquired) {
                index = @intCast(i);
                break;
            }
        }
        const selected = index orelse {
            self.report("all swapchain images are already acquired", .{});
            return error.InvalidState;
        };

        const bytes = out orelse return error.InsufficientBuffer;
        if (bytes.len < @sizeOf(u32)) return error.InsufficientBuffer;

        chain.acquired[selected] = true;
        if (signal) |semaphore| {
            switch (semaphore.kind) {
                .binary => semaphore.signaled = true,
                .timeline => semaphore.value +%= 1,
            }
        }
        if (fence) |state| state.signaled = true;
        @memcpy(bytes[0..@sizeOf(u32)], std.mem.asBytes(&selected));
    }

    fn swapchainImage(
        self: *Backend,
        request: Device.SwapchainImage,
        out: ?[]u8,
    ) Device.DeviceOpError!void {
        const chain = self.get(request.swapchain, .swapchain) orelse
            return error.InvalidResource;
        if (@as(usize, request.index) >= chain.images.len) {
            return error.IndexOutOfBounds;
        }

        const handle = chain.images[request.index];
        const bytes = out orelse return error.InsufficientBuffer;
        if (bytes.len < @sizeOf(Resource)) return error.InsufficientBuffer;
        @memcpy(bytes[0..@sizeOf(Resource)], std.mem.asBytes(&handle));
    }

    fn updateDescriptorSet(
        self: *Backend,
        write: Device.DescriptorWrite,
    ) Device.DeviceOpError!void {
        const set = self.get(write.set, .descriptor_set) orelse
            return error.InvalidResource;
        const layout = self.get(set.layout, .descriptor_set_layout) orelse
            return error.InvalidResource;

        for (write.writes) |item| switch (item) {
            .buffer => |buffer_write| {
                const binding = findBinding(layout.bindings, buffer_write.binding) orelse
                    return error.InvalidDesc;
                if (binding.kind != .uniform_buffer and
                    binding.kind != .storage_buffer)
                {
                    return error.InvalidDesc;
                }
                if (buffer_write.index >= binding.count) {
                    return error.InvalidDesc;
                }
                const buffer = self.get(buffer_write.buffer, .buffer) orelse
                    return error.InvalidResource;
                if (binding.kind == .uniform_buffer and !buffer.usage.uniform) {
                    return error.InvalidDesc;
                }
                if (binding.kind == .storage_buffer and !buffer.usage.storage) {
                    return error.InvalidDesc;
                }
                const size = if (buffer_write.size == 0)
                    buffer.size - buffer_write.offset
                else
                    buffer_write.size;
                if (!fits(buffer_write.offset, size, buffer.size)) {
                    return error.InvalidDesc;
                }
            },
            .texture => |texture_write| {
                const binding = findBinding(layout.bindings, texture_write.binding) orelse
                    return error.InvalidDesc;
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
                if (texture_write.index >= binding.count) {
                    return error.InvalidDesc;
                }
                try validateSubresourceRange(texture.desc, texture_write.range);
            },
            .sampler => |sampler_write| {
                const binding = findBinding(layout.bindings, sampler_write.binding) orelse
                    return error.InvalidDesc;
                if (binding.kind != .sampler) return error.InvalidDesc;
                if (sampler_write.index >= binding.count) {
                    return error.InvalidDesc;
                }
                _ = self.get(sampler_write.sampler, .sampler) orelse
                    return error.InvalidResource;
            },
            .combined => |combined| {
                const binding = findBinding(layout.bindings, combined.binding) orelse
                    return error.InvalidDesc;
                if (binding.kind != .combined_sampler) {
                    return error.InvalidDesc;
                }
                if (combined.index >= binding.count) {
                    return error.InvalidDesc;
                }
                const texture = self.get(combined.texture, .texture) orelse
                    return error.InvalidResource;
                if (!texture.desc.usage.sampled) return error.InvalidDesc;
                _ = self.get(combined.sampler, .sampler) orelse
                    return error.InvalidResource;
                try validateSubresourceRange(texture.desc, combined.range);
            },
        };
    }

    // ---------------------------------------------------------------
    // Feature queries
    // ---------------------------------------------------------------

    fn query(
        self: *Backend,
        feature: Device.Feature,
        feature_index: u32,
        handle: ?Resource,
        out: ?[]u8,
        ret_addr: usize,
    ) Device.QueryError!Device.Support {
        _ = ret_addr;
        switch (feature) {
            .format => |format| {
                if (feature_index != 0) return error.IndexOutOfBounds;
                return if (formatSupports(format.format, format.usage))
                    .supported
                else
                    .unsupported;
            },
            .texture => |desc| {
                if (feature_index != 0) return error.IndexOutOfBounds;
                return if (self.textureDescSupported(desc))
                    .supported
                else
                    .unsupported;
            },
            .swapchain => |swapchain| {
                if (feature_index != 0) return error.IndexOutOfBounds;
                const resource = handle orelse return error.InvalidResource;
                const surface = self.get(resource, .surface) orelse
                    return error.InvalidResource;
                if (surface.desc.platform == .headless) {
                    return .unsupported;
                }
                return if (swapchainFormatSupported(swapchain.format))
                    .supported
                else
                    .unsupported;
            },
            .queue_family => {
                if (@as(usize, feature_index) >= self.families.len) {
                    return error.IndexOutOfBounds;
                }
                const family = self.families[feature_index];
                try writeOut(Device.QueueFamily, out, .{
                    .count = family.count,
                    .kinds = family.kinds,
                });
                return .supported;
            },
            .limits => {
                if (feature_index != 0) return error.IndexOutOfBounds;
                try writeOut(Device.Limits, out, self.limits);
                return .supported;
            },
        }
    }

    fn getQueue(
        self: *Backend,
        index: u32,
        ret_addr: usize,
    ) Device.GetQueueError!Queue {
        _ = ret_addr;
        if (self.queues.len == 0) return error.NoQueues;
        if (@as(usize, index) >= self.queues.len) return error.IndexOutOfBounds;
        const state = &self.queues[index];
        return .{
            .vptr = state,
            .vtable = &queue_vtable,
            .family_index = state.family_index,
            .queue_index = state.queue_index,
        };
    }
};

// -------------------------------------------------------------------
// State
// -------------------------------------------------------------------

const FamilyState = struct {
    count: u32,
    kinds: Device.QueueKind,
};

const QueueState = struct {
    backend: *Backend,
    family_index: u32,
    queue_index: u32,
};

const Entry = struct {
    kind: Resource.Kind,
    generation: u32,
    live: bool,
    label: ?[]u8,
    data: Data,
    next_free: u32 = no_index,
};

const Data = union(Resource.Kind) {
    buffer: BufferState,
    command_buffer: CommandBufferState,
    command_pool: CommandPoolState,
    descriptor_set: DescriptorSetState,
    descriptor_set_layout: DescriptorSetLayoutState,
    fence: FenceState,
    pipeline_compute: PipelineState,
    pipeline_graphics: PipelineState,
    pipeline_layout: PipelineLayoutState,
    sampler: SamplerState,
    semaphore: SemaphoreState,
    shader: ShaderState,
    surface: SurfaceState,
    swapchain: SwapchainState,
    texture: TextureState,
};

const BufferState = struct {
    size: u64,
    usage: Buffer.Usage,
    memory: Buffer.Memory,
    host: ?[]u8 = null,
    mapped: bool = false,
    map_offset: u64 = 0,
    map_size: u64 = 0,
};

const TextureState = struct {
    desc: Texture.Desc,
    owner: ?Resource = null,
    layout: Texture.Layout = .undefined,
};

const SamplerState = struct {
    desc: Sampler.Desc,
};

const ShaderState = struct {
    stage: Shader.Stage,
    language: Shader.Language,
    entry: []u8,
};

const DescriptorSetLayoutState = struct {
    bindings: []DescriptorSetLayout.Binding,
};

const DescriptorSetState = struct {
    layout: Resource,
};

const PipelineLayoutState = struct {
    sets: []Resource,
    push_constants: []PipelineLayout.PushConstantRange,
};

const PipelineKind = enum {
    graphics,
    compute,
};

const PipelineState = struct {
    layout: Resource,
    kind: PipelineKind,
};

const CommandPoolState = struct {
    queue_family: u32,
    reset_individual: bool,
    buffers: std.ArrayListUnmanaged(Resource) = .empty,
};

const Phase = enum {
    initial,
    recording,
    ended,
};

const CommandBufferState = struct {
    pool: Resource,
    level: CommandBuffer.Level,
    phase: Phase = .initial,
    one_time: bool = false,
    in_render_pass: bool = false,
    pipeline: ?Resource = null,
    pipeline_kind: ?PipelineKind = null,
    viewport_set: bool = false,
    scissor_set: bool = false,
    index_buffer: bool = false,
    kinds: Device.QueueKind = .{},
    refs: std.ArrayListUnmanaged(Resource) = .empty,
};

const FenceState = struct {
    signaled: bool,
};

const SemaphoreState = struct {
    kind: Semaphore.Kind,
    value: u64 = 0,
    signaled: bool = false,
};

const SurfaceState = struct {
    desc: Surface.Desc,
};

const SwapchainState = struct {
    surface: Resource,
    format: Resource.Format,
    extent: Resource.Extent,
    usage: Texture.Usage,
    images: []Resource,
    acquired: []bool,
};

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

fn requireRecording(
    buffer: *CommandBufferState,
) Device.RecordError!void {
    if (buffer.phase != .recording) return error.CommandBufferState;
}

fn requireGraphics(
    buffer: *CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind != .graphics) return error.CommandBufferState;
}

fn requireCompute(
    buffer: *CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind != .compute) return error.CommandBufferState;
}

fn requireGraphicsOrCompute(
    buffer: *CommandBufferState,
) Device.RecordError!void {
    try requireRecording(buffer);
    if (buffer.pipeline_kind == null) return error.CommandBufferState;
}

fn requireDraw(buffer: *CommandBufferState) Device.RecordError!void {
    try requireGraphics(buffer);
    if (!buffer.in_render_pass) return error.CommandBufferState;
    if (!buffer.viewport_set or !buffer.scissor_set) {
        return error.CommandBufferState;
    }
}

fn resetCommandBuffer(buffer: *CommandBufferState) void {
    buffer.phase = .initial;
    buffer.one_time = false;
    buffer.in_render_pass = false;
    buffer.pipeline = null;
    buffer.pipeline_kind = null;
    buffer.viewport_set = false;
    buffer.scissor_set = false;
    buffer.index_buffer = false;
    buffer.kinds = .{};
    buffer.refs.clearRetainingCapacity();
}

fn asResource(value: anytype) Resource {
    if (@TypeOf(value) == Resource) return value;
    return value.handle;
}

fn resourceEql(a: anytype, b: anytype) bool {
    const lhs = asResource(a);
    const rhs = asResource(b);
    return lhs.kind == rhs.kind and lhs.handle == rhs.handle;
}

fn fits(offset: u64, size: u64, total: u64) bool {
    if (offset > total) return false;
    if (size == 0) return true;
    return size <= total - offset;
}

fn findBinding(
    bindings: []const DescriptorSetLayout.Binding,
    index: u32,
) ?DescriptorSetLayout.Binding {
    for (bindings) |binding| {
        if (binding.binding == index) return binding;
    }
    return null;
}

fn stageFlagsAny(flags: Shader.StageFlags) bool {
    return flags.vertex or flags.fragment or flags.compute;
}

fn stageFlagsIntersect(a: Shader.StageFlags, b: Shader.StageFlags) bool {
    return (a.vertex and b.vertex) or
        (a.fragment and b.fragment) or
        (a.compute and b.compute);
}

fn pushConstantOverlap(
    a: PipelineLayout.PushConstantRange,
    b: PipelineLayout.PushConstantRange,
) bool {
    const a_end = @as(u64, a.offset) + a.size;
    const b_end = @as(u64, b.offset) + b.size;
    return a.offset < b_end and b.offset < a_end;
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

fn needsDisplay(platform: Surface.Platform) bool {
    return switch (platform) {
        .xlib, .xcb, .wayland => true,
        else => false,
    };
}

fn isDepthStencil(format: Resource.Format) bool {
    return switch (format) {
        .depth16_unorm,
        .depth24_plus_stencil8,
        .depth32_float,
        .stencil8,
        => true,
        else => false,
    };
}

fn isSrgb(format: Resource.Format) bool {
    return switch (format) {
        .r8g8b8a8_srgb, .b8g8r8a8_srgb => true,
        else => false,
    };
}

fn formatSupports(
    format: Resource.Format,
    usage: Texture.Usage,
) bool {
    if (std.meta.eql(usage, Texture.Usage{})) return false;
    const depth = isDepthStencil(format);
    if (usage.color_attachment and depth) return false;
    if (usage.depth_stencil_attachment and !depth) return false;
    if (usage.storage and (depth or isSrgb(format))) return false;
    return true;
}

fn swapchainFormatSupported(format: Resource.Format) bool {
    return switch (format) {
        .r8g8b8a8_unorm,
        .r8g8b8a8_srgb,
        .b8g8r8a8_unorm,
        .b8g8r8a8_srgb,
        .r16g16b16a16_float,
        => true,
        else => false,
    };
}

fn vertexFormatSize(format: PipelineGraphics.VertexFormat) u32 {
    return switch (format) {
        .u8x2, .i8x2 => 2,
        .f32, .u32, .u8x4, .i8x4, .u8x4_unorm => 4,
        .u16x2, .u16x2_unorm, .i16x2 => 4,
        .f32x2, .u32x2, .u16x4, .u16x4_unorm, .i16x4 => 8,
        .f32x3, .u32x3 => 12,
        .f32x4, .u32x4 => 16,
    };
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

    const shift = @as(u5, @intCast(@min(region.mip, 31)));
    const mip_width = @max(@as(u32, 1), desc.extent.width >> shift);
    const mip_height = @max(@as(u32, 1), desc.extent.height >> shift);
    const mip_depth = @max(@as(u32, 1), desc.extent.depth >> shift);

    if (!fits(region.offset[0], region.extent[0], mip_width)) {
        return error.InvalidDesc;
    }
    if (!fits(region.offset[1], region.extent[1], mip_height)) {
        return error.InvalidDesc;
    }
    if (!fits(region.offset[2], region.extent[2], mip_depth)) {
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

fn writeOut(
    comptime T: type,
    out: ?[]u8,
    value: T,
) Device.QueryError!void {
    const bytes = out orelse return error.InsufficientBuffer;
    if (bytes.len < @sizeOf(T)) return error.InsufficientBuffer;
    @memcpy(bytes[0..@sizeOf(T)], std.mem.asBytes(&value));
}

// -------------------------------------------------------------------
// Virtual tables
// -------------------------------------------------------------------

const device_vtable = Device.VTable{
    .create = deviceCreate,
    .destroy = deviceDestroy,
    .map = deviceMap,
    .unmap = deviceUnmap,
    .record = deviceRecord,
    .query = deviceQuery,
    .get_queue = deviceGetQueue,
    .device_op = deviceOp,
};

const queue_vtable = Queue.VTable{
    .submit = queueSubmit,
    .present = queuePresent,
    .wait = queueWait,
};

fn backendFrom(vptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(vptr));
}

fn deviceCreate(
    vptr: *anyopaque,
    desc: *const Resource.Desc,
    ret_addr: usize,
) Device.CreateError!Resource {
    return backendFrom(vptr).createResource(desc, ret_addr);
}

fn deviceDestroy(
    vptr: *anyopaque,
    handle: Resource,
    ret_addr: usize,
) void {
    _ = ret_addr;
    const self = backendFrom(vptr);
    if (self.lookup(handle)) |entry| {
        if (entry.kind == .texture and
            entry.data.texture.owner != null)
        {
            self.report("destroy of a swapchain-owned texture", .{});
            return;
        }
        if (entry.kind == .command_buffer) {
            self.removeChild(handle);
        }
        if (entry.kind == .surface and self.surfaceInUse(handle)) {
            self.report("destroy of a surface used by a swapchain", .{});
            return;
        }
        self.destroyEntry(handle);
        return;
    }
    self.report(
        "destroy of invalid {s} handle",
        .{@tagName(handle.kind)},
    );
}

fn deviceMap(
    vptr: *anyopaque,
    handle: Resource,
    offset: u64,
    size: u64,
    flags: Device.MapFlags,
    ret_addr: usize,
) Device.MapError![]u8 {
    return backendFrom(vptr).mapResource(
        handle,
        offset,
        size,
        flags,
        ret_addr,
    );
}

fn deviceUnmap(
    vptr: *anyopaque,
    handle: Resource,
    dirty: []const Device.DirtyRange,
    ret_addr: usize,
) void {
    backendFrom(vptr).unmapResource(handle, dirty, ret_addr);
}

fn deviceRecord(
    vptr: *anyopaque,
    handle: Resource,
    cmd: *const Device.Command,
    ret_addr: usize,
) Device.RecordError!void {
    return backendFrom(vptr).recordCommand(handle, cmd, ret_addr);
}

fn deviceQuery(
    vptr: *anyopaque,
    feature: Device.Feature,
    feature_index: u32,
    handle: ?Resource,
    out: ?[]u8,
    ret_addr: usize,
) Device.QueryError!Device.Support {
    return backendFrom(vptr).query(
        feature,
        feature_index,
        handle,
        out,
        ret_addr,
    );
}

fn deviceGetQueue(
    vptr: *anyopaque,
    index: u32,
    ret_addr: usize,
) Device.GetQueueError!Queue {
    return backendFrom(vptr).getQueue(index, ret_addr);
}

fn deviceOp(
    vptr: *anyopaque,
    op: Device.DeviceOp,
    out: ?[]u8,
    ret_addr: usize,
) Device.DeviceOpError!void {
    return backendFrom(vptr).deviceOp(op, out, ret_addr);
}

fn queueFrom(vptr: *anyopaque) *QueueState {
    return @ptrCast(@alignCast(vptr));
}

fn queueSubmit(
    vptr: *anyopaque,
    submits: []const Queue.SubmitInfo,
    ret_addr: usize,
) Queue.SubmitError!void {
    const state = queueFrom(vptr);
    return state.backend.queueSubmit(state, submits, ret_addr);
}

fn queuePresent(
    vptr: *anyopaque,
    swapchain: Resource,
    image_index: u32,
    wait_ops: []const Queue.SyncOp,
    ret_addr: usize,
) Queue.PresentError!Queue.PresentResult {
    const state = queueFrom(vptr);
    return state.backend.queuePresent(
        state,
        swapchain,
        image_index,
        wait_ops,
        ret_addr,
    );
}

fn queueWait(
    vptr: *anyopaque,
    syncs: []const Queue.SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
    ret_addr: usize,
) Queue.WaitError!Queue.WaitResult {
    return queueFrom(vptr).backend.queueWait(
        syncs,
        wait_any,
        timeout_ns,
        ret_addr,
    );
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

const testing = std.testing;

fn testBackend() !*Backend {
    return init(testing.allocator, .{ .verbose = false });
}

test "null backend buffer mapping and resource lifetime" {
    const backend = try testBackend();
    const device = backend.device();

    const buffer = try Buffer.create(device, .{
        .size = 16,
        .usage = .{ .uniform = true, .transfer_dst = true },
        .memory = .host,
        .label = "host buffer",
    });

    const memory = try buffer.map(0, 16, .{ .read = true, .write = true });
    @memcpy(memory[0..4], "test");
    buffer.unmap(&.{.{ .offset = 0, .size = 4 }});

    const read = try buffer.map(0, 16, .{ .read = true });
    try testing.expectEqualStrings("test", read[0..4]);
    buffer.unmap(&.{});

    buffer.destroy();
    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend reports mapping misuse" {
    const backend = try testBackend();
    const device = backend.device();

    const device_buffer = try Buffer.create(device, .{
        .size = 8,
        .usage = .{ .storage = true },
        .memory = .device,
    });

    try testing.expectError(
        error.NotHostVisible,
        device_buffer.map(0, 8, .{ .read = true }),
    );

    const host = try Buffer.create(device, .{
        .size = 8,
        .usage = .{ .uniform = true },
        .memory = .host,
    });

    try testing.expectError(
        error.OutOfRange,
        host.map(4, 8, .{ .read = true }),
    );

    const memory = try host.map(0, 8, .{ .read = true });
    _ = memory;
    try testing.expectError(
        error.AlreadyMapped,
        host.map(0, 8, .{ .read = true }),
    );
    host.unmap(&.{});

    device_buffer.destroy();
    host.destroy();
    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend validates command buffer lifecycle" {
    const backend = try testBackend();
    const device = backend.device();
    const queue = try device.getQueue(0);

    const pool = try CommandPool.create(device, .{});
    const buffer = try Buffer.create(device, .{
        .size = 16,
        .usage = .{ .transfer_dst = true },
        .memory = .device,
    });
    const command_buffer = try CommandBuffer.create(device, .{
        .pool = pool,
    });

    try testing.expectError(
        error.CommandBufferState,
        command_buffer.record(&.{ .fill_buffer = .{ .buffer = buffer, .size = 16 } }),
    );

    try command_buffer.record(&.{ .begin = .{} });
    try command_buffer.record(&.{ .fill_buffer = .{
        .buffer = buffer,
        .size = 16,
        .value = 0xdeadbeef,
    } });
    try testing.expectError(
        error.CommandBufferState,
        command_buffer.record(&.{ .begin = .{} }),
    );
    try command_buffer.record(&.{ .end = {} });

    const fence = try Fence.create(device, .{});
    try queue.submit(&.{.{ .commands = &.{command_buffer}, .fence = fence }});
    const wait = try queue.wait(&.{.{ .fence = fence }}, false, 0);
    try testing.expectEqual(@as(usize, 0), wait.index);

    fence.destroy();
    command_buffer.destroy();
    buffer.destroy();
    pool.destroy();

    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend rejects submit of an unended buffer" {
    const backend = try testBackend();
    const device = backend.device();
    const queue = try device.getQueue(0);

    const pool = try CommandPool.create(device, .{});
    const buffer = try CommandBuffer.create(device, .{ .pool = pool });
    try buffer.record(&.{ .begin = .{} });

    try testing.expectError(
        error.InvalidDesc,
        queue.submit(&.{.{ .commands = &.{buffer} }}),
    );

    buffer.destroy();
    pool.destroy();

    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend queries and descriptor validation" {
    const backend = try testBackend();
    const device = backend.device();

    var limits: Device.Limits = undefined;
    const support = try device.query(
        .limits,
        0,
        null,
        std.mem.asBytes(&limits),
    );
    try testing.expectEqual(Device.Support.supported, support);
    try testing.expect(limits.max_texture_extent_2d > 0);

    try testing.expectError(
        error.InsufficientBuffer,
        device.query(.limits, 0, null, null),
    );

    const layout = try DescriptorSetLayout.create(device, .{
        .bindings = &.{.{
            .binding = 0,
            .kind = .uniform_buffer,
            .stages = .{ .vertex = true },
        }},
    });
    const set = try DescriptorSet.create(device, .{ .layout = layout });
    const buffer = try Buffer.create(device, .{
        .size = 16,
        .usage = .{ .uniform = true },
        .memory = .device,
    });

    try set.update(&.{.{ .buffer = .{
        .binding = 0,
        .buffer = buffer,
        .size = 16,
    } }});

    try testing.expectError(
        error.InvalidDesc,
        set.update(&.{.{ .buffer = .{
            .binding = 1,
            .buffer = buffer,
        } }}),
    );

    buffer.destroy();
    set.destroy();
    layout.destroy();

    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend reports leaks at deinit" {
    const backend = try testBackend();
    const device = backend.device();

    _ = try Buffer.create(device, .{
        .size = 4,
        .usage = .{ .storage = true },
        .memory = .device,
        .label = "leaked",
    });

    const status = backend.deinit();
    try testing.expectEqual(@as(usize, 1), status.leaks);
    try testing.expect(!status.ok());
}

test "null backend reports double destroy" {
    const backend = try testBackend();
    const device = backend.device();

    const buffer = try Buffer.create(device, .{
        .size = 4,
        .usage = .{ .storage = true },
        .memory = .device,
    });

    buffer.destroy();
    buffer.destroy();

    const status = backend.deinit();
    try testing.expectEqual(@as(usize, 1), status.violations);
    try testing.expect(!status.ok());
}

test "null backend enforces queue family work kinds" {
    const families = [_]QueueFamily{.{
        .kinds = .{ .transfer = true },
    }};
    const backend = try init(testing.allocator, .{
        .queue_families = &families,
        .verbose = false,
    });
    const device = backend.device();
    const queue = try device.getQueue(0);

    const layout = try PipelineLayout.create(device, .{});
    const shader = try Shader.create(device, .{
        .code = &.{ 0, 0, 0, 0 },
        .stage = .compute,
    });
    const pipeline = try PipelineCompute.create(device, .{
        .layout = layout,
        .shader = shader,
    });
    const pool = try CommandPool.create(device, .{});
    const commands = try CommandBuffer.create(device, .{ .pool = pool });
    try commands.record(&.{ .begin = .{} });
    try commands.record(&.{ .bind_pipeline = .{ .compute = pipeline } });
    try commands.record(&.{ .dispatch = .{ .groups = .{ 1, 1, 1 } } });
    try commands.record(&.{ .end = {} });

    const semaphore = try Semaphore.create(device, .{});
    try testing.expectError(
        error.InvalidDesc,
        queue.submit(&.{.{
            .commands = &.{commands},
            .signals = &.{.{ .semaphore = semaphore }},
        }}),
    );
    try testing.expectError(
        error.Timeout,
        queue.wait(
            &.{.{ .semaphore = semaphore }},
            false,
            0,
        ),
    );

    const surface = try Surface.create(device, .{
        .window = 1,
        .platform = .win32,
    });
    const swapchain = try Swapchain.create(device, .{
        .surface = surface,
        .format = .r8g8b8a8_unorm,
        .extent = .{ .width = 4, .height = 4 },
        .images = 1,
    });
    const image_index = try swapchain.acquire(null, null, 0);
    try testing.expectError(
        error.InvalidDesc,
        queue.present(
            swapchain,
            image_index,
            &.{},
        ),
    );

    swapchain.destroy();
    surface.destroy();
    semaphore.destroy();
    commands.destroy();
    pool.destroy();
    pipeline.destroy();
    shader.destroy();
    layout.destroy();

    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "null backend acquire failure leaves sync unsignalled" {
    const backend = try testBackend();
    const device = backend.device();
    const queue = try device.getQueue(0);

    const surface = try Surface.create(device, .{
        .window = 1,
        .platform = .win32,
    });
    const swapchain = try Swapchain.create(device, .{
        .surface = surface,
        .format = .r8g8b8a8_unorm,
        .extent = .{ .width = 4, .height = 4 },
        .images = 1,
    });
    const semaphore = try Semaphore.create(device, .{ .kind = .timeline });
    const fence = try Fence.create(device, .{});

    _ = try swapchain.acquire(null, null, 0);
    try testing.expectError(
        error.InvalidState,
        swapchain.acquire(semaphore, fence, 0),
    );
    try testing.expectError(
        error.Timeout,
        queue.wait(&.{.{ .timeline = .{
            .semaphore = semaphore,
            .value = 1,
        } }}, false, 0),
    );
    try testing.expectError(
        error.Timeout,
        queue.wait(&.{.{ .fence = fence }}, false, 0),
    );

    fence.destroy();
    semaphore.destroy();
    swapchain.destroy();
    surface.destroy();

    const status = backend.deinit();
    try testing.expectEqual(@as(usize, 1), status.violations);
    try testing.expectEqual(@as(usize, 0), status.leaks);
}
