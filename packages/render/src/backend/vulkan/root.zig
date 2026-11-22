//! A Vulkan implementation of the RHI.
//!
//! The backend owns a `VkInstance`, one physical
//! device, one logical device and one queue per queue
//! family. RHI handles are indices into a registry of
//! backend state; the generation counter in every handle
//! makes stale handles detectable.
//!
//! The implementation loads `vulkan-1.dll`,
//! `libvulkan.so.1` or `libvulkan.dylib` at runtime, so
//! no link-time dependency on a Vulkan loader is
//! required.
//!
//! Rendering uses `VK_KHR_dynamic_rendering` (core in
//! Vulkan 1.3). Graphics pipelines therefore declare
//! their attachment formats at creation time, and
//! `Device.Command.begin_render_pass` moves attachments
//! into the optimal layout before it begins.
//!
//! Secondary command buffers are recorded into host
//! memory and replayed into the primary buffer when
//! `execute_commands` is recorded. This keeps the RHI
//! semantics, where render passes may begin inside a
//! secondary buffer, without requiring the fixed
//! inheritance information native secondary buffers need
//! for dynamic rendering.
//!
//! Image layouts are tracked once per texture rather
//! than per subresource. A barrier that names a range
//! updates the layout of the whole texture.

const std = @import("std");
const builtin = @import("builtin");
const rhi = @import("rhi");
const vk = @import("vk");

const format = @import("format.zig");
const state = @import("state.zig");
const resource = @import("resource.zig");
const command = @import("command.zig");
const device_op = @import("device_op.zig");
const queue = @import("queue.zig");

const Device = rhi.Device;
const Queue = rhi.Queue;
const Resource = rhi.Resource;

const log = std.log.scoped(.vulkan_rhi);

/// Errors returned by `init`.
pub const InitError = std.mem.Allocator.Error || error{
    /// No Vulkan loader library could be opened.
    LoaderUnavailable,
    /// The loader advertises an API older than 1.2.
    ApiUnsupported,
    /// No physical device can run the backend.
    NoPhysicalDevice,
    /// The chosen device exposes no queue family.
    NoQueueFamily,
    /// A Vulkan call failed during initialization.
    VulkanFailure,
};

/// Behaviour of one queue family.
pub const FamilyState = struct {
    /// Work kinds accepted by the family.
    kinds: Device.QueueKind,
    /// Queue flags reported by the driver.
    vk_flags: vk.QueueFlags,
    /// Present support for the most recent surface,
    /// when one has been created.
    ///
    /// Before the first surface, graphics families
    /// are assumed to support presentation.
    present: bool = false,
};

/// One queue exposed by the backend.
pub const QueueState = struct {
    /// Backend that owns the queue.
    backend: *Backend,
    /// Native queue.
    queue: vk.Queue,
    /// Family the queue belongs to.
    family_index: u32,
    /// Index inside the family; always zero.
    queue_index: u32,
};

/// Backend configuration.
pub const Options = struct {
    /// Name reported to the driver.
    app_name: [*:0]const u8 = "carn",
    /// Version of the application.
    app_version: u32 = 0,
    /// Engine name reported to the driver.
    engine_name: [*:0]const u8 = "carn",
    /// Version of the engine.
    engine_version: u32 = 0,
    /// Highest API version the backend may request.
    api_version: u32 = vk.API_VERSION_1_3.toU32(),
    /// Extra instance extensions to enable.
    instance_extensions: []const [*:0]const u8 = &.{},
    /// Extra device extensions to enable.
    device_extensions: []const [*:0]const u8 = &.{},
    /// Enable `VK_LAYER_KHRONOS_validation` when present.
    validation: bool = false,
    /// Log initialization decisions and violations.
    verbose: bool = false,
    /// Force a specific index in the physical device list.
    physical_device_index: ?usize = null,
};

/// Result of `Backend.deinit`.
pub const Status = struct {
    /// Resources that were still alive.
    leaks: usize = 0,
    /// Contract violations observed at runtime.
    violations: usize = 0,

    /// Reports whether shutdown was clean.
    ///
    /// # Return
    /// True when both counters are zero.
    pub fn ok(self: Status) bool {
        return self.leaks == 0 and self.violations == 0;
    }
};

/// Loads a Vulkan implementation and creates its device.
///
/// # Args
/// - `allocator`: owns the backend and its bookkeeping.
/// - `options`: application names, versions and
///   extensions.
///
/// # Return
/// A backend whose address is stable. Release it with
/// `deinit`.
///
/// # Error
/// - `error.OutOfMemory`: host allocation fails.
/// - `error.LoaderUnavailable`: no Vulkan library was
///   found.
/// - `error.ApiUnsupported`: the loader or the selected
///   device reports an API older than 1.2.
/// - `error.NoPhysicalDevice`: no device supports
///   graphics, transfer and swapchain work.
/// - `error.NoQueueFamily`: the selected device exposes
///   no queue family.
/// - `error.VulkanFailure`: a Vulkan call failed.
///
/// # Note
/// The backend requires Vulkan 1.2. Timeline
/// semaphores are used when the device exposes the
/// feature; creating a timeline semaphore otherwise
/// reports `error.Unsupport`. Dynamic rendering is
/// used when the device supports it; graphics
/// pipelines report `error.Unsupport` otherwise.
pub fn init(
    allocator: std.mem.Allocator,
    options: Options,
) InitError!*Backend {
    const self = try allocator.create(Backend);
    errdefer allocator.destroy(self);

    self.* = .{
        .allocator = allocator,
        .options = options,
        .library = undefined,
        .base = undefined,
        .instance = undefined,
        .instance_api = undefined,
        .physical_device = undefined,
        .vk_device = undefined,
        .device_api = undefined,
        .api_version = 0,
        .core_dynamic_rendering = false,
        .timeline_supported = false,
        .dynamic_rendering_supported = false,
        .features = undefined,
        .properties = undefined,
        .memory_properties = undefined,
        .limits = undefined,
        .families = &.{},
        .queues = &.{},
        .entries = .empty,
        .free_head = state.no_index,
        .violations = 0,
    };

    self.library = openLibrary() catch |err| {
        if (options.verbose) {
            log.err("cannot open a Vulkan loader: {s}", .{@errorName(err)});
        }
        return error.LoaderUnavailable;
    };
    errdefer self.library.close();

    const get_instance_proc_addr = self.library.lookup(
        vk.PfnGetInstanceProcAddr,
        "vkGetInstanceProcAddr",
    ) orelse return error.LoaderUnavailable;
    self.base = vk.BaseWrapper.load(get_instance_proc_addr);

    const loader_version = if (self.base.dispatch.vkEnumerateInstanceVersion != null)
        self.base.enumerateInstanceVersion() catch
            vk.API_VERSION_1_0.toU32()
    else
        vk.API_VERSION_1_0.toU32();
    if (loader_version < vk.API_VERSION_1_2.toU32()) return error.ApiUnsupported;

    const api_version = @min(options.api_version, loader_version);
    if (api_version < vk.API_VERSION_1_2.toU32()) return error.ApiUnsupported;

    var instance_extensions: std.ArrayListUnmanaged([*:0]const u8) = .empty;
    defer instance_extensions.deinit(allocator);
    try self.collectInstanceExtensions(allocator, &instance_extensions);

    var layer_names: std.ArrayListUnmanaged([*:0]const u8) = .empty;
    defer layer_names.deinit(allocator);
    if (options.validation) {
        try self.collectValidationLayer(allocator, &layer_names);
    }

    const app_info = vk.ApplicationInfo{
        .p_application_name = options.app_name,
        .application_version = options.app_version,
        .p_engine_name = options.engine_name,
        .engine_version = options.engine_version,
        .api_version = api_version,
    };

    var create_flags = vk.InstanceCreateFlags{};
    if (hasExtension(instance_extensions.items, "VK_KHR_portability_enumeration")) {
        create_flags.enumerate_portability_khr = true;
    }

    const instance_info = vk.InstanceCreateInfo{
        .flags = create_flags,
        .p_application_info = &app_info,
        .enabled_layer_count = @intCast(layer_names.items.len),
        .pp_enabled_layer_names = if (layer_names.items.len == 0)
            null
        else
            layer_names.items.ptr,
        .enabled_extension_count = @intCast(instance_extensions.items.len),
        .pp_enabled_extension_names = if (instance_extensions.items.len == 0)
            null
        else
            instance_extensions.items.ptr,
    };

    self.instance = self.base.createInstance(&instance_info, null) catch |err| {
        log.err("vkCreateInstance failed: {s}", .{@errorName(err)});
        return error.VulkanFailure;
    };

    self.instance_api = vk.InstanceWrapper.load(
        self.instance,
        get_instance_proc_addr,
    );
    errdefer self.instance_api.destroyInstance(self.instance, null);
    self.api_version = api_version;

    try self.selectPhysicalDevice(allocator);
    try self.createDevice(allocator);

    return self;
}

/// Destroys the backend and reports leaked resources.
///
/// # Return
/// Leak and violation counters observed at shutdown.
///
/// # Note
/// Every live resource is released. The backend storage
/// is freed; device and queue values obtained from it
/// are invalid afterwards.
fn deinitBackend(self: *Backend) Status {
    const leaks = self.liveResourceCount();
    if (self.options.verbose and leaks > 0) {
        log.err("{d} resource(s) leaked", .{leaks});
    }

    self.device_api.deviceWaitIdle(self.vk_device) catch {};
    resource.releaseAll(self);

    self.allocator.free(self.families);
    self.allocator.free(self.queues);

    self.device_api.destroyDevice(self.vk_device, null);
    self.instance_api.destroyInstance(self.instance, null);
    self.library.close();

    const status = Status{
        .leaks = leaks,
        .violations = self.violations,
    };

    const allocator = self.allocator;
    self.* = undefined;
    allocator.destroy(self);
    return status;
}

/// A Vulkan backend instance.
pub const Backend = struct {
    allocator: std.mem.Allocator,
    options: Options,
    library: Library,
    base: vk.BaseWrapper,
    instance: vk.Instance,
    instance_api: vk.InstanceWrapper,
    physical_device: vk.PhysicalDevice,
    vk_device: vk.Device,
    device_api: vk.DeviceWrapper,
    api_version: u32,
    core_dynamic_rendering: bool,
    timeline_supported: bool,
    dynamic_rendering_supported: bool,
    features: vk.PhysicalDeviceFeatures,
    properties: vk.PhysicalDeviceProperties,
    memory_properties: vk.PhysicalDeviceMemoryProperties,
    limits: Device.Limits,
    families: []FamilyState,
    queues: []QueueState,
    entries: std.ArrayListUnmanaged(state.Entry) = .empty,
    free_head: u32 = state.no_index,
    violations: usize = 0,

    /// Returns the RHI device implemented by the backend.
    ///
    /// # Return
    /// A device value that borrows this backend.
    pub fn device(self: *Backend) Device {
        return .{
            .vptr = self,
            .vtable = &device_vtable,
        };
    }

    /// Destroys the backend and reports leaked resources.
    ///
    /// # Return
    /// Leak and violation counters observed at shutdown.
    ///
    /// # Note
    /// Every live resource is released. The backend storage
    /// is freed; device and queue values obtained from it
    /// are invalid afterwards.
    pub fn deinit(self: *Backend) Status {
        return deinitBackend(self);
    }

    /// Returns the number of contract violations.
    ///
    /// # Return
    /// Violations counted since `init`.
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

    /// Logs and counts a contract violation.
    ///
    /// # Args
    /// - `fmt`: format string.
    /// - `args`: format arguments.
    pub fn report(
        self: *Backend,
        comptime fmt: []const u8,
        args: anytype,
    ) void {
        self.violations += 1;
        if (self.options.verbose) log.err(fmt, args);
    }

    /// Resolves an RHI handle to its registry entry.
    ///
    /// # Args
    /// - `value`: typed resource or raw handle.
    ///
    /// # Return
    /// The live entry, or null when the handle is stale
    /// or has the wrong kind.
    pub fn lookup(self: *Backend, value: anytype) ?*state.Entry {
        const handle = asResource(value);
        const low: u32 = @truncate(handle.handle);
        if (low == 0) return null;
        const index = low - 1;
        if (@as(usize, index) >= self.entries.items.len) return null;
        const entry = &self.entries.items[index];
        if (!entry.live) return null;
        if (entry.kind != handle.kind) return null;
        const generation: u32 = @truncate(handle.handle >> 32);
        if (entry.generation != generation) return null;
        return entry;
    }

    /// Resolves a handle and checks its kind.
    ///
    /// # Args
    /// - `value`: typed resource or raw handle.
    /// - `kind`: expected kind.
    ///
    /// # Return
    /// The matching state, or null.
    pub fn get(
        self: *Backend,
        value: anytype,
        comptime kind: Resource.Kind,
    ) ?*state.StateFor(kind) {
        const entry = self.lookup(value) orelse return null;
        if (entry.kind != kind) return null;
        return &@field(entry.data, @tagName(kind));
    }

    /// Allocates a registry entry for a new resource.
    ///
    /// # Args
    /// - `kind`: resource kind.
    /// - `data`: native state of the resource.
    /// - `label`: borrowed debug label.
    ///
    /// # Return
    /// A fresh handle that identifies the entry.
    ///
    /// # Error
    /// - `error.OutOfMemory`: the registry or label
    ///   cannot be allocated.
    pub fn newEntry(
        self: *Backend,
        kind: Resource.Kind,
        data: state.Data,
        label: []const u8,
    ) std.mem.Allocator.Error!Resource {
        const owned_label: ?[]u8 = if (label.len == 0)
            null
        else
            try self.allocator.dupe(u8, label);
        errdefer if (owned_label) |owned| self.allocator.free(owned);

        var index: u32 = undefined;
        var generation: u32 = 1;

        if (self.free_head != state.no_index) {
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

    /// Marks an entry dead and links it into the free
    /// list. Native state must already be released.
    ///
    /// # Args
    /// - `index`: registry slot.
    pub fn recycleEntry(self: *Backend, index: u32) void {
        if (@as(usize, index) >= self.entries.items.len) return;
        const entry = &self.entries.items[index];
        if (!entry.live) return;

        if (entry.label) |label| self.allocator.free(label);

        entry.live = false;
        entry.label = null;
        entry.generation +%= 1;
        if (entry.generation == 0) entry.generation = 1;
        entry.next_free = self.free_head;
        self.free_head = index;
    }

    fn collectInstanceExtensions(
        self: *Backend,
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged([*:0]const u8),
    ) InitError!void {
        const available = self.base.enumerateInstanceExtensionPropertiesAlloc(
            null,
            allocator,
        ) catch |err| {
            log.err(
                "vkEnumerateInstanceExtensionProperties failed: {s}",
                .{@errorName(err)},
            );
            return error.VulkanFailure;
        };
        defer allocator.free(available);

        const wanted = [_][*:0]const u8{
            "VK_KHR_surface",
            "VK_KHR_win32_surface",
            "VK_KHR_xlib_surface",
            "VK_KHR_xcb_surface",
            "VK_KHR_wayland_surface",
            "VK_KHR_android_surface",
            "VK_EXT_metal_surface",
            "VK_EXT_headless_surface",
            "VK_KHR_portability_enumeration",
        };
        for (wanted) |name| {
            if (findExtension(available, std.mem.span(name))) {
                try out.append(allocator, name);
            }
        }
        for (self.options.instance_extensions) |name| {
            if (hasExtension(out.items, std.mem.span(name))) continue;
            try out.append(allocator, name);
        }
    }

    fn collectValidationLayer(
        self: *Backend,
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged([*:0]const u8),
    ) InitError!void {
        const layers = self.base.enumerateInstanceLayerPropertiesAlloc(
            allocator,
        ) catch return;
        defer allocator.free(layers);

        for (layers) |layer| {
            const name = std.mem.sliceTo(&layer.layer_name, 0);
            if (std.mem.eql(u8, name, "VK_LAYER_KHRONOS_validation")) {
                try out.append(
                    allocator,
                    "VK_LAYER_KHRONOS_validation",
                );
                return;
            }
        }
    }

    fn selectPhysicalDevice(
        self: *Backend,
        allocator: std.mem.Allocator,
    ) InitError!void {
        const devices = self.instance_api.enumeratePhysicalDevicesAlloc(
            self.instance,
            allocator,
        ) catch |err| {
            log.err(
                "vkEnumeratePhysicalDevices failed: {s}",
                .{@errorName(err)},
            );
            return error.VulkanFailure;
        };
        defer allocator.free(devices);

        var best: ?vk.PhysicalDevice = null;
        var best_score: i32 = -1;

        for (devices, 0..) |candidate, index| {
            if (self.options.physical_device_index) |wanted| {
                if (wanted != index) continue;
            }

            const properties = self.instance_api.getPhysicalDeviceProperties(
                candidate,
            );
            if (properties.api_version < vk.API_VERSION_1_2.toU32()) continue;

            const extensions = self.instance_api
                .enumerateDeviceExtensionPropertiesAlloc(
                candidate,
                null,
                allocator,
            ) catch continue;
            defer allocator.free(extensions);

            if (!findExtension(extensions, "VK_KHR_swapchain")) {
                continue;
            }

            const has_dynamic_rendering =
                @min(properties.api_version, self.api_version) >=
                vk.API_VERSION_1_3.toU32() or
                findExtension(extensions, "VK_KHR_dynamic_rendering");
            if (!has_dynamic_rendering) continue;

            if (!self.deviceHasQueueFamily(candidate, allocator)) {
                continue;
            }

            var score: i32 = switch (properties.device_type) {
                .discrete_gpu => 100,
                .integrated_gpu => 50,
                .virtual_gpu => 20,
                .cpu => 10,
                else => 5,
            };
            if (self.options.physical_device_index != null) {
                score += 1000;
            }

            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }

        self.physical_device = best orelse return error.NoPhysicalDevice;
        self.properties = self.instance_api.getPhysicalDeviceProperties(
            self.physical_device,
        );
        self.memory_properties = self.instance_api
            .getPhysicalDeviceMemoryProperties(self.physical_device);
        self.limits = limitsFromProperties(self.properties.limits);

        if (self.options.verbose) {
            log.info(
                "selected {s} (api {d}.{d}.{d})",
                .{
                    std.mem.sliceTo(&self.properties.device_name, 0),
                    vkVersionMajor(self.properties.api_version),
                    vkVersionMinor(self.properties.api_version),
                    vkVersionPatch(self.properties.api_version),
                },
            );
        }
    }

    fn deviceHasQueueFamily(
        self: *Backend,
        physical_device: vk.PhysicalDevice,
        allocator: std.mem.Allocator,
    ) bool {
        const families = self.instance_api
            .getPhysicalDeviceQueueFamilyPropertiesAlloc(
            physical_device,
            allocator,
        ) catch return false;
        defer allocator.free(families);

        var graphics = false;
        var compute = false;
        var transfer = false;
        for (families) |family| {
            if (family.queue_flags.graphics) graphics = true;
            if (family.queue_flags.compute) compute = true;
            if (family.queue_flags.transfer or
                family.queue_flags.graphics or
                family.queue_flags.compute)
            {
                transfer = true;
            }
        }
        return graphics and compute and transfer;
    }

    fn createDevice(
        self: *Backend,
        allocator: std.mem.Allocator,
    ) InitError!void {
        const families = self.instance_api
            .getPhysicalDeviceQueueFamilyPropertiesAlloc(
            self.physical_device,
            allocator,
        ) catch |err| {
            log.err(
                "vkGetPhysicalDeviceQueueFamilyProperties failed: {s}",
                .{@errorName(err)},
            );
            return error.VulkanFailure;
        };
        defer allocator.free(families);
        if (families.len == 0) return error.NoQueueFamily;

        self.families = try allocator.alloc(FamilyState, families.len);
        errdefer allocator.free(self.families);
        for (families, 0..) |family, i| {
            self.families[i] = .{
                .kinds = .{
                    .graphics = family.queue_flags.graphics,
                    .compute = family.queue_flags.compute,
                    .transfer = family.queue_flags.transfer or
                        family.queue_flags.graphics or
                        family.queue_flags.compute,
                    .present = family.queue_flags.graphics,
                },
                .vk_flags = family.queue_flags,
            };
        }

        const extensions = self.instance_api
            .enumerateDeviceExtensionPropertiesAlloc(
            self.physical_device,
            null,
            allocator,
        ) catch |err| {
            log.err(
                "vkEnumerateDeviceExtensionProperties failed: {s}",
                .{@errorName(err)},
            );
            return error.VulkanFailure;
        };
        defer allocator.free(extensions);

        var device_extensions: std.ArrayListUnmanaged([*:0]const u8) = .empty;
        defer device_extensions.deinit(allocator);

        try device_extensions.append(allocator, "VK_KHR_swapchain");

        const core_rendering =
            @min(self.properties.api_version, self.api_version) >=
            vk.API_VERSION_1_3.toU32();
        if (!core_rendering) {
            if (!findExtension(extensions, "VK_KHR_dynamic_rendering")) {
                log.err("device has no dynamic rendering support", .{});
                return error.NoPhysicalDevice;
            }
            try device_extensions.append(
                allocator,
                "VK_KHR_dynamic_rendering",
            );
        }
        self.core_dynamic_rendering = core_rendering;

        if (findExtension(extensions, "VK_KHR_portability_subset")) {
            try device_extensions.append(
                allocator,
                "VK_KHR_portability_subset",
            );
        }
        for (self.options.device_extensions) |name| {
            if (hasExtension(device_extensions.items, std.mem.span(name))) {
                continue;
            }
            try device_extensions.append(allocator, name);
        }

        var features12 = vk.PhysicalDeviceVulkan12Features{};
        var features13 = vk.PhysicalDeviceVulkan13Features{};
        var rendering_features =
            vk.PhysicalDeviceDynamicRenderingFeatures{};

        features12.p_next = if (core_rendering)
            @ptrCast(&features13)
        else
            @ptrCast(&rendering_features);

        var queried = vk.PhysicalDeviceFeatures2{
            .p_next = @ptrCast(&features12),
            .features = .{},
        };
        self.instance_api.getPhysicalDeviceFeatures2(
            self.physical_device,
            &queried,
        );

        self.timeline_supported =
            features12.timeline_semaphore == .true;

        var enabled13 = vk.PhysicalDeviceVulkan13Features{};
        var enabled12 = vk.PhysicalDeviceVulkan12Features{
            .timeline_semaphore = features12.timeline_semaphore,
        };
        var enabled_rendering =
            vk.PhysicalDeviceDynamicRenderingFeatures{};

        if (core_rendering) {
            enabled13.dynamic_rendering = features13.dynamic_rendering;
            enabled12.p_next = @ptrCast(&enabled13);
            self.dynamic_rendering_supported =
                features13.dynamic_rendering == .true;
        } else {
            enabled_rendering.dynamic_rendering =
                rendering_features.dynamic_rendering;
            enabled12.p_next = @ptrCast(&enabled_rendering);
            self.dynamic_rendering_supported =
                rendering_features.dynamic_rendering == .true;
        }

        var enabled_features = vk.PhysicalDeviceFeatures{};
        enabled_features.sampler_anisotropy =
            queried.features.sampler_anisotropy;
        enabled_features.fill_mode_non_solid =
            queried.features.fill_mode_non_solid;
        enabled_features.wide_lines = queried.features.wide_lines;
        enabled_features.depth_clamp = queried.features.depth_clamp;

        var features2 = vk.PhysicalDeviceFeatures2{
            .p_next = @ptrCast(&enabled12),
            .features = enabled_features,
        };

        const queue_infos = try allocator.alloc(
            vk.DeviceQueueCreateInfo,
            families.len,
        );
        defer allocator.free(queue_infos);
        var priority = [_]f32{1};
        for (families, 0..) |_, i| {
            queue_infos[i] = .{
                .queue_family_index = @intCast(i),
                .queue_count = 1,
                .p_queue_priorities = &priority,
            };
        }

        const create_info = vk.DeviceCreateInfo{
            .p_next = &features2,
            .queue_create_info_count = @intCast(queue_infos.len),
            .p_queue_create_infos = queue_infos.ptr,
            .enabled_extension_count = @intCast(device_extensions.items.len),
            .pp_enabled_extension_names = device_extensions.items.ptr,
        };

        self.features = enabled_features;

        self.vk_device = self.instance_api.createDevice(
            self.physical_device,
            &create_info,
            null,
        ) catch |err| {
            log.err("vkCreateDevice failed: {s}", .{@errorName(err)});
            return error.VulkanFailure;
        };

        const get_device_proc_addr: vk.PfnGetDeviceProcAddr = @ptrCast(
            self.base.getInstanceProcAddr(
                self.instance,
                "vkGetDeviceProcAddr",
            ) orelse return error.VulkanFailure,
        );
        self.device_api = vk.DeviceWrapper.load(
            self.vk_device,
            get_device_proc_addr,
        );
        errdefer self.device_api.destroyDevice(self.vk_device, null);

        self.queues = try allocator.alloc(QueueState, families.len);
        errdefer allocator.free(self.queues);
        for (families, 0..) |_, i| {
            self.queues[i] = .{
                .backend = self,
                .queue = self.device_api.getDeviceQueue(
                    self.vk_device,
                    @intCast(i),
                    0,
                ),
                .family_index = @intCast(i),
                .queue_index = 0,
            };
        }
    }

    /// Queries present support for a surface.
    ///
    /// # Args
    /// - `surface`: surface to test.
    /// - `family_index`: queue family index.
    ///
    /// # Return
    /// True when the family can present to the surface.
    pub fn familySupportsPresent(
        self: *Backend,
        surface: vk.SurfaceKHR,
        family_index: u32,
    ) bool {
        const supported = self.instance_api
            .getPhysicalDeviceSurfaceSupportKHR(
            self.physical_device,
            family_index,
            surface,
        ) catch return false;
        return supported == .true;
    }

    /// Refreshes the cached present flags for a surface.
    ///
    /// # Args
    /// - `surface`: newly created surface.
    pub fn refreshPresentSupport(
        self: *Backend,
        surface: vk.SurfaceKHR,
    ) void {
        for (self.families, 0..) |*family, i| {
            family.present = self.familySupportsPresent(
                surface,
                @intCast(i),
            );
        }
    }

    /// Picks a family that can run graphics work for a
    /// swapchain and present it.
    ///
    /// # Args
    /// - `surface`: surface to present to.
    ///
    /// # Return
    /// A graphics family that supports presentation.
    pub fn graphicsPresentFamily(
        self: *Backend,
        surface: vk.SurfaceKHR,
    ) ?u32 {
        for (self.families, 0..) |family, i| {
            if (!family.kinds.graphics) continue;
            if (!self.familySupportsPresent(surface, @intCast(i))) {
                continue;
            }
            return @intCast(i);
        }
        return null;
    }

    /// Selects a memory type for a resource.
    ///
    /// # Args
    /// - `type_bits`: allowed memory types from the
    ///   driver.
    /// - `required`: flags the type must have.
    /// - `preferred`: flags preferred but not required.
    ///
    /// # Return
    /// The selected type index, or null when no type
    /// matches.
    pub fn findMemoryType(
        self: *Backend,
        type_bits: u32,
        required: vk.MemoryPropertyFlags,
        preferred: vk.MemoryPropertyFlags,
    ) ?u32 {
        const count = @min(
            self.memory_properties.memory_type_count,
            vk.MAX_MEMORY_TYPES,
        );
        var fallback: ?u32 = null;
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            if ((type_bits & (@as(u32, 1) << @intCast(i))) == 0) {
                continue;
            }
            const flags =
                self.memory_properties.memory_types[i].property_flags;
            if (!flags.contains(required)) continue;
            if (flags.contains(preferred)) return i;
            if (fallback == null) fallback = i;
        }
        return fallback;
    }
};

fn encode(index: u32, generation: u32) u64 {
    return (@as(u64, generation) << 32) | (@as(u64, index) + 1);
}

/// Extracts the registry slot from a handle.
///
/// # Args
/// - `handle`: RHI handle.
///
/// # Return
/// The index stored in the low half.
pub fn entryIndex(handle: Resource) u32 {
    return @as(u32, @truncate(handle.handle)) - 1;
}

/// Normalises a typed resource to a raw handle.
///
/// # Args
/// - `value`: typed resource or raw handle.
///
/// # Return
/// The raw handle.
pub fn asResource(value: anytype) Resource {
    if (@TypeOf(value) == Resource) return value;
    return value.handle;
}

/// Compares two handles by kind and value.
///
/// # Args
/// - `a`: first resource.
/// - `b`: second resource.
///
/// # Return
/// True when both describe the same live slot.
pub fn resourceEql(a: anytype, b: anytype) bool {
    const lhs = asResource(a);
    const rhs = asResource(b);
    return lhs.kind == rhs.kind and lhs.handle == rhs.handle;
}

/// Reports whether a byte range fits in a resource.
///
/// # Args
/// - `offset`: first byte.
/// - `size`: number of bytes.
/// - `total`: resource size.
///
/// # Return
/// True when the range lies inside the resource.
pub fn fits(offset: u64, size: u64, total: u64) bool {
    if (offset > total) return false;
    if (size == 0) return true;
    return size <= total - offset;
}

const Library = if (builtin.os.tag == .windows) struct {
    const windows = std.os.windows;

    module: windows.HMODULE,

    extern "kernel32" fn LoadLibraryA(
        name: [*:0]const u8,
    ) callconv(.winapi) ?windows.HMODULE;
    extern "kernel32" fn GetProcAddress(
        module: windows.HMODULE,
        name: [*:0]const u8,
    ) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(
        module: windows.HMODULE,
    ) callconv(.winapi) windows.BOOL;

    fn open(name: [:0]const u8) !Library {
        const module = LoadLibraryA(name.ptr) orelse
            return error.FileNotFound;
        return .{ .module = module };
    }

    fn close(self: *Library) void {
        _ = FreeLibrary(self.module);
    }

    fn lookup(
        self: *Library,
        comptime T: type,
        name: [:0]const u8,
    ) ?T {
        const address = GetProcAddress(self.module, name.ptr) orelse
            return null;
        return @ptrCast(address);
    }
} else struct {
    inner: std.DynLib,

    fn open(name: [:0]const u8) !Library {
        return .{ .inner = try std.DynLib.open(name) };
    }

    fn close(self: *Library) void {
        self.inner.close();
    }

    fn lookup(
        self: *Library,
        comptime T: type,
        name: [:0]const u8,
    ) ?T {
        return self.inner.lookup(T, name);
    }
};

fn openLibrary() !Library {
    const candidates: []const [:0]const u8 = switch (builtin.os.tag) {
        .windows => &.{ "vulkan-1.dll", "vulkan-1" },
        .macos => &.{
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
        },
        else => &.{ "libvulkan.so.1", "libvulkan.so" },
    };
    for (candidates) |name| {
        if (Library.open(name)) |library| {
            return library;
        } else |_| {}
    }
    return error.FileNotFound;
}

fn hasExtension(
    extensions: []const [*:0]const u8,
    name: []const u8,
) bool {
    for (extensions) |extension| {
        if (std.mem.eql(u8, std.mem.span(extension), name)) {
            return true;
        }
    }
    return false;
}

fn findExtension(
    properties: []const vk.ExtensionProperties,
    name: []const u8,
) bool {
    for (properties) |property| {
        const found = std.mem.sliceTo(&property.extension_name, 0);
        if (std.mem.eql(u8, found, name)) return true;
    }
    return false;
}

fn limitsFromProperties(limits: vk.PhysicalDeviceLimits) Device.Limits {
    return .{
        .max_texture_extent_1d = limits.max_image_dimension_1d,
        .max_texture_extent_2d = limits.max_image_dimension_2d,
        .max_texture_extent_3d = limits.max_image_dimension_3d,
        .max_texture_layers = limits.max_image_array_layers,
        .max_texture_mips = @intCast(std.math.log2_int(
            u32,
            @max(limits.max_image_dimension_2d, 1),
        ) + 1),
        .max_vertex_stride = limits.max_vertex_input_binding_stride,
        .max_vertex_buffers = limits.max_vertex_input_bindings,
        .max_descriptor_sets = limits.max_bound_descriptor_sets,
        .max_push_constant_size = limits.max_push_constants_size,
        .max_storage_buffers = limits.max_per_stage_descriptor_storage_buffers,
        .max_sampled_textures = limits.max_per_stage_descriptor_sampled_images,
        .max_workgroup_size = limits.max_compute_work_group_size,
        .max_workgroup_memory = limits.max_compute_shared_memory_size,
    };
}

fn vkVersionMajor(version: u32) u32 {
    return (version >> 22) & 0x7f;
}

fn vkVersionMinor(version: u32) u32 {
    return (version >> 12) & 0x3ff;
}

fn vkVersionPatch(version: u32) u32 {
    return version & 0xfff;
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

pub const queue_vtable = Queue.VTable{
    .submit = queueSubmit,
    .present = queuePresent,
    .wait = queueWait,
};

/// Casts a vtable pointer back to a backend.
///
/// # Args
/// - `vptr`: opaque pointer stored in a device.
///
/// # Return
/// The owning backend.
pub fn backendFrom(vptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(vptr));
}

fn deviceCreate(
    vptr: *anyopaque,
    desc: *const Resource.Desc,
    ret_addr: usize,
) Device.CreateError!Resource {
    return resource.create(backendFrom(vptr), desc, ret_addr);
}

fn deviceDestroy(
    vptr: *anyopaque,
    handle: Resource,
    ret_addr: usize,
) void {
    resource.destroy(backendFrom(vptr), handle, ret_addr);
}

fn deviceMap(
    vptr: *anyopaque,
    handle: Resource,
    offset: u64,
    size: u64,
    flags: Device.MapFlags,
    ret_addr: usize,
) Device.MapError![]u8 {
    return resource.map(
        backendFrom(vptr),
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
    resource.unmap(backendFrom(vptr), handle, dirty, ret_addr);
}

fn deviceRecord(
    vptr: *anyopaque,
    handle: Resource,
    cmd: *const Device.Command,
    ret_addr: usize,
) Device.RecordError!void {
    return command.record(backendFrom(vptr), handle, cmd, ret_addr);
}

fn deviceQuery(
    vptr: *anyopaque,
    feature: Device.Feature,
    feature_index: u32,
    handle: ?Resource,
    out: ?[]u8,
    ret_addr: usize,
) Device.QueryError!Device.Support {
    return device_op.query(
        backendFrom(vptr),
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
    return device_op.getQueue(backendFrom(vptr), index, ret_addr);
}

fn deviceOp(
    vptr: *anyopaque,
    op: Device.DeviceOp,
    out: ?[]u8,
    ret_addr: usize,
) Device.DeviceOpError!void {
    return device_op.deviceOp(backendFrom(vptr), op, out, ret_addr);
}

fn queueFrom(vptr: *anyopaque) *QueueState {
    return @ptrCast(@alignCast(vptr));
}

fn queueSubmit(
    vptr: *anyopaque,
    submits: []const Queue.SubmitInfo,
    ret_addr: usize,
) Queue.SubmitError!void {
    const q = queueFrom(vptr);
    return queue.submit(q.backend, q, submits, ret_addr);
}

fn queuePresent(
    vptr: *anyopaque,
    swapchain: Resource,
    image_index: u32,
    wait_ops: []const Queue.SyncOp,
    ret_addr: usize,
) Queue.PresentError!Queue.PresentResult {
    const q = queueFrom(vptr);
    return queue.present(
        q.backend,
        q,
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
    const q = queueFrom(vptr);
    return queue.wait(
        q.backend,
        q,
        syncs,
        wait_any,
        timeout_ns,
        ret_addr,
    );
}

test "vtable entry points are analysed" {
    var backend: Backend = undefined;
    _ = backend.device();
    _ = &device_vtable;
    _ = &queue_vtable;
}

const testing = std.testing;

fn testBackend() !*Backend {
    return init(testing.allocator, .{ .verbose = false }) catch |err| switch (err) {
        error.LoaderUnavailable,
        error.NoPhysicalDevice,
        error.NoQueueFamily,
        error.ApiUnsupported,
        => return error.SkipZigTest,
        else => return err,
    };
}

test "vulkan backend creates and submits work" {
    const backend = try testBackend();
    const status = backend.deinit();
    try testing.expect(status.ok());
}

test "vulkan backend buffers, fences and command submission" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();

    var family: Device.QueueFamily = undefined;
    const support = try device.rawQuery(
        .queue_family,
        0,
        null,
        std.mem.asBytes(&family),
        @returnAddress(),
    );
    try testing.expectEqual(Device.Support.supported, support);
    try testing.expect(family.count > 0);

    var limits: Device.Limits = undefined;
    _ = try device.rawQuery(
        .limits,
        0,
        null,
        std.mem.asBytes(&limits),
        @returnAddress(),
    );
    try testing.expect(limits.max_texture_extent_2d > 0);

    const buffer = try rhi.Resource.Buffer.create(device, .{
        .size = 256,
        .usage = .{ .transfer_dst = true },
        .memory = .host,
    });
    defer buffer.destroy();

    const mapped = try buffer.map(0, 256, .{ .write = true });
    try testing.expectEqual(@as(usize, 256), mapped.len);
    @memset(mapped, 0);
    buffer.unmap(&.{.{ .offset = 0, .size = 256 }});

    const rhi_queue = try device.rawGetQueue(0, @returnAddress());

    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
        .reset_individual = true,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const begin_command = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin_command);
    const fill_command = Device.Command{ .fill_buffer = .{
        .buffer = buffer,
        .value = 0x1234_5678,
    } };
    try command_buffer.record(&fill_command);
    const end_command = Device.Command{ .end = {} };
    try command_buffer.record(&end_command);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();

    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());

    const result = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );
    try testing.expectEqual(@as(usize, 0), result.index);

    try command_buffer.reset();
    try pool.reset();
}

test "vulkan backend textures and capability queries" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();

    const desc = rhi.Resource.Texture.Desc{
        .kind = .d2,
        .extent = .{ .width = 8, .height = 8 },
        .format = .r8g8b8a8_unorm,
        .usage = .{ .sampled = true, .transfer_dst = true },
    };

    const support = try device.rawQuery(
        .{ .texture = desc },
        0,
        null,
        null,
        @returnAddress(),
    );
    try testing.expectEqual(Device.Support.supported, support);

    const texture = try rhi.Resource.Texture.create(device, desc);
    defer texture.destroy();

    const sampler = try rhi.Resource.Sampler.create(device, .{});
    defer sampler.destroy();

    const layout = try rhi.Resource.DescriptorSetLayout.create(device, .{
        .bindings = &.{.{
            .binding = 0,
            .kind = .combined_sampler,
            .stages = .{ .fragment = true },
        }},
    });
    defer layout.destroy();

    const set = try rhi.Resource.DescriptorSet.create(device, .{
        .layout = layout,
    });
    defer set.destroy();

    try set.update(&.{.{ .combined = .{
        .binding = 0,
        .texture = texture,
        .sampler = sampler,
    } }});

    if (backend.timeline_supported) {
        const semaphore = try rhi.Resource.Semaphore.create(device, .{
            .kind = .timeline,
        });
        defer semaphore.destroy();

        const fence = try rhi.Resource.Fence.create(device, .{});
        defer fence.destroy();

        const rhi_queue = try device.rawGetQueue(0, @returnAddress());
        const signals = [_]Queue.SyncOp{.{
            .timeline = .{ .semaphore = semaphore, .value = 1 },
        }};
        const submissions = [_]Queue.SubmitInfo{.{
            .commands = &.{},
            .signals = &signals,
            .fence = fence,
        }};
        try rhi_queue.submit(&submissions, @returnAddress());

        const wait_ops = [_]Queue.SyncOp{
            .{ .timeline = .{ .semaphore = semaphore, .value = 1 } },
            .{ .fence = fence },
        };
        const result = try rhi_queue.wait(
            &wait_ops,
            false,
            5 * std.time.ns_per_s,
            @returnAddress(),
        );
        try testing.expect(result.value >= 1);
    }
}

test "vulkan backend clears a texture and reads it back" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();
    const rhi_queue = try device.rawGetQueue(0, @returnAddress());

    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const texture = try rhi.Resource.Texture.create(device, .{
        .kind = .d2,
        .extent = .{ .width = 6, .height = 6 },
        .format = .r8g8b8a8_unorm,
        .usage = .{
            .color_attachment = true,
            .transfer_src = true,
        },
    });
    defer texture.destroy();

    const readback = try rhi.Resource.Buffer.create(device, .{
        .size = 6 * 6 * 4,
        .usage = .{ .transfer_dst = true },
        .memory = .host,
    });
    defer readback.destroy();

    const begin = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin);

    const render_pass = Device.Command{ .begin_render_pass = .{
        .area = .{ .width = 6, .height = 6 },
        .color = &.{.{
            .texture = texture,
            .load = .clear,
            .store = .store,
            .clear = .{ .float = .{ 1, 0, 0, 1 } },
        }},
    } };
    try command_buffer.record(&render_pass);

    const end_pass = Device.Command{ .end_render_pass = {} };
    try command_buffer.record(&end_pass);

    const copy = Device.Command{ .copy_texture_to_buffer = .{
        .src = texture,
        .dst = readback,
        .regions = &.{.{
            .region = .{ .extent = .{ 6, 6, 1 } },
        }},
    } };
    try command_buffer.record(&copy);

    const end = Device.Command{ .end = {} };
    try command_buffer.record(&end);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());
    _ = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );

    const data = try readback.map(0, 6 * 6 * 4, .{ .read = true });
    try testing.expectEqual(@as(u8, 255), data[0]);
    try testing.expectEqual(@as(u8, 0), data[1]);
    try testing.expectEqual(@as(u8, 0), data[2]);
    try testing.expectEqual(@as(u8, 255), data[3]);
    readback.unmap(&.{});
}

test "vulkan backend replays secondary command buffers" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();
    const rhi_queue = try device.rawGetQueue(0, @returnAddress());

    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const buffer = try rhi.Resource.Buffer.create(device, .{
        .size = 16,
        .usage = .{ .transfer_dst = true },
        .memory = .host,
    });
    defer buffer.destroy();

    const secondary = try rhi.Resource.CommandBuffer.create(device, .{
        .pool = pool,
        .level = .secondary,
    });
    defer secondary.destroy();

    const secondary_begin = Device.Command{ .begin = .{} };
    try secondary.record(&secondary_begin);
    const fill = Device.Command{ .fill_buffer = .{
        .buffer = buffer,
        .value = 0xfeed_face,
    } };
    try secondary.record(&fill);
    const secondary_end = Device.Command{ .end = {} };
    try secondary.record(&secondary_end);

    const primary = try rhi.Resource.CommandBuffer.create(device, .{
        .pool = pool,
    });
    defer primary.destroy();

    const primary_begin = Device.Command{ .begin = .{} };
    try primary.record(&primary_begin);
    const execute = Device.Command{ .execute_commands = .{
        .commands = &.{secondary},
    } };
    try primary.record(&execute);
    const primary_end = Device.Command{ .end = {} };
    try primary.record(&primary_end);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{primary},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());
    _ = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );

    const data = try buffer.map(0, 16, .{ .read = true });
    const written = std.mem.readInt(u32, data[0..4], .little);
    try testing.expectEqual(@as(u32, 0xfeed_face), written);
    buffer.unmap(&.{});
}

test "vulkan backend clears depth and stencil" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();
    const rhi_queue = try device.rawGetQueue(0, @returnAddress());

    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const texture = try rhi.Resource.Texture.create(device, .{
        .kind = .d2,
        .extent = .{ .width = 4, .height = 4 },
        .format = .depth24_plus_stencil8,
        .usage = .{
            .depth_stencil_attachment = true,
            .transfer_src = true,
        },
    });
    defer texture.destroy();

    const readback = try rhi.Resource.Buffer.create(device, .{
        .size = 4 * 4 * 4,
        .usage = .{ .transfer_dst = true },
        .memory = .host,
    });
    defer readback.destroy();

    const begin = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin);

    const render_pass = Device.Command{ .begin_render_pass = .{
        .area = .{ .width = 4, .height = 4 },
        .depth_stencil = .{
            .texture = texture,
            .load = .clear,
            .store = .store,
            .depth_clear = 0.5,
            .stencil_clear = 3,
        },
    } };
    try command_buffer.record(&render_pass);

    const end_pass = Device.Command{ .end_render_pass = {} };
    try command_buffer.record(&end_pass);

    const copy = Device.Command{ .copy_texture_to_buffer = .{
        .src = texture,
        .dst = readback,
        .regions = &.{.{
            .region = .{ .extent = .{ 4, 4, 1 } },
        }},
    } };
    try command_buffer.record(&copy);

    const end = Device.Command{ .end = {} };
    try command_buffer.record(&end);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());
    _ = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );

    const data = try readback.map(0, 4 * 4 * 4, .{ .read = true });
    const raw = std.mem.readInt(u32, data[0..4], .little);
    const depth = @as(f32, @floatFromInt(raw & 0x00ff_ffff)) /
        16_777_215.0;
    try testing.expect(depth >= 0.49 and depth <= 0.51);
    readback.unmap(&.{});
}

test "vulkan backend creates and dispatches a compute pipeline" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();

    // Minimal SPIR-V 1.0 compute shader with an empty body.
    const spirv = [_]u32{
        0x07230203, 0x00010000, 0x00290881, 0x00000009, 0x00000000,
        0x00020011, 0x00000001, 0x0003000e, 0x00000000, 0x00000001,
        0x0005000f, 0x00000005, 0x00000001, 0x6e69616d, 0x00000000,
        0x00060010, 0x00000001, 0x00000011, 0x00000001, 0x00000001,
        0x00000001, 0x00040004, 0x5f67697a, 0x6f727265, 0x003a7372,
        0x00030003, 0x0000000c, 0x00000881, 0x00040005, 0x00000002,
        0x64696f76, 0x00000000, 0x00030005, 0x00000006, 0x00323375,
        0x00050005, 0x00000001, 0x64616873, 0x6d2e7265, 0x006e6961,
        0x00020013, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
        0x00000000, 0x00030001, 0x00000006, 0x00000007, 0x00030021,
        0x00000008, 0x00000002, 0x00050036, 0x00000002, 0x00000001,
        0x00000000, 0x00000008, 0x000200f8, 0x00000004, 0x000100fd,
        0x000200f8, 0x00000005, 0x000100ff, 0x00010038,
    };

    const shader = try rhi.Resource.Shader.create(device, .{
        .code = std.mem.sliceAsBytes(&spirv),
        .stage = .compute,
        .entry = "main",
    });
    defer shader.destroy();

    const layout = try rhi.Resource.PipelineLayout.create(device, .{});
    defer layout.destroy();

    const pipeline = try rhi.Resource.PipelineCompute.create(device, .{
        .layout = layout,
        .shader = shader,
    });
    defer pipeline.destroy();

    const rhi_queue = try device.rawGetQueue(0, @returnAddress());
    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const begin = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin);

    const bind = Device.Command{ .bind_pipeline = .{
        .compute = pipeline,
    } };
    try command_buffer.record(&bind);

    const dispatch = Device.Command{ .dispatch = .{
        .groups = .{ 1, 1, 1 },
    } };
    try command_buffer.record(&dispatch);

    const end = Device.Command{ .end = {} };
    try command_buffer.record(&end);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());
    _ = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );
}

test "vulkan backend creates a graphics pipeline and draws" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();

    // Minimal vertex and fragment entry points with empty bodies.
    const spirv = [_]u32{
        0x07230203, 0x00010000, 0x00290881, 0x00000011, 0x00000000,
        0x00020011, 0x00000001, 0x0003000e, 0x00000000, 0x00000001,
        0x0005000f, 0x00000004, 0x00000001, 0x67617266, 0x00000000,
        0x0005000f, 0x00000000, 0x00000002, 0x74726576, 0x00000000,
        0x00030010, 0x00000001, 0x00000007, 0x00040004, 0x5f67697a,
        0x6f727265, 0x003a7372, 0x00030003, 0x0000000c, 0x00000881,
        0x00040005, 0x00000003, 0x64696f76, 0x00000000, 0x00030005,
        0x00000007, 0x00323375, 0x00050005, 0x00000001, 0x2e786667,
        0x67617266, 0x00000000, 0x00050005, 0x00000002, 0x2e786667,
        0x74726576, 0x00000000, 0x00020013, 0x00000003, 0x00040015,
        0x00000007, 0x00000020, 0x00000000, 0x00030001, 0x00000007,
        0x00000008, 0x00030021, 0x00000009, 0x00000003, 0x00050036,
        0x00000003, 0x00000001, 0x00000000, 0x00000009, 0x000200f8,
        0x00000005, 0x000100fd, 0x000200f8, 0x00000006, 0x000100ff,
        0x00010038, 0x00050036, 0x00000003, 0x00000002, 0x00000000,
        0x00000009, 0x000200f8, 0x0000000c, 0x000100fd, 0x000200f8,
        0x0000000d, 0x000100ff, 0x00010038,
    };

    const code = std.mem.sliceAsBytes(&spirv);

    const vertex = try rhi.Resource.Shader.create(device, .{
        .code = code,
        .stage = .vertex,
        .entry = "vert",
    });
    defer vertex.destroy();

    const fragment = try rhi.Resource.Shader.create(device, .{
        .code = code,
        .stage = .fragment,
        .entry = "frag",
    });
    defer fragment.destroy();

    const layout = try rhi.Resource.PipelineLayout.create(device, .{});
    defer layout.destroy();

    const pipeline = try rhi.Resource.PipelineGraphics.create(device, .{
        .layout = layout,
        .vertex = vertex,
        .fragment = fragment,
        .color = .{ .formats = &.{.r8g8b8a8_unorm} },
    });
    defer pipeline.destroy();

    const texture = try rhi.Resource.Texture.create(device, .{
        .kind = .d2,
        .extent = .{ .width = 4, .height = 4 },
        .format = .r8g8b8a8_unorm,
        .usage = .{ .color_attachment = true },
    });
    defer texture.destroy();

    const rhi_queue = try device.rawGetQueue(0, @returnAddress());
    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const begin = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin);

    const pass = Device.Command{ .begin_render_pass = .{
        .area = .{ .width = 4, .height = 4 },
        .color = &.{.{
            .texture = texture,
            .load = .clear,
            .clear = .{ .float = .{ 0, 0, 0, 1 } },
        }},
    } };
    try command_buffer.record(&pass);

    const bind = Device.Command{ .bind_pipeline = .{
        .graphics = pipeline,
    } };
    try command_buffer.record(&bind);

    const viewport = Device.Command{ .set_viewport = .{
        .width = 4,
        .height = 4,
    } };
    try command_buffer.record(&viewport);

    const scissor = Device.Command{ .set_scissor = .{
        .width = 4,
        .height = 4,
    } };
    try command_buffer.record(&scissor);

    const draw = Device.Command{ .draw = .{
        .vertices = .{ .count = 3 },
    } };
    try command_buffer.record(&draw);

    const end_pass = Device.Command{ .end_render_pass = {} };
    try command_buffer.record(&end_pass);

    const end = Device.Command{ .end = {} };
    try command_buffer.record(&end);

    const fence = try rhi.Resource.Fence.create(device, .{});
    defer fence.destroy();
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .fence = fence,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());
    _ = try rhi_queue.wait(
        &.{.{ .fence = fence }},
        false,
        5 * std.time.ns_per_s,
        @returnAddress(),
    );
}

test "vulkan backend presents through a headless swapchain" {
    const backend = try testBackend();
    defer {
        const status = backend.deinit();
        testing.expect(status.ok()) catch @panic("vulkan backend shutdown failed");
    }

    const device = backend.device();

    const surface = rhi.Resource.Surface.create(device, .{
        .window = 0,
        .platform = .headless,
    }) catch |err| switch (err) {
        error.Unsupport => return error.SkipZigTest,
        else => return err,
    };
    defer surface.destroy();

    const support = try device.rawQuery(
        .{ .swapchain = .{
            .format = .r8g8b8a8_unorm,
            .mode = .fifo,
        } },
        0,
        surface.handle,
        null,
        @returnAddress(),
    );
    if (support != .supported) return error.SkipZigTest;

    const swapchain = rhi.Resource.Swapchain.create(device, .{
        .surface = surface,
        .format = .r8g8b8a8_unorm,
        .extent = .{ .width = 16, .height = 16 },
        .mode = .fifo,
        .images = 2,
        .usage = .{ .color_attachment = true },
    }) catch |err| switch (err) {
        error.Unsupport, error.InvalidDesc => return error.SkipZigTest,
        else => return err,
    };
    defer swapchain.destroy();

    const rhi_queue = try device.rawGetQueue(0, @returnAddress());
    const pool = try rhi.Resource.CommandPool.create(device, .{
        .queue_family = rhi_queue.family_index,
    });
    defer pool.destroy();

    const command_buffer = try rhi.Resource.CommandBuffer.create(
        device,
        .{ .pool = pool },
    );
    defer command_buffer.destroy();

    const image_ready = try rhi.Resource.Semaphore.create(device, .{});
    defer image_ready.destroy();
    const render_done = try rhi.Resource.Semaphore.create(device, .{});
    defer render_done.destroy();

    const image_index = try swapchain.acquire(image_ready, null, 5 * std.time.ns_per_s);
    const image = try swapchain.image(image_index);

    const begin = Device.Command{ .begin = .{} };
    try command_buffer.record(&begin);

    const pass = Device.Command{ .begin_render_pass = .{
        .area = .{ .width = 1, .height = 1 },
        .color = &.{.{
            .texture = image,
            .load = .clear,
            .store = .store,
            .clear = .{ .float = .{ 0, 1, 0, 1 } },
        }},
    } };
    try command_buffer.record(&pass);

    const end_pass = Device.Command{ .end_render_pass = {} };
    try command_buffer.record(&end_pass);

    const present_barrier = Device.Command{ .barrier = .{
        .textures = &.{.{
            .texture = image,
            .src = .{
                .stages = .{ .color_attachment = true },
                .access = .{ .color_write = true },
            },
            .dst = .{ .stages = .{ .all = true } },
            .layout = .present,
        }},
    } };
    try command_buffer.record(&present_barrier);

    const end = Device.Command{ .end = {} };
    try command_buffer.record(&end);

    const signals = [_]Queue.SyncOp{.{ .semaphore = render_done }};
    const waits = [_]Queue.SyncOp{.{ .semaphore = image_ready }};
    const submissions = [_]Queue.SubmitInfo{.{
        .commands = &.{command_buffer},
        .waits = &waits,
        .signals = &signals,
    }};
    try rhi_queue.submit(&submissions, @returnAddress());

    _ = try rhi_queue.present(
        swapchain,
        image_index,
        &.{.{ .semaphore = render_done }},
        @returnAddress(),
    );

    try device.waitIdle(@returnAddress());
}
