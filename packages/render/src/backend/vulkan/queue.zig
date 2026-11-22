//! Queue submission, presentation and host waits.

const std = @import("std");
const rhi = @import("rhi");
const vk = @import("vk");

const root = @import("root.zig");

const Backend = root.Backend;
const Queue = rhi.Queue;
const QueueState = root.QueueState;
const Resource = rhi.Resource;

/// Submits recorded work to a queue.
///
/// # Args
/// - `self`: owning backend.
/// - `state`: queue receiving the work.
/// - `submits`: submission records, executed in order.
/// - `ret_addr`: address of the public call site.
///
/// # Error
/// - `error.InvalidDesc`: a submission is malformed.
/// - `error.InvalidResource`: a referenced resource is
///   not live.
/// - `error.QueueState`: the queue cannot accept
///   submissions.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub fn submit(
    self: *Backend,
    state: *QueueState,
    submits: []const Queue.SubmitInfo,
    ret_addr: usize,
) Queue.SubmitError!void {
    _ = ret_addr;

    const family = self.families[state.family_index];

    for (submits) |submission| {
        for (submission.commands) |command_buffer| {
            const entry = self.lookup(command_buffer) orelse
                return error.InvalidResource;
            if (entry.kind != .command_buffer) {
                return error.InvalidResource;
            }
            const buffer = &entry.data.command_buffer;
            if (buffer.level != .primary) return error.InvalidDesc;
            if (buffer.phase != .ended) return error.InvalidDesc;

            if (buffer.kinds.graphics and !family.kinds.graphics) {
                return error.InvalidDesc;
            }
            if (buffer.kinds.compute and !family.kinds.compute) {
                return error.InvalidDesc;
            }
            if (buffer.kinds.transfer and !family.kinds.transfer) {
                return error.InvalidDesc;
            }
        }

        for (submission.waits) |sync| {
            _ = try syncSemaphore(self, sync);
        }
        for (submission.signals) |sync| {
            _ = try syncSemaphore(self, sync);
        }
        if (submission.fence) |handle| {
            _ = self.get(handle, .fence) orelse
                return error.InvalidResource;
        }
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    for (submits) |submission| {
        const native_commands = try scratch.alloc(
            vk.CommandBuffer,
            submission.commands.len,
        );
        for (submission.commands, 0..) |handle, i| {
            const entry = self.lookup(handle) orelse
                return error.InvalidResource;
            native_commands[i] = entry.data.command_buffer.native;
        }

        const native_waits = try scratch.alloc(
            vk.Semaphore,
            submission.waits.len,
        );
        const wait_stages = try scratch.alloc(
            vk.PipelineStageFlags,
            submission.waits.len,
        );
        const wait_values = try scratch.alloc(
            u64,
            submission.waits.len,
        );
        var has_timeline = false;
        for (submission.waits, 0..) |sync, i| {
            const native = try syncSemaphore(self, sync);
            native_waits[i] = native.semaphore;
            wait_stages[i] = .{ .all_commands = true };
            wait_values[i] = native.value;
            has_timeline = has_timeline or native.timeline;
        }

        const native_signals = try scratch.alloc(
            vk.Semaphore,
            submission.signals.len,
        );
        const signal_values = try scratch.alloc(
            u64,
            submission.signals.len,
        );
        for (submission.signals, 0..) |sync, i| {
            const native = try syncSemaphore(self, sync);
            if (native.timeline and native.value == 0) {
                return error.InvalidDesc;
            }
            native_signals[i] = native.semaphore;
            signal_values[i] = native.value;
            has_timeline = has_timeline or native.timeline;
        }

        const timeline_info = vk.TimelineSemaphoreSubmitInfo{
            .wait_semaphore_value_count = @intCast(wait_values.len),
            .p_wait_semaphore_values = if (wait_values.len == 0)
                null
            else
                wait_values.ptr,
            .signal_semaphore_value_count = @intCast(signal_values.len),
            .p_signal_semaphore_values = if (signal_values.len == 0)
                null
            else
                signal_values.ptr,
        };

        const info = vk.SubmitInfo{
            .p_next = if (has_timeline)
                @ptrCast(&timeline_info)
            else
                null,
            .wait_semaphore_count = @intCast(native_waits.len),
            .p_wait_semaphores = if (native_waits.len == 0)
                null
            else
                native_waits.ptr,
            .p_wait_dst_stage_mask = if (wait_stages.len == 0)
                null
            else
                wait_stages.ptr,
            .command_buffer_count = @intCast(native_commands.len),
            .p_command_buffers = if (native_commands.len == 0)
                null
            else
                native_commands.ptr,
            .signal_semaphore_count = @intCast(native_signals.len),
            .p_signal_semaphores = if (native_signals.len == 0)
                null
            else
                native_signals.ptr,
        };

        const fence: vk.Fence = if (submission.fence) |handle|
            (self.get(handle, .fence) orelse
                return error.InvalidResource).fence
        else
            .null;

        self.device_api.queueSubmit(
            state.queue,
            &.{info},
            fence,
        ) catch |err| return submitError(err);
    }
}

/// Presents a swapchain image to its surface.
///
/// # Args
/// - `self`: owning backend.
/// - `state`: queue used for presentation.
/// - `swapchain`: swapchain owning the image.
/// - `image_index`: index of the acquired image.
/// - `wait_ops`: semaphores to wait for.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// Whether the swapchain still matches the surface.
///
/// # Error
/// - `error.InvalidDesc`: the image index or wait
///   list is invalid.
/// - `error.InvalidResource`: `swapchain` is not live.
/// - `error.SwapchainOutOfDate`: the swapchain is
///   stale.
/// - `error.SurfaceLost`: the surface is gone.
/// - `error.SurfaceNotSupported`: the queue cannot
///   present to the surface.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub fn present(
    self: *Backend,
    state: *QueueState,
    swapchain: Resource,
    image_index: u32,
    wait_ops: []const Queue.SyncOp,
    ret_addr: usize,
) Queue.PresentError!Queue.PresentResult {
    _ = ret_addr;

    const entry = self.lookup(swapchain) orelse
        return error.InvalidResource;
    if (entry.kind != .swapchain) return error.InvalidResource;
    const chain = &entry.data.swapchain;

    const surface_entry = self.lookup(chain.surface) orelse
        return error.InvalidResource;
    if (surface_entry.kind != .surface) {
        return error.InvalidResource;
    }
    const surface = surface_entry.data.surface.surface;

    if (!self.familySupportsPresent(surface, state.family_index)) {
        return error.SurfaceNotSupported;
    }
    if (@as(usize, image_index) >= chain.images.len) {
        return error.InvalidDesc;
    }
    if (!chain.acquired[image_index]) {
        return error.InvalidDesc;
    }

    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const waits = try scratch.alloc(vk.Semaphore, wait_ops.len);
    for (wait_ops, 0..) |sync, i| {
        switch (sync) {
            .fence => return error.InvalidDesc,
            .timeline => return error.InvalidDesc,
            .semaphore => |handle| {
                const semaphore = self.get(handle, .semaphore) orelse
                    return error.InvalidResource;
                if (semaphore.kind != .binary) {
                    return error.InvalidDesc;
                }
                waits[i] = semaphore.semaphore;
            },
        }
    }

    const swapchains = [_]vk.SwapchainKHR{chain.swapchain};
    const indices = [_]u32{image_index};
    const info = vk.PresentInfoKHR{
        .wait_semaphore_count = @intCast(waits.len),
        .p_wait_semaphores = if (waits.len == 0)
            null
        else
            waits.ptr,
        .swapchain_count = 1,
        .p_swapchains = &swapchains,
        .p_image_indices = &indices,
    };

    const result = self.device_api.queuePresentKHR(
        state.queue,
        &info,
    ) catch |err| return switch (err) {
        error.OutOfDateKHR => error.SwapchainOutOfDate,
        error.SurfaceLostKHR => error.SurfaceLost,
        error.DeviceLost => error.DeviceLost,
        error.OutOfHostMemory => error.OutOfMemory,
        else => error.DeviceFailure,
    };

    chain.acquired[image_index] = false;

    if (self.get(chain.images[image_index], .texture)) |texture| {
        texture.layout = .present_src_khr;
    }

    return switch (result) {
        .suboptimal_khr => .suboptimal,
        else => .success,
    };
}

/// Blocks the host until sync points are signalled.
///
/// # Args
/// - `self`: owning backend.
/// - `state`: queue owning the wait.
/// - `syncs`: sync points to wait for.
/// - `wait_any`: wait for one point instead of all.
/// - `timeout_ns`: nanosecond timeout; null waits
///   forever.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The sync point that ended the wait.
///
/// # Error
/// - `error.InvalidResource`: a sync point is not
///   live.
/// - `error.Unsupport`: a binary semaphore is waited
///   on, or fences and timelines are mixed.
/// - `error.Timeout`: `timeout_ns` elapsed first.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
pub fn wait(
    self: *Backend,
    state: *QueueState,
    syncs: []const Queue.SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
    ret_addr: usize,
) Queue.WaitError!Queue.WaitResult {
    _ = state;
    _ = ret_addr;

    if (syncs.len == 0) return .{};

    // Validate every sync point before waiting on any.
    for (syncs) |sync| {
        switch (sync) {
            .fence => |handle| {
                _ = self.get(handle, .fence) orelse
                    return error.InvalidResource;
            },
            .timeline => |timeline| {
                const semaphore = self.get(
                    timeline.semaphore,
                    .semaphore,
                ) orelse return error.InvalidResource;
                if (semaphore.kind != .timeline) {
                    return error.Unsupport;
                }
            },
            .semaphore => |handle| {
                _ = self.get(handle, .semaphore) orelse
                    return error.InvalidResource;
                return error.Unsupport;
            },
        }
    }

    var has_fence = false;
    var has_timeline = false;
    for (syncs) |sync| {
        switch (sync) {
            .fence => has_fence = true,
            .timeline => has_timeline = true,
            .semaphore => unreachable,
        }
    }

    if (has_fence and !has_timeline) {
        return waitFences(self, syncs, wait_any, timeout_ns);
    }
    if (has_timeline and !has_fence) {
        return waitTimelines(self, syncs, wait_any, timeout_ns);
    }
    return waitMixed(self, syncs, wait_any, timeout_ns);
}

fn waitFences(
    self: *Backend,
    syncs: []const Queue.SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
) Queue.WaitError!Queue.WaitResult {
    const fences = try self.allocator.alloc(vk.Fence, syncs.len);
    defer self.allocator.free(fences);
    for (syncs, 0..) |sync, i| {
        const handle = switch (sync) {
            .fence => |fence| fence,
            else => return error.Unsupport,
        };
        fences[i] = (self.get(handle, .fence) orelse
            return error.InvalidResource).fence;
    }

    const timeout = timeout_ns orelse std.math.maxInt(u64);
    const result = self.device_api.waitForFences(
        self.vk_device,
        fences,
        vk.Bool32.fromBool(!wait_any),
        timeout,
    ) catch |err| return waitError(err);

    if (result == .timeout) return error.Timeout;

    var out = Queue.WaitResult{};
    if (wait_any) {
        for (fences, 0..) |fence, i| {
            const status = self.device_api.getFenceStatus(
                self.vk_device,
                fence,
            ) catch |err| return waitError(err);
            if (status == .success) {
                out.index = i;
                break;
            }
        }
    }
    return out;
}

fn waitTimelines(
    self: *Backend,
    syncs: []const Queue.SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
) Queue.WaitError!Queue.WaitResult {
    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const semaphores = try scratch.alloc(vk.Semaphore, syncs.len);
    const values = try scratch.alloc(u64, syncs.len);
    for (syncs, 0..) |sync, i| {
        const timeline = sync.timeline;
        const semaphore = self.get(timeline.semaphore, .semaphore) orelse
            return error.InvalidResource;
        semaphores[i] = semaphore.semaphore;
        values[i] = timeline.value;
    }

    const info = vk.SemaphoreWaitInfo{
        .flags = .{ .any = wait_any },
        .semaphore_count = @intCast(semaphores.len),
        .p_semaphores = semaphores.ptr,
        .p_values = values.ptr,
    };

    const timeout = timeout_ns orelse std.math.maxInt(u64);
    const result = self.device_api.waitSemaphores(
        self.vk_device,
        &info,
        timeout,
    ) catch |err| return waitError(err);
    if (result == .timeout) return error.Timeout;

    var out = Queue.WaitResult{};
    for (syncs, 0..) |sync, i| {
        if (!wait_any and i > 0) break;
        const timeline = sync.timeline;
        const semaphore = self.get(timeline.semaphore, .semaphore).?;
        const value = self.device_api.getSemaphoreCounterValue(
            self.vk_device,
            semaphore.semaphore,
        ) catch |err| return waitError(err);
        if (value >= timeline.value) {
            out.index = i;
            out.value = value;
            break;
        }
    }
    return out;
}

fn waitMixed(
    self: *Backend,
    syncs: []const Queue.SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
) Queue.WaitError!Queue.WaitResult {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const clock: std.Io.Clock = .awake;
    const start = std.Io.Clock.Timestamp.now(io, clock);
    const deadline: ?std.Io.Clock.Timestamp = if (timeout_ns) |timeout|
        std.Io.Clock.Timestamp.addDuration(start, .{
            .raw = std.Io.Duration.fromNanoseconds(@intCast(timeout)),
            .clock = clock,
        })
    else
        null;
    const step = std.Io.Clock.Duration{
        .raw = std.Io.Duration.fromMilliseconds(1),
        .clock = clock,
    };

    while (true) {
        var ready_count: usize = 0;
        var first_ready: ?usize = null;
        var first_value: u64 = 0;

        for (syncs, 0..) |sync, i| {
            const status = syncReady(self, sync) catch |err| return err;
            if (status.ready) {
                ready_count += 1;
                if (first_ready == null) {
                    first_ready = i;
                    first_value = status.value;
                }
            }
        }

        const satisfied = if (wait_any)
            ready_count > 0
        else
            ready_count == syncs.len;

        if (satisfied) {
            var out = Queue.WaitResult{};
            if (first_ready) |index| {
                if (wait_any) {
                    out.index = index;
                    out.value = first_value;
                } else {
                    out.value = first_value;
                }
            }
            return out;
        }

        if (deadline) |limit| {
            const now = std.Io.Clock.Timestamp.now(io, clock);
            if (now.raw.nanoseconds >= limit.raw.nanoseconds) {
                return error.Timeout;
            }
        }
        step.sleep(io) catch {};
    }
}

fn syncReady(
    self: *Backend,
    sync: Queue.SyncOp,
) Queue.WaitError!Ready {
    switch (sync) {
        .fence => |handle| {
            const fence = self.get(handle, .fence) orelse
                return error.InvalidResource;
            const result = self.device_api.getFenceStatus(
                self.vk_device,
                fence.fence,
            ) catch |err| return waitError(err);
            return .{ .ready = result == .success, .value = 0 };
        },
        .timeline => |timeline| {
            const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                return error.InvalidResource;
            const value = self.device_api.getSemaphoreCounterValue(
                self.vk_device,
                semaphore.semaphore,
            ) catch |err| return waitError(err);
            return .{
                .ready = value >= timeline.value,
                .value = value,
            };
        },
        .semaphore => return error.Unsupport,
    }
}

const Ready = struct {
    ready: bool,
    value: u64,
};

const NativeSync = struct {
    semaphore: vk.Semaphore,
    value: u64,
    timeline: bool,
};

fn syncSemaphore(
    self: *Backend,
    sync: Queue.SyncOp,
) Queue.SubmitError!NativeSync {
    switch (sync) {
        .fence => return error.InvalidDesc,
        .semaphore => |handle| {
            const semaphore = self.get(handle, .semaphore) orelse
                return error.InvalidResource;
            if (semaphore.kind != .binary) return error.InvalidDesc;
            return .{
                .semaphore = semaphore.semaphore,
                .value = 0,
                .timeline = false,
            };
        },
        .timeline => |timeline| {
            const semaphore = self.get(timeline.semaphore, .semaphore) orelse
                return error.InvalidResource;
            if (semaphore.kind != .timeline) return error.InvalidDesc;
            return .{
                .semaphore = semaphore.semaphore,
                .value = timeline.value,
                .timeline = true,
            };
        },
    }
}

fn submitError(err: anyerror) Queue.SubmitError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.DeviceLost => error.DeviceLost,
        else => error.DeviceFailure,
    };
}

fn waitError(err: anyerror) Queue.WaitError {
    return switch (err) {
        error.OutOfHostMemory, error.OutOfMemory => error.OutOfMemory,
        error.DeviceLost => error.DeviceLost,
        error.Timeout => error.Timeout,
        else => error.DeviceFailure,
    };
}
