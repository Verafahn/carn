//! Submission and synchronization on a device queue.
//!
//! A `Queue` is a value that identifies one queue inside
//! a family and carries the backend vtable needed to use
//! it. Obtain one with `Device.getQueue`.
//!
//! `submit` sends recorded command buffers to the GPU,
//! `present` hands a swapchain image to the presentation
//! engine, and `wait` blocks the host on synchronization
//! objects.
//!
//! The `raw*` forms take an explicit call-site address;
//! prefer `submit`, `present` and `wait`.

const Device = @import("Device.zig");
const Error = Device.Error;
const Resource = @import("Resource.zig");
const Queue = @This();

const CommandBuffer = Resource.CommandBuffer;
const Fence = Resource.Fence;
const Semaphore = Resource.Semaphore;
const Swapchain = Resource.Swapchain;

/// Backend state; interpreted only by `vtable`.
vptr: *anyopaque,
/// Backend implementation of the queue.
vtable: *const VTable,
/// The queue family this queue belongs to.
family_index: u32,
/// The index of this queue inside its family.
queue_index: u32,

/// The backend implementation of the queue.
///
/// # Note
/// Every entry receives the backend pointer stored in
/// `Queue.vptr`. Entries must not retain borrowed
/// slices after returning.
pub const VTable = struct {
    /// Submits recorded work to the queue.
    ///
    /// # Args
    /// - `submits`: submission records, executed in
    ///   order.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Error
    /// - `error.InvalidDesc`: a submission is
    ///   malformed.
    /// - `error.InvalidResource`: a referenced resource
    ///   is not live or has the wrong kind.
    /// - `error.QueueState`: the queue cannot accept
    ///   submissions.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    submit: *const fn (
        *anyopaque,
        submits: []const SubmitInfo,
        ret_addr: usize,
    ) SubmitError!void,

    /// Presents a swapchain image to its surface.
    ///
    /// # Args
    /// - `swapchain`: swapchain owning the image.
    /// - `image_index`: index of the acquired image.
    /// - `wait_ops`: sync points to wait for; must not
    ///   contain `.fence`.
    /// - `ret_addr`: address of the public call site.
    ///
    /// # Return
    /// Whether the swapchain still matches the
    /// surface.
    ///
    /// # Error
    /// - `error.InvalidDesc`: the image index or wait
    ///   list is invalid.
    /// - `error.InvalidResource`: `swapchain` is not
    ///   live or is not a swapchain.
    /// - `error.SwapchainOutOfDate`: the swapchain is
    ///   stale.
    /// - `error.SurfaceLost`: the surface is gone.
    /// - `error.SurfaceNotSupported`: the presentation
    ///   engine rejects the request.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    present: *const fn (
        *anyopaque,
        swapchain: Resource,
        image_index: u32,
        wait_ops: []const SyncOp,
        ret_addr: usize,
    ) PresentError!PresentResult,

    /// Waits for synchronization objects.
    ///
    /// # Args
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
    ///   live or has the wrong kind.
    /// - `error.Timeout`: `timeout_ns` elapsed first.
    /// - `error.Unsupport`: the backend cannot wait on
    ///   a given sync type.
    /// - `error.OutOfMemory`: host bookkeeping fails.
    /// - `error.DeviceLost`, `error.DeviceFailure`:
    ///   the device is unusable.
    wait: *const fn (
        *anyopaque,
        syncs: []const SyncOp,
        wait_any: bool,
        timeout_ns: ?u64,
        ret_addr: usize,
    ) WaitError!WaitResult,
};

/// One submission of recorded work.
pub const SubmitInfo = struct {
    /// Command buffers executed in order.
    ///
    /// Each buffer must be a primary buffer and must
    /// have been ended.
    commands: []const CommandBuffer,
    /// Semaphores waited on before execution.
    ///
    /// Must not contain `.fence`; use `Queue.wait` for
    /// host waits.
    waits: []const SyncOp = &.{},
    /// Semaphores signalled when execution ends.
    ///
    /// Must not contain `.fence`; use `fence` to
    /// signal host work.
    signals: []const SyncOp = &.{},
    /// Fence signalled when execution ends; null
    /// signals nothing.
    fence: ?Fence = null,
};

/// A synchronization point used by queue operations.
pub const SyncOp = union(enum) {
    /// A binary semaphore.
    ///
    /// Valid for device-side waits and signals. Host
    /// waits are backend-defined.
    semaphore: Semaphore,
    /// A timeline semaphore at a specific value.
    timeline: Timeline,
    /// A fence.
    ///
    /// Valid for host waits through `Queue.wait`.
    fence: Fence,

    /// A timeline semaphore and value pair.
    pub const Timeline = struct {
        /// Semaphore to wait on or signal.
        semaphore: Semaphore,
        /// Value that marks the operation.
        value: u64,
    };
};

/// Outcome of a successful presentation.
pub const PresentResult = enum {
    /// The image was presented and the swapchain is
    /// still usable.
    success,
    /// The image was presented but the swapchain no
    /// longer matches the surface.
    ///
    /// Recreate the swapchain before presenting again.
    suboptimal,
};

/// Outcome of a successful queue wait.
pub const WaitResult = struct {
    /// Sync point that ended the wait.
    ///
    /// Always 0 when `wait_any` is false.
    index: usize = 0,
    /// Timeline value observed at `index`.
    ///
    /// 0 for binary semaphores and fences.
    value: u64 = 0,
};

/// Errors that report a failed submission.
pub const SubmitError = error{
    /// A submit description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the
    /// wrong kind.
    InvalidResource,

    /// The queue is not able to accept submissions
    /// (for example because it was destroyed with its
    /// device).
    QueueState,
} || Error;

/// Errors that report a failed presentation.
pub const PresentError = error{
    /// The image index or wait list is invalid, for
    /// example because it contains `.fence`.
    InvalidDesc,

    /// The swapchain handle is not live or is not a
    /// swapchain.
    InvalidResource,

    /// The swapchain no longer matches the surface.
    SwapchainOutOfDate,

    /// The surface backing the swapchain has been
    /// lost.
    SurfaceLost,

    /// The surface's presentation engine does not
    /// support the request.
    SurfaceNotSupported,
} || Error;

/// Errors that report a failed wait.
pub const WaitError = error{
    /// A referenced synchronization object is not
    /// live or has the wrong kind.
    InvalidResource,

    /// `timeout_ns` elapsed before the wait was
    /// satisfied.
    Timeout,
} || Error;

/// Submits recorded work through the backend vtable.
///
/// # Args
/// - `submits`: ordered submission records.
/// - `ret_addr`: address of the public call site.
///
/// # Note
/// Prefer `submit`, which captures the call site.
/// The error and lifetime contract matches `submit`.
pub inline fn rawSubmit(
    self: Queue,
    submits: []const SubmitInfo,
    ret_addr: usize,
) SubmitError!void {
    try self.vtable.submit(self.vptr, submits, ret_addr);
}

/// Submits command buffers to the queue.
///
/// # Args
/// - `submits`: ordered submission records.
///
/// # Error
/// - `error.InvalidDesc`: a submission is malformed.
/// - `error.InvalidResource`: a referenced resource
///   is not live or has the wrong kind.
/// - `error.QueueState`: the queue cannot accept
///   submissions.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`:
///   the device is unusable.
///
/// # Note
/// The submitted command buffers and the resources
/// they reference must stay alive until the submission
/// completes. The slice passed as `submits` is borrowed
/// only for this call.
pub inline fn submit(
    self: Queue,
    submits: []const SubmitInfo,
) SubmitError!void {
    try self.rawSubmit(submits, @returnAddress());
}

/// Presents an image through the backend vtable.
///
/// # Args
/// - `swapchain`: swapchain owning the image.
/// - `image_index`: index of the acquired image.
/// - `wait_ops`: sync points to wait for.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// Whether the swapchain still matches the surface.
///
/// # Note
/// Prefer `present`, which captures the call site.
/// The error contract matches `present`.
pub inline fn rawPresent(
    self: Queue,
    swapchain: Swapchain,
    image_index: u32,
    wait_ops: []const SyncOp,
    ret_addr: usize,
) PresentError!PresentResult {
    return try self.vtable.present(
        self.vptr,
        swapchain.handle,
        image_index,
        wait_ops,
        ret_addr,
    );
}

/// Presents a swapchain image to the surface.
///
/// # Args
/// - `swapchain`: swapchain owning the image.
/// - `image_index`: index returned by
///   `Swapchain.acquire`.
/// - `wait_ops`: sync points to wait for before
///   presentation; must not contain `.fence`.
///
/// # Return
/// Whether the surface still matches the swapchain.
///
/// # Error
/// - `error.InvalidDesc`: the image index or wait
///   list is invalid.
/// - `error.InvalidResource`: `swapchain` is not
///   live or is not a swapchain.
/// - `error.SwapchainOutOfDate`: the swapchain is
///   stale.
/// - `error.SurfaceLost`: the surface is gone.
/// - `error.SurfaceNotSupported`: the presentation
///   engine rejects the request.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`: the
///   device is unusable.
///
/// # Note
/// `wait_ops` usually contains a semaphore that is
/// signalled when rendering into `image_index` ends.
pub inline fn present(
    self: Queue,
    swapchain: Swapchain,
    image_index: u32,
    wait_ops: []const SyncOp,
) PresentError!PresentResult {
    return try self.rawPresent(
        swapchain,
        image_index,
        wait_ops,
        @returnAddress(),
    );
}

/// Waits for sync points through the backend
/// vtable.
///
/// # Args
/// - `syncs`: sync points to wait for.
/// - `wait_any`: wait for one point instead of all.
/// - `timeout_ns`: nanosecond timeout; null waits
///   forever.
/// - `ret_addr`: address of the public call site.
///
/// # Return
/// The sync point that ended the wait.
///
/// # Note
/// Prefer `wait`, which captures the call site. The
/// error contract matches `wait`.
pub inline fn rawWait(
    self: Queue,
    syncs: []const SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
    ret_addr: usize,
) WaitError!WaitResult {
    return try self.vtable.wait(
        self.vptr,
        syncs,
        wait_any,
        timeout_ns,
        ret_addr,
    );
}

/// Blocks the host until sync points are signalled.
///
/// # Args
/// - `syncs`: sync points to wait for.
/// - `wait_any`: wait for one point instead of all.
/// - `timeout_ns`: nanosecond timeout; null waits
///   forever.
///
/// # Return
/// The sync point that ended the wait.
///
/// # Error
/// - `error.InvalidResource`: a sync point is not
///   live or has the wrong kind.
/// - `error.Timeout`: `timeout_ns` elapsed first.
/// - `error.Unsupport`: the backend cannot wait on a
///   given sync type.
/// - `error.OutOfMemory`: host bookkeeping fails.
/// - `error.DeviceLost`, `error.DeviceFailure`:
///   the device is unusable.
///
/// # Note
/// Timeline semaphores report their observed value in
/// `WaitResult.value`. Fences and binary semaphores
/// report 0.
pub inline fn wait(
    self: Queue,
    syncs: []const SyncOp,
    wait_any: bool,
    timeout_ns: ?u64,
) WaitError!WaitResult {
    return try self.rawWait(
        syncs,
        wait_any,
        timeout_ns,
        @returnAddress(),
    );
}
