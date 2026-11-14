const std = @import("std");

pub const Error = error{
    /// The operation, or a feature it requires, is not
    /// supported by this backend, device, or object.
    Unsupport,

    /// The device has been lost.
    ///
    /// All further use of the device and its resources
    /// is undefined; the application must destroy and
    /// recreate it.
    DeviceLost,

    /// The device failed for a reason other than
    /// being lost.
    DeviceFailure,

    /// An argument failed validation.
    ///
    /// The exact failure is intentionally not exposed:
    /// the RHI validation layer reports the details,
    /// and backends must not invent additional argument
    /// errors.
    InvalidArgument,

    /// The object is in a state that does not permit
    /// this operation, or the operation would violate
    /// the ordering rules of the interface.
    InvalidState,

    /// A backend violated the RHI contract
    ///
    /// For example by returning a handle whose tag does
    /// not match the supplied description.
    BackendFailure,
} || std.mem.Allocator.Error;

pub const Create = error{
    /// The resource description is invalid.
    InvalidDesc,

    /// Device memory is exhausted.
    ///
    /// This is distinct from `OutOfMemory`, which is about
    /// host memory.
    OutOfDeviceMemory,

    /// The backend cannot track another live resource of
    /// this kind.
    TooManyObjects,
} || Error;

pub const Map = error{
    /// The requested mapping is invalid.
    InvalidDesc,

    /// The handle does not refer to a live resource.
    InvalidResource,

    /// The resource was not created with `host_visible` memory.
    NotHostVisible,

    /// The resource is already mapped.
    AlreadyMapped,

    /// The requested range lies outside of the resource.
    OutOfRange,
} || Error;

pub const Record = error{
    /// The description of the command is invalid.
    InvalidDesc,

    /// A resource referenced by the command is not live or has
    /// the wrong kind.
    InvalidResource,

    /// The command list is not currently recording.
    CommandBufferState,
} || Error;

pub const Query = error{
    /// `feature_index` is outside of the range accepted by the
    /// feature.
    IndexOutOfBounds,

    /// `out_len` is too small for the feature's output
    /// structure.
    InsufficientBuffer,
} || Error;

pub const GetQueue = error{
    /// The device exposes no queues
    /// (a headless compute-only device).
    NoQueues,

    /// `index` is outside of the flat queue list.
    IndexOutOfBounds,
} || Error;

pub const Op = error{
    /// The operation description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the wrong kind.
    InvalidResource,

    /// An index into a backend-side list is out of range.
    IndexOutOfBounds,

    /// An output buffer is too small, or an output was
    /// required and omitted.
    InsufficientBuffer,

    /// The device has no queue with the requested index.
    NoQueues,

    /// A command list is not in the state required by the
    /// operation.
    CommandBufferState,

    /// A timed operation expired before it could complete.
    Timeout,

    /// The current swapchain can no longer be used.
    SwapchainOutOfDate,

    /// The surface backing a swapchain is no longer available.
    SurfaceLost,
} || Error;

pub const Submit = error{
    /// A submit description is malformed.
    InvalidDesc,

    /// A referenced resource is not live or has the wrong kind.
    InvalidResource,

    /// The queue is not able to accept submissions
    /// (for example because it was destroyed with its device).
    QueueState,
} || Error;

pub const Present = error{
    /// The swapchain handle is not live or is not a swapchain.
    InvalidResource,

    /// The surface backing the swapchain has been lost.
    SurfaceLost,

    /// The surface's presentation engine does not support the
    /// request.
    SurfaceNotSupported,
} || Error;

pub const Wait = error{
    /// A referenced synchronisation object is not live or has
    /// the wrong kind.
    InvalidResource,
} || Error;
