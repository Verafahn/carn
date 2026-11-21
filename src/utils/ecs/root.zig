const std = @import("std");
const Io = std.Io;
const map = @import("Map.zig");
const sparse_set = @import("SparseSet.zig");
const archtype = @import("Archtype.zig");

pub const Map = map.Map;
pub const SparseSet = sparse_set.SparseSet;
pub const Archtype = archtype.Archtype;

/// Apply a set of systems to `world`.
///
/// - `world` must be a pointer to a specific ECS data implementation type.
/// - If `io_` is `null`, it will synchronously call the runner.
/// - `ctx` will be directly used as the second parameter of the runner.
///
/// This function will generate a reasonable scheduling order based on
/// `read`,`write`, `after`, and `before`, and make concurrent calls in this order.
///
/// `Runner` requires:
/// ```
/// const Runner = struct { // any container types.
///     // Optionals:
///     // Which components will be read.
///     pub const read = .{...};
///     // Which components will be write.
///     pub const write = .{...};
///     // Which runners must be run after.
///     pub const after = .{...};
///     // Which runners must be run before.
///     pub const before = .{...};
///
///     // Requries
///     pub fn run(world: *World, ctx: Context) void {
///         // ...
///     }
/// };
/// ```
/// e.g.
/// ```
/// pub const Move = struct {
///     pub const read = .{ .pos, .vel };
///     pub const write = .{.pos};
///
///     pub fn run(world: *World, ctx: Context) void {
///         var iter = world.iterator(&.{ .pos, .vel });
///         while (iter.next()) |item| {
///             const pos: *Pos, const vel: *Vel = item[1];
///             pos.x += vel.x * ctx.dt;
///             pos.y += vel.y * ctx.dt;
///         }
///     }
/// };
/// ```
pub fn execute(io_: ?Io, world: anytype, comptime Runners: []const type, ctx: anytype) !void {
    const step = ct: {
        const utils = struct {
            pub fn indexOf(comptime R: type) comptime_int {
                for (Runners, 0..) |Run, i| if (Run == R) return i;
                unreachable;
            }

            pub fn makeArray(comptime T: type, init: T, size: comptime_int) [size][size]T {
                comptime var ret: [size][size]T = undefined;
                for (0..size) |i| {
                    for (0..size) |j| ret[i][j] = init;
                }
                return ret;
            }

            pub fn conflict(comptime A: type, comptime B: type) bool {
                const Enum = std.meta.Child(@TypeOf(world)).Enum;
                const a_read: std.EnumSet(Enum) =
                    if (@hasDecl(A, "read"))
                        .initMany(&A.read)
                    else
                        .empty;
                const b_read: std.EnumSet(Enum) =
                    if (@hasDecl(B, "read"))
                        .initMany(&B.read)
                    else
                        .empty;
                const a_write: std.EnumSet(Enum) =
                    if (@hasDecl(A, "write"))
                        .initMany(&A.write)
                    else
                        .empty;
                const b_write: std.EnumSet(Enum) =
                    if (@hasDecl(B, "write"))
                        .initMany(&B.write)
                    else
                        .empty;

                return a_read.intersectWith(b_write).count() > 0 or
                    a_write.intersectWith(b_read).count() > 0 or
                    a_write.intersectWith(b_write).count() > 0;
            }
        };

        // before[i][j] => Runner i before Runner j.
        comptime var before = utils.makeArray(bool, false, Runners.len);
        inline for (Runners, 0..) |R, i| {
            if (@hasDecl(R, "after")) {
                inline for (R.after) |A| before[utils.indexOf(A)][i] = true;
            }
            if (@hasDecl(R, "before")) {
                inline for (R.before) |B| before[i][utils.indexOf(B)] = true;
            }
        }

        // conflict[i][j] = true => Runner i and Runner j cannot concurrent.
        comptime var conflict = utils.makeArray(bool, false, Runners.len);
        inline for (0..Runners.len) |i| inline for (0..Runners.len) |j| if (i != j) {
            conflict[i][j] = utils.conflict(Runners[i], Runners[j]);
        };

        // |R1, R2, R3| -> The same line can be concurrent.
        // |R4, R5, R6|
        // |R7, R8, R9|
        // -> The lines are synchronization points.
        comptime var step = utils.makeArray(?type, null, Runners.len);
        comptime var finished = 0;
        comptime var current_layer = 0;
        comptime var done: [Runners.len]bool = blk: {
            var ret: [Runners.len]bool = undefined;
            for (0..Runners.len) |i| ret[i] = false;
            break :blk ret;
        };
        inline while (finished < Runners.len) {
            const done_shot = done;
            const finished_shot = finished;
            comptime var current_layer_index = 0;

            inline for (0..Runners.len) |i| if (!done[i]) {
                comptime var ready = true;
                inline for (0..Runners.len) |j| {
                    if (before[j][i] and !done_shot[i]) ready = false;
                }
                if (ready) inline for (0..Runners.len) |j| if (step[current_layer][j]) |R| {
                    if (conflict[Runners[i]][utils.indexOf(R)]) ready = false;
                };
                if (ready) {
                    step[current_layer][current_layer_index] = Runners[i];
                    current_layer_index += 1;
                    done[i] = true;
                    finished += 1;
                }
            };

            current_layer += 1;

            if (finished_shot == finished) {
                @compileError("Runner has circular dependencies.");
            }
        }
        break :ct step;
    };

    if (io_) |io| {
        inline for (0..Runners.len) |i| {
            var futures: [Runners.len]?Io.Future(void) = blk: {
                var futures: [Runners.len]?Io.Future(void) = undefined;
                inline for (&futures) |*future| future.* = null;
                break :blk futures;
            };
            inline for (0..Runners.len) |j| if (step[i][j]) |R| {
                futures[j] = try io.concurrent(R.run, .{ world, ctx });
            };
            inline for (&futures) |*future|
                if (future.*) |*f| f.await(io);
        }
    } else {
        inline for (0..Runners.len) |i| inline for (0..Runners.len) |j|
            if (step[i][j]) |R| R.run(world, ctx);
    }
}

test {
    _ = map;
    _ = sparse_set;
    _ = archtype;
}

const Test = struct {
    pub const Pos = struct {
        x: f32,
        y: f32,
    };

    pub const Vel = struct {
        x: f32,
        y: f32,
    };

    pub const Enum = enum {
        pos,
        vel,
    };

    pub const Meta = struct {
        pos: Pos,
        vel: Vel,
    };

    pub const Entity = usize;

    pub const World = Map(Entity, Meta);
    pub const Context = struct {
        dt: f32,
    };

    pub const Move = struct {
        pub const read = .{ .pos, .vel };
        pub const write = .{.pos};

        pub fn run(world: *World, ctx: Context) void {
            var iter = world.iterator(&.{ .pos, .vel });
            while (iter.next()) |item| {
                const data = item[1];
                data.pos.x += data.vel.x * ctx.dt;
                data.pos.y += data.vel.y * ctx.dt;
            }
        }
    };
};

test execute {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var world: Test.World = .init(allocator);
    defer world.deinit();

    try world.register(0, .pos, Test.Pos{
        .x = 10,
        .y = 20,
    });
    try world.register(0, .vel, Test.Vel{
        .x = 1,
        .y = 2,
    });

    try execute(io, &world, &.{Test.Move}, Test.Context{
        .dt = 1,
    });

    try std.testing.expectEqual(Test.Pos{
        .x = 11,
        .y = 22,
    }, world.get(0, .pos));
}
