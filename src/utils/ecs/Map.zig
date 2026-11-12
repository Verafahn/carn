const std = @import("std");
const Allocator = std.mem.Allocator;

/// Make ECS Type.
///
/// `Entity` requires a integer type, `Enum` requires an enum type,
/// `Components` requires a non tuple struct type.
/// `Enum` and `Components` requires all fields to be consistent.
///
/// This implementation stores data internally in a hash table:
/// ```
/// const InnerData = struct {
///     pos: std.AutoHashMap(Entity, Pos),
///     vel: std.AutoHashMap(Entity, Vel),
///     acc: std.AutoHashMap(Entity, Acc),
/// };
/// ```
///
/// Regarding the internal organizational structure, see `generateInnerDataType`.
pub fn Map(comptime Entity: type, comptime Enum: type, comptime Components: type) type {
    comptime verifyTypes(Entity, Enum, Components);
    return struct {
        const Self = @This();
        const Data = generateInnerDataType(Entity, Enum, Components);

        data: Data,

        pub fn init(gpa: Allocator) Self {
            var data: Data = undefined;
            inline for (@typeInfo(Enum).@"enum".field_names) |name| {
                @field(data, name) = .init(gpa);
            }
            return .{ .data = data };
        }

        pub fn deinit(self: *Self) void {
            inline for (@typeInfo(Enum).@"enum".field_names) |name| {
                @field(self.data, name).deinit();
            }
        }

        /// Register a new `component` for `entity` and use `data` as the initial value.
        ///
        /// If the component already exists, it will overwrite the old value.
        pub fn register(
            self: *Self,
            entity: Entity,
            comptime component: Enum,
            data: @FieldType(Components, @tagName(component)),
        ) Allocator.Error!void {
            const p: *std.AutoHashMap(Entity, @TypeOf(data)) = &@field(
                self.data,
                @tagName(component),
            );
            try p.put(entity, data);
        }

        /// Remove the `component` of `entity` and return its value.
        ///
        /// If `entity` does not have a `component`, return `null`.
        pub fn remove(
            self: *Self,
            entity: Entity,
            comptime component: Enum,
        ) ?@FieldType(Components, @tagName(component)) {
            const p: *std.AutoHashMap(
                Entity,
                @FieldType(Components, @tagName(component)),
            ) = &@field(
                self.data,
                @tagName(component),
            );
            const kv = p.fetchRemove(entity) orelse return null;
            return kv.value;
        }

        /// Obtain a copy of the `component` data for the `entity`.
        ///
        /// If `entity` does not have a `component`, return `null`.
        pub fn get(
            self: *Self,
            entity: Entity,
            comptime component: Enum,
        ) ?@FieldType(Components, @tagName(component)) {
            const p: *std.AutoHashMap(
                Entity,
                @FieldType(Components, @tagName(component)),
            ) = &@field(
                self.data,
                @tagName(component),
            );
            return p.get(entity);
        }

        /// Retrieve the `component` data pointer of the `entity`.
        ///
        /// If `entity` does not have a `component`, return `null`.
        pub fn getPtr(
            self: *Self,
            entity: Entity,
            comptime component: Enum,
        ) ?*@FieldType(Components, @tagName(component)) {
            const p: *std.AutoHashMap(
                Entity,
                @FieldType(Components, @tagName(component)),
            ) = &@field(
                self.data,
                @tagName(component),
            );
            return p.getPtr(entity);
        }

        /// Check if the `entity` has `component`.
        pub fn has(
            self: *Self,
            entity: Entity,
            comptime component: Enum,
        ) bool {
            const p: *std.AutoHashMap(
                Entity,
                @FieldType(Components, @tagName(component)),
            ) = &@field(
                self.data,
                @tagName(component),
            );
            return p.contains(entity);
        }

        pub fn Iterator(comptime E: []const Enum) type {
            return struct {
                const Iter = @This();
                const Component = E[1..];
                const Base = @FieldType(Data, @tagName(E[0]));

                fn Ret() type {
                    comptime var types: [E.len]type = undefined;
                    for (E, 0..) |component, i| {
                        types[i] = *@FieldType(Components, @tagName(component));
                    }
                    return @Tuple(&types);
                }

                ref: *Self,
                base: Base.Iterator,

                pub fn next(self: *Iter) ?struct { Entity, Ret() } {
                    var ret: Ret() = undefined;
                    while (self.base.next()) |v| {
                        const key = v.key_ptr.*;
                        ret[0] = v.value_ptr;
                        inline for (Component, 1..) |component, i| {
                            if (self.ref.getPtr(key, component)) |item| {
                                ret[i] = item;
                            } else {
                                return null;
                            }
                        }
                        return .{ key, ret };
                    }
                    return null;
                }
            };
        }

        /// Joint Component Iterator
        ///
        /// Only entities that have all the components in `components` will be iterated to,
        /// and component data will be returned in tuple form in the order given by the parameters.
        pub fn iterator(self: *Self, comptime components: []const Enum) Iterator(components) {
            return .{
                .ref = self,
                .base = @field(self.data, @tagName(components[0])).iterator(),
            };
        }
    };
}

/// ```zig
/// struct {
///     a: usize,
///     b: f32,
///     c: *anyopaque,
/// };
/// ```
/// transform into
/// ```
/// struct {
///     a: std.AutoHashMap(Entity, usize),
///     b: std.AutoHashMap(Entity, f32),
///     c: std.AutoHashMap(Entity, *anyopaque),
/// };
/// ```
fn generateInnerDataType(comptime Entity: type, comptime Enum: type, comptime Components: type) type {
    const field_names = @typeInfo(Enum).@"enum".field_names;
    const count = field_names.len;
    comptime var types: [count]type = undefined;
    for (@typeInfo(Components).@"struct".field_types, 0..) |T, i| {
        types[i] = std.AutoHashMap(Entity, T);
    }
    comptime var attrs: [count]std.builtin.Type.Struct.FieldAttributes = undefined;
    for (0..attrs.len) |i| {
        attrs[i] = .{};
    }
    return @Struct(
        .auto,
        null,
        field_names,
        &types,
        &attrs,
    );
}

fn verifyTypes(comptime Entity: type, comptime Enum: type, comptime Components: type) void {
    if (@typeInfo(Entity) != .int) {
        @compileError("Expected integer type, but it is " ++ @typeName(Entity) ++ ".");
    }
    if (@typeInfo(Enum) != .@"enum") {
        @compileError("Expected enum type, but it is " ++ @typeName(Enum) ++ ".");
    }
    if (@typeInfo(Components) != .@"struct" or @typeInfo(Components).@"struct".is_tuple) {
        @compileError("Expected non tuple struct type, but it is " ++ @typeName(Enum) ++ ".");
    }
    const enum_field_names = @typeInfo(Enum).@"enum".field_names;
    for (enum_field_names) |name| {
        if (!@hasField(Components, name)) {
            @compileError("Missing declared field " ++ name ++ ".");
        }
    }
}

const Test = struct {
    pub const EntityGenerator = struct {
        i: usize = 0,

        pub fn next(self: *EntityGenerator) usize {
            self.i +%= 1;
            return self.i;
        }
    };

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

    pub const Component = struct {
        pos: Pos,
        vel: Vel,
    };
};

const testing = std.testing;

test Map {
    const World = Map(usize, Test.Enum, Test.Component);
    var world: World = .init(std.testing.allocator);
    defer world.deinit();
}

test "base" {
    var g: Test.EntityGenerator = .{};

    const World = Map(usize, Test.Enum, Test.Component);
    var world: World = .init(std.testing.allocator);
    defer world.deinit();

    const id1 = g.next();
    try world.register(id1, .pos, Test.Pos{
        .x = 10.0,
        .y = 20.0,
    });
    try world.register(id1, .vel, Test.Vel{
        .x = 1.0,
        .y = 5.0,
    });

    try testing.expect(world.has(id1, .pos));
    try testing.expect(world.has(id1, .vel));
    try testing.expectEqual(Test.Pos{
        .x = 10.0,
        .y = 20.0,
    }, world.get(id1, .pos));
    try testing.expectEqual(Test.Vel{
        .x = 1.0,
        .y = 5.0,
    }, world.get(id1, .vel));

    try testing.expectEqual(Test.Pos{
        .x = 10.0,
        .y = 20.0,
    }, world.remove(id1, .pos));
    try testing.expectEqual(Test.Vel{
        .x = 1.0,
        .y = 5.0,
    }, world.remove(id1, .vel));

    try testing.expect(!world.has(id1, .pos));
    try testing.expect(!world.has(id1, .vel));
}

test "iterator" {
    const World = Map(usize, Test.Enum, Test.Component);
    var world: World = .init(std.testing.allocator);
    defer world.deinit();

    for (0..100) |i| {
        if (i % 2 == 0) {
            try world.register(i, .pos, Test.Pos{
                .x = 20.0,
                .y = 30.0,
            });
        }
        if (i % 3 == 0) {
            try world.register(i, .vel, Test.Vel{
                .x = 10.0,
                .y = 15.0,
            });
        }
    }

    var iter_pos = world.iterator(&.{.pos});
    while (iter_pos.next()) |kv| {
        try testing.expect(kv[0] % 2 == 0);
    }

    var iter_vel = world.iterator(&.{.vel});
    while (iter_vel.next()) |kv| {
        try testing.expect(kv[0] % 3 == 0);
    }

    var iter_pos_vel = world.iterator(&.{ .pos, .vel });
    while (iter_pos_vel.next()) |kv| {
        try testing.expect(kv[0] % 2 == 0 and kv[0] % 3 == 0);
    }
}
