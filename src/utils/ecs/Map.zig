const std = @import("std");
const Allocator = std.mem.Allocator;
const HashMap = std.AutoHashMap;
const FieldAttributes = std.builtin.Type.Struct.FieldAttributes;

/// Strips every attribute from a generated field list.
///
/// The attributes stored in `Meta` describe the component payload type, not
/// the generated storage (`HashMap(Entity, T)`) or the generated pointer
/// (`*T`), so copying them would attach default values to fields whose type
/// no longer matches.
fn noAttributes(comptime len: usize) [len]FieldAttributes {
    var attrs: [len]FieldAttributes = undefined;
    for (&attrs) |*attr| attr.* = .{};
    return attrs;
}

/// Validates `Meta` and returns its type information.
fn structInfo(comptime Meta: type) std.builtin.Type.Struct {
    const info = @typeInfo(Meta);
    return switch (info) {
        .@"struct" => |payload| blk: {
            if (payload.is_tuple) {
                @compileError("Map: `Meta` must have named fields, but `" ++ @typeName(Meta) ++
                    "` is a tuple; declare one named field per component, " ++
                    "e.g. `struct { pos: Pos, vel: Vel }`");
            }
            if (payload.field_names.len == 0) {
                @compileError("Map: `Meta` must declare at least one component field, but `" ++
                    @typeName(Meta) ++ "` is empty; declare one named field per component, " ++
                    "e.g. `struct { pos: Pos }`");
            }
            break :blk payload;
        },
        else => @compileError("Map: `Meta` must be a struct type, but `" ++ @typeName(Meta) ++
            "` is a `" ++ @tagName(info) ++ "`; declare one named field per component, " ++
            "e.g. `struct { pos: Pos, vel: Vel }`"),
    };
}

fn MakeData(comptime Entity: type, comptime Meta: type) type {
    const info = structInfo(Meta);
    comptime var fields: [info.field_types.len]type = undefined;
    inline for (&fields, 0..) |*f, i| {
        f.* = HashMap(Entity, info.field_types[i]);
    }
    const attrs = noAttributes(info.field_types.len);
    return @Struct(
        .auto,
        null,
        info.field_names,
        &fields,
        &attrs,
    );
}

/// An entity-component storage built from a component schema.
///
/// `Map` keeps one `std.AutoHashMap(Entity, T)` per field of `Meta`: the fields
/// of `Meta` are the components, so `Meta` doubles as the component registry,
/// e.g. `Map(usize, struct { pos: Pos, vel: Vel })` stores a `Pos` and a `Vel`
/// per entity.
///
/// # Args
/// - `Entity`: the key that identifies an entity. It must be usable as a
///   `std.AutoHashMap` key (integer, enum, pointer, ...), so it must be
///   hashable and comparable by value.
/// - `Meta`: a struct with named fields; every field type is a component
///   payload and gets its own storage map.
///
/// # Note
/// The component set, the component types and their count are fixed at compile
/// time; there is no runtime registration of new component kinds. The initial
/// capacity of every map is zero, so `init` allocates lazily.
///
/// # Compile Error
/// - `Meta` is not a struct type.
/// - `Meta` is a tuple instead of a struct with named fields.
/// - `Meta` declares no fields.
///
/// See `init`, `deinit`, `register`, `get`, `has`, `iterator`.
pub fn Map(comptime Entity: type, comptime Meta: type) type {
    return struct {
        const Self = @This();

        /// The raw storage of `Self`: one `std.AutoHashMap` per component.
        ///
        /// Reaching into `data` bypasses the `Map` API and the compile-time
        /// component checking that comes with it; mutating a map directly also
        /// invalidates pointers previously returned by `Iterator.next`.
        ///
        /// # Note
        /// Prefer `register`, `get` and `has` over touching the maps directly.
        ///
        /// See `register`, `get`, `iterator`.
        pub const Data = MakeData(Entity, Meta);

        /// The enum of component names, derived from the fields of `Meta`.
        ///
        /// # Note
        /// Pass its members as the `comptime component` argument of `register`,
        /// `unregister`, `replace`, `get`, `has` and `iterator`. Using the enum
        /// keeps those calls checked at compile time, unlike raw tag names.
        ///
        /// See `TypeOf`, `iterator`.
        pub const Component = std.meta.FieldEnum(Meta);

        /// Resolves the payload type stored for one component.
        ///
        /// # Args
        /// - `component`: the component whose payload type is wanted; must be
        ///   known at compile time.
        ///
        /// # Return
        /// The type of the `Meta` field named by `component`, e.g. `TypeOf(.pos)`
        /// is `Pos` for `struct { pos: Pos }`.
        ///
        /// # Compile Error
        /// `component` is not a field of `Meta`.
        ///
        /// See `register`, `replace`, `get`.
        pub fn TypeOf(comptime component: Component) type {
            return @FieldType(Meta, @tagName(component));
        }

        /// Raw per-component storage. Mutating it invalidates pointers previously
        /// returned by `Iterator.next`; prefer the `Map` methods over using it.
        data: Data,

        /// Creates an empty `Map` with one hash map per component. Every value
        /// returned here owns its storage and must be released with exactly one
        /// `deinit`.
        ///
        /// # Args
        /// - `allocator`: shared by every component map. `deinit` frees through
        ///   the allocator stored inside each `std.AutoHashMap`.
        ///
        /// # Return
        /// A `Map` in which no entity owns any component.
        ///
        /// # Note
        /// No allocation happens here, so `init` cannot fail; the maps grow on
        /// their first `register`.
        ///
        /// See `deinit`, `register`.
        pub fn init(allocator: Allocator) Self {
            const info = structInfo(Meta);
            var data: Data = undefined;
            inline for (info.field_names, info.field_types) |name, T| {
                @field(data, name) = HashMap(Entity, T).init(allocator);
            }
            return .{ .data = data };
        }

        /// Releases the storage of every component map and consumes the map. Once
        /// this returns the map must not be used again, and no iterator over it
        /// may still be alive.
        ///
        /// # Return
        /// Nothing.
        ///
        /// # Note
        /// Only the map storage is freed; the component payloads themselves are
        /// not deinitialized. Types that own allocations must be freed first,
        /// e.g. by walking them with `iterator` and freeing each payload.
        ///
        /// See `init`, `unregister`, `iterator`.
        pub fn deinit(self: *Self) void {
            const info = structInfo(Meta);
            inline for (info.field_names) |name| {
                @field(self.data, name).deinit();
            }
            self.* = undefined;
        }

        /// Stores `value` as `entity`'s `component`, overwriting any old payload.
        /// An insertion may rehash the component map, so pointers previously
        /// obtained from `Iterator.next` or from `data` become invalid.
        ///
        /// # Args
        /// - `entity`: the entity that gains the component.
        /// - `component`: the component to write; must be known at compile time.
        /// - `value`: the payload, stored by copy.
        ///
        /// # Return
        /// `error.OutOfMemory` if the component map cannot grow, otherwise
        /// nothing.
        ///
        /// # Note
        /// An existing payload is overwritten silently, so `register` gives no
        /// hint whether the component was already present. Use `replace` when the
        /// previous payload is needed, or `has` to test presence first.
        ///
        /// # Compile Error
        /// - `component` is not a field of `Meta`.
        /// - `value` does not match `TypeOf(component)`.
        ///
        /// See `replace`, `unregister`, `get`, `has`.
        pub fn register(
            self: *Self,
            entity: Entity,
            comptime component: Component,
            value: TypeOf(component),
        ) Allocator.Error!void {
            try @field(self.data, @tagName(component)).put(entity, value);
        }

        /// Removes `entity`'s `component`, if it has one. Removal may rehash the
        /// component map, so pointers previously obtained from `Iterator.next` or
        /// from `data` become invalid; do not remove components while a loop over
        /// the map is still running.
        ///
        /// # Args
        /// - `entity`: the entity that loses the component.
        /// - `component`: the component to remove; must be known at compile time.
        ///
        /// # Return
        /// Nothing; removing a component the entity does not own is a no-op.
        ///
        /// # Note
        /// The call never allocates and never fails, so it cannot report whether
        /// anything was removed; use `has` to tell, or `get` to read the payload
        /// before removing it.
        ///
        /// # Compile Error
        /// `component` is not a field of `Meta`.
        ///
        /// See `register`, `has`, `get`, `iterator`.
        pub fn unregister(
            self: *Self,
            entity: Entity,
            comptime component: Component,
        ) void {
            _ = @field(self.data, @tagName(component)).remove(entity);
        }

        /// Stores `value` for `entity`'s `component`, returning the old payload.
        ///
        /// # Args
        /// - `entity`: the entity whose payload is swapped.
        /// - `component`: the component to update; must be known at compile time.
        /// - `value`: the new payload, stored by copy.
        ///
        /// # Return
        /// The previous payload, or `null` when `entity` does not own
        /// `component` (nothing is written in that case).
        ///
        /// # Note
        /// `replace` only updates an existing payload, so it never allocates and
        /// never fails; use `register` if inserting a missing component is
        /// acceptable. The `null` return does not distinguish "absent" from a
        /// payload value, but payloads are non-optional here, so use `has` when
        /// only presence matters.
        ///
        /// # Compile Error
        /// - `component` is not a field of `Meta`.
        /// - `value` does not match `TypeOf(component)`.
        ///
        /// See `register`, `get`, `has`.
        pub fn replace(
            self: *Self,
            entity: Entity,
            comptime component: Component,
            value: TypeOf(component),
        ) ?TypeOf(component) {
            const slot = @field(self.data, @tagName(component)).getPtr(entity) orelse return null;
            const old = slot.*;
            slot.* = value;
            return old;
        }

        /// Copies out `entity`'s payload for `component`.
        ///
        /// # Args
        /// - `entity`: the entity to look up.
        /// - `component`: the component to read; must be known at compile time.
        ///
        /// # Return
        /// A copy of the stored payload, or `null` when `entity` does not own
        /// `component`.
        ///
        /// # Note
        /// A `null` result means "this entity does not have this component"; use
        /// `has` when only presence matters, because that avoids copying the
        /// payload. The returned value is a copy, so writing to it does not
        /// change the stored data; use `iterator` (which yields pointers) or
        /// `replace` to modify a component.
        ///
        /// # Compile Error
        /// `component` is not a field of `Meta`.
        ///
        /// See `has`, `replace`, `iterator`.
        pub fn get(
            self: *Self,
            entity: Entity,
            comptime component: Component,
        ) ?TypeOf(component) {
            return @field(self.data, @tagName(component)).get(entity);
        }

        /// Tests whether `entity` currently owns `component`.
        ///
        /// # Args
        /// - `entity`: the entity to test.
        /// - `component`: the component to test for; must be known at compile
        ///   time.
        ///
        /// # Return
        /// `true` when a payload is stored for `entity`/`component`, otherwise
        /// `false`.
        ///
        /// # Note
        /// Prefer this over `get(...) != null` for pure existence checks: nothing
        /// is copied and nothing is allocated.
        ///
        /// # Compile Error
        /// `component` is not a field of `Meta`.
        ///
        /// See `get`, `register`, `unregister`.
        pub fn has(self: *Self, entity: Entity, comptime component: Component) bool {
            return @field(self.data, @tagName(component)).contains(entity);
        }

        /// Iterates the entities that own every requested component. The returned
        /// iterator borrows the map: the map must outlive it and must not be
        /// mutated through `register`, `unregister` or `deinit` until the loop
        /// ends, otherwise the yielded pointers and the cursor become invalid.
        ///
        /// # Args
        /// - `components`: the components an entity must own to be yielded; must
        ///   be known at compile time. `components[0]` drives the scan and the
        ///   remaining components filter the result.
        ///
        /// # Return
        /// An `Iterator(components)`; each `next` call yields
        /// `.{ entity, Item(components) }` while the iterator is not exhausted.
        ///
        /// # Note
        /// Yield order follows the storage map of `components[0]` and is not
        /// specified. Entities that miss any requested component are skipped, so
        /// the loop visits a subset rather than stopping early.
        ///
        /// # Compile Error
        /// - `components` is empty.
        /// - `components` repeats a component name.
        ///
        /// See `Iterator`, `Item`, `get`, `has`.
        pub fn iterator(self: *Self, comptime components: []const Component) Iterator(components) {
            comptime validateComponents(components);
            return .{
                .ref = self,
                .iter = @field(
                    self.data,
                    @tagName(components[0]),
                ).iterator(),
            };
        }

        /// Builds the item type yielded by `Iterator(components)`. Its fields are
        /// pointers into the component maps of the iterated map, so they stay
        /// valid only until that map is mutated.
        ///
        /// # Args
        /// - `components`: the components to expose; must be known at compile
        ///   time.
        ///
        /// # Return
        /// A struct type with one mutable pointer field per requested component,
        /// e.g. `Item(&.{.pos, .vel})` has fields `pos: *Pos` and `vel: *Vel`.
        ///
        /// # Note
        /// The fields are ordered like the declaration order of `Meta`, not like
        /// the order given in `components`, so `Item(&.{.vel, .pos})` still
        /// declares `pos` before `vel`; access the fields by name instead of
        /// relying on order. Items are returned inside a tuple, so `item[1]` is
        /// the `Item` and `item[0]` is the entity.
        ///
        /// # Compile Error
        /// - `components` is empty.
        /// - `components` repeats a component name.
        ///
        /// See `Iterator`, `iterator`.
        pub fn Item(comptime components: []const Component) type {
            comptime validateComponents(components);
            const info = structInfo(Meta);
            comptime var names: [components.len][:0]const u8 = undefined;
            comptime var types: [components.len]type = undefined;
            comptime var i = 0;
            inline for (
                info.field_names,
                info.field_types,
            ) |name, ty| {
                inline for (components) |component| {
                    if (std.mem.eql(u8, name, @tagName(component))) {
                        names[i] = name;
                        types[i] = *ty;
                        i += 1;
                    }
                }
            }
            const attrs = noAttributes(components.len);
            return @Struct(
                .auto,
                null,
                &names,
                &types,
                &attrs,
            );
        }

        /// The iterator type returned by `iterator`. It keeps a pointer to the map
        /// it scans plus a cursor into it, so the map must outlive the iterator
        /// and must not be mutated (or `deinit`-ed) until the iteration ends.
        ///
        /// # Args
        /// - `components`: the components every yielded item owns; must be known
        ///   at compile time.
        ///
        /// # Note
        /// Obtain it through `iterator`; the type is public so it can be named in
        /// function signatures and stored where the borrow rules allow.
        ///
        /// # Compile Error
        /// - `components` is empty.
        /// - `components` repeats a component name.
        ///
        /// See `iterator`, `Item`.
        pub fn Iterator(comptime components: []const Component) type {
            comptime validateComponents(components);
            return struct {
                ref: *Self,
                iter: HashMap(Entity, TypeOf(components[0])).Iterator,

                /// Advances to the next entity that owns every requested component.
                /// The data returned by a call aliases the maps it was taken from,
                /// so it stays valid only until the next mutation of the iterated
                /// map; edit payloads through the yielded pointers instead of
                /// calling `register` again.
                ///
                /// # Return
                /// `.{ entity, item }` where `item` is an `Item(components)` holding
                /// mutable pointers to the payloads, or `null` when the iterator is
                /// exhausted.
                ///
                /// # Note
                /// Entities missing at least one requested component are skipped
                /// rather than truncating the iteration, so `null` always means
                /// "no more matching entities".
                ///
                /// See `iterator`, `Item`, `get`.
                pub fn next(self: *Iterator(components)) ?struct { Entity, Item(components) } {
                    iterate: while (self.iter.next()) |entry| {
                        const entity = entry.key_ptr.*;
                        inline for (components[1..]) |component| {
                            if (!@field(self.ref.data, @tagName(component)).contains(entity)) {
                                continue :iterate;
                            }
                        }
                        var item: Item(components) = undefined;
                        inline for (@typeInfo(Item(components)).@"struct".field_names) |name| {
                            @field(item, name) = @field(self.ref.data, name).getPtr(entity) orelse
                                continue :iterate;
                        }
                        return .{ entity, item };
                    }
                    return null;
                }
            };
        }

        /// Fails compilation when `components` cannot drive an iteration.
        fn validateComponents(comptime components: []const Component) void {
            if (components.len == 0) {
                @compileError("Map: `components` must contain at least one component, " ++
                    "but it is empty; pass at least one component, e.g. `&.{.pos}`");
            }
            for (components, 0..) |lhs, i| {
                for (components[i + 1 ..]) |rhs| {
                    if (lhs == rhs) {
                        @compileError("Map: duplicate component `." ++ @tagName(lhs) ++
                            "` in `components`; list each component at most once, " ++
                            "e.g. `&.{.pos, .vel}`");
                    }
                }
            }
        }
    };
}

test "init/deinit" {
    var world: Map(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();
}

test "register/unregister" {
    var world: Map(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{
        .x = 20,
        .y = 30,
    });

    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 20,
        .y = 30,
    }, world.data.pos.get(0).?);

    world.unregister(0, .pos);

    try std.testing.expect(!world.data.pos.contains(0));
}

test "get/has" {
    var world: Map(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{
        .x = 20,
        .y = 30,
    });

    try std.testing.expect(world.has(0, .pos));
    try std.testing.expect(!world.has(0, .vel));
    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 20,
        .y = 30,
    }, world.get(0, .pos).?);

    world.unregister(0, .pos);

    try std.testing.expect(!world.has(0, .pos));
    try std.testing.expect(!world.has(0, .vel));
}

test "replace" {
    var world: Map(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{
        .x = 20,
        .y = 30,
    });

    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 20,
        .y = 30,
    }, world.data.pos.get(0).?);

    const ret = world.replace(0, .pos, .{
        .x = 40,
        .y = 50,
    });

    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 40,
        .y = 50,
    }, world.data.pos.get(0).?);
    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 20,
        .y = 30,
    }, ret);

    // `replace` reports absence with `null` and writes nothing.
    try std.testing.expectEqual(@as(?@TypeOf(world).TypeOf(.vel), null), world.replace(0, .vel, .{
        .x = 1,
        .y = 2,
    }));
    try std.testing.expect(!world.has(0, .vel));
}

test "iterator" {
    var world: Map(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{
        .x = 20,
        .y = 30,
    });
    try world.register(0, .vel, .{
        .x = 2,
        .y = 3,
    });
    try world.register(1, .pos, .{
        .x = 50,
        .y = 80,
    });
    try world.register(1, .vel, .{
        .x = 0,
        .y = -1,
    });
    // Entity 2 owns `pos` only, so an iteration over both components skips it.
    try world.register(2, .pos, .{
        .x = 1,
        .y = 1,
    });

    var visited: u32 = 0;
    var iter = world.iterator(&.{ .pos, .vel });
    while (iter.next()) |item| {
        const entity = item[0];
        const data = item[1];
        try std.testing.expect(entity == 0 or entity == 1);
        try std.testing.expect(world.has(entity, .pos));
        try std.testing.expect(world.has(entity, .vel));

        // The yielded pointers alias the stored payloads.
        data.pos.x += data.vel.x;
        data.pos.y += data.vel.y;
        visited += 1;
    }
    try std.testing.expectEqual(@as(u32, 2), visited);

    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 22,
        .y = 33,
    }, world.get(0, .pos).?);
    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 50,
        .y = 79,
    }, world.get(1, .pos).?);

    // A single-component iteration also yields the entity that misses `vel`.
    var pos_iter = world.iterator(&.{.pos});
    var pos_visited: u32 = 0;
    while (pos_iter.next()) |item| {
        try std.testing.expect(world.has(item[0], .pos));
        pos_visited += 1;
    }
    try std.testing.expectEqual(@as(u32, 3), pos_visited);
}

test "meta with default values" {
    const Pos = struct { x: f32, y: f32 };
    var world: Map(u32, struct {
        pos: Pos = .{ .x = 0, .y = 0 },
        vel: Pos,
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{ .x = 1, .y = 2 });
    var iter = world.iterator(&.{.pos});
    try std.testing.expectEqual(@as(f32, 1), iter.next().?[1].pos.x);
}
