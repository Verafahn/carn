const std = @import("std");
const Allocator = std.mem.Allocator;
const FieldAttributes = std.builtin.Type.Struct.FieldAttributes;

/// Strips every attribute from a generated field list.
///
/// The attributes stored in `Meta` describe the component payload type, not
/// the generated storage, so copying them would attach default values to
/// fields whose type no longer matches.
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
                @compileError("Archtype: `Meta` must have named fields, but `" ++ @typeName(Meta) ++
                    "` is a tuple; declare one named field per component, " ++
                    "e.g. `struct { pos: Pos, vel: Vel }`");
            }
            if (payload.field_names.len == 0) {
                @compileError("Archtype: `Meta` must declare at least one component field, but `" ++
                    @typeName(Meta) ++ "` is empty; declare one named field per component, " ++
                    "e.g. `struct { pos: Pos }`");
            }
            break :blk payload;
        },
        else => @compileError("Archtype: `Meta` must be a struct type, but `" ++ @typeName(Meta) ++
            "` is a `" ++ @tagName(info) ++ "`; declare one named field per component, " ++
            "e.g. `struct { pos: Pos, vel: Vel }`"),
    };
}

/// Builds the column struct of one archetype: one `std.ArrayList` per component
/// selected by `mask`.
fn MakeColumns(comptime Meta: type, comptime mask: usize) type {
    const info = structInfo(Meta);
    const selected = comptime blk: {
        var count: usize = 0;
        for (0..info.field_types.len) |i| {
            if (mask & (@as(usize, 1) << i) != 0) count += 1;
        }
        break :blk count;
    };

    comptime var names: [selected][:0]const u8 = undefined;
    comptime var types: [selected]type = undefined;
    comptime var n: usize = 0;
    inline for (info.field_names, info.field_types, 0..) |name, T, i| {
        if (mask & (@as(usize, 1) << i) != 0) {
            names[n] = name;
            types[n] = std.ArrayList(T);
            n += 1;
        }
    }

    const attrs = noAttributes(selected);
    return @Struct(
        .auto,
        null,
        &names,
        &types,
        &attrs,
    );
}

/// One archetype: the packed entities that share the component set encoded by
/// `mask`, their columns, and an index from entity to row.
fn ArchetypeStore(comptime Entity: type, comptime Meta: type, comptime mask: usize) type {
    const info = structInfo(Meta);
    const Columns = MakeColumns(Meta, mask);

    return struct {
        const Self = @This();

        /// Columns of this archetype; every column has one entry per row.
        columns: Columns,
        /// Packed entities, parallel to every column.
        entities: std.ArrayList(Entity),
        /// Maps an entity to its row in `entities` and every column.
        rows: std.AutoHashMap(Entity, usize),
        allocator: Allocator,

        fn init(allocator: Allocator) Self {
            var columns: Columns = undefined;
            inline for (info.field_names, info.field_types, 0..) |name, T, i| {
                if (mask & (@as(usize, 1) << i) != 0) {
                    @field(columns, name) = std.ArrayList(T).empty;
                }
            }
            return .{
                .columns = columns,
                .entities = .empty,
                .rows = std.AutoHashMap(Entity, usize).init(allocator),
                .allocator = allocator,
            };
        }

        fn deinit(self: *Self) void {
            inline for (info.field_names, 0..) |name, i| {
                if (mask & (@as(usize, 1) << i) != 0) {
                    @field(self.columns, name).deinit(self.allocator);
                }
            }
            self.entities.deinit(self.allocator);
            self.rows.deinit();
            self.* = undefined;
        }

        fn rowOf(self: *const Self, entity: Entity) ?usize {
            return self.rows.get(entity);
        }

        /// Reserves one row in every column and in the entity index.
        fn ensureRowCapacity(self: *Self, allocator: Allocator) Allocator.Error!void {
            try self.entities.ensureUnusedCapacity(allocator, 1);
            inline for (info.field_names, 0..) |name, i| {
                if (mask & (@as(usize, 1) << i) != 0) {
                    try @field(self.columns, name).ensureUnusedCapacity(allocator, 1);
                }
            }
            try self.rows.ensureUnusedCapacity(1);
        }

        /// Removes one row with a swap-remove and keeps the surviving rows
        /// addressable. Never allocates.
        fn removeRow(self: *Self, entity: Entity) void {
            const row = self.rows.get(entity) orelse return;
            const last = self.entities.items.len - 1;

            if (row != last) {
                const moved = self.entities.items[last];
                self.entities.items[row] = moved;
                inline for (info.field_names, 0..) |name, i| {
                    if (mask & (@as(usize, 1) << i) != 0) {
                        @field(self.columns, name).items[row] = @field(self.columns, name).items[last];
                    }
                }
                self.rows.getPtr(moved).?.* = row;
            }

            self.entities.items.len = last;
            inline for (info.field_names, 0..) |name, i| {
                if (mask & (@as(usize, 1) << i) != 0) {
                    @field(self.columns, name).items.len = last;
                }
            }
            _ = self.rows.remove(entity);
        }
    };
}

fn MakeData(comptime Entity: type, comptime Meta: type) type {
    const info = structInfo(Meta);
    const max_mask = (@as(usize, 1) << info.field_types.len) - 1;

    comptime var names: [max_mask][:0]const u8 = undefined;
    comptime var types: [max_mask]type = undefined;
    inline for (0..max_mask) |i| {
        names[i] = std.fmt.comptimePrint("m{d}", .{i + 1});
        types[i] = ArchetypeStore(Entity, Meta, i + 1);
    }

    const attrs = noAttributes(max_mask);
    return @Struct(
        .auto,
        null,
        &names,
        &types,
        &attrs,
    );
}

/// An entity-component storage built from a component schema.
///
/// `Archtype` groups entities by their set of components: every non-empty
/// subset of the fields of `Meta` is one archetype, and entities that own the
/// exact same components share a packed row in that archetype. A component is
/// stored in a column, so iteration over an archetype walks contiguous memory.
///
/// # Args
/// - `Entity`: the key that identifies an entity. It must be usable as a
///   `std.AutoHashMap` key (integer, enum, pointer, ...), so it must be
///   hashable and comparable by value.
/// - `Meta`: a struct with named fields; every field type is a component
///   payload and becomes a column of the archetypes that contain it.
///
/// # Note
/// The component set, the component types and their count are fixed at compile
/// time; there is no runtime registration of new component kinds. Moving an
/// entity between archetypes on `register` may allocate, while `unregister`
/// only clears the logical component bit so it stays infallible. Removing a
/// component therefore leaves the entity in its old row until the entity loses
/// its last component; iteration filters those rows through the logical mask.
///
/// # Compile Error
/// - `Meta` is not a struct type.
/// - `Meta` is a tuple instead of a struct with named fields.
/// - `Meta` declares no fields.
///
/// See `init`, `deinit`, `register`, `get`, `has`, `iterator`.
pub fn Archtype(comptime Entity: type, comptime Meta: type) type {
    return struct {
        const Self = @This();
        const info = structInfo(Meta);
        const MaskCount = @as(usize, 1) << info.field_types.len;
        const Record = struct {
            /// Components the entity logically owns.
            logical: usize,
            /// Components the archetype physically stores for the entity.
            /// Always a superset of `logical`.
            physical: usize,
        };

        /// The raw storage of `Self`: one archetype per non-empty component set.
        ///
        /// Reaching into `data` bypasses the `Archtype` API and the compile-time
        /// component checking that comes with it; mutating an archetype
        /// directly also invalidates pointers previously returned by
        /// `Iterator.next`.
        ///
        /// # Note
        /// Prefer `register`, `get` and `has` over touching the archetypes
        /// directly.
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

        /// Raw per-archetype storage. Mutating it invalidates pointers previously
        /// returned by `Iterator.next`; prefer the `Archtype` methods over using
        /// it.
        data: Data,

        /// Logical and physical component masks of every live entity.
        entities: std.AutoHashMap(Entity, Record),

        /// Creates an empty `Archtype` with one storage per component set. Every
        /// value returned here owns its storage and must be released with
        /// exactly one `deinit`.
        ///
        /// # Args
        /// - `allocator`: shared by every archetype. `deinit` frees through the
        ///   allocator stored inside each archetype.
        ///
        /// # Return
        /// An `Archtype` in which no entity owns any component.
        ///
        /// # Note
        /// No allocation happens here, so `init` cannot fail; the archetypes
        /// grow on their first `register`.
        ///
        /// See `deinit`, `register`.
        pub fn init(allocator: Allocator) Self {
            var data: Data = undefined;
            const data_info = @typeInfo(Data).@"struct";
            inline for (data_info.field_names, data_info.field_types) |name, T| {
                @field(data, name) = T.init(allocator);
            }
            return .{
                .data = data,
                .entities = std.AutoHashMap(Entity, Record).init(allocator),
            };
        }

        /// Releases the storage of every archetype and consumes the value. Once
        /// this returns the value must not be used again, and no iterator over
        /// it may still be alive.
        ///
        /// # Return
        /// Nothing.
        ///
        /// # Note
        /// Only the archetype storage is freed; the component payloads themselves
        /// are not deinitialized. Types that own allocations must be freed first,
        /// e.g. by walking them with `iterator` and freeing each payload.
        ///
        /// See `init`, `unregister`, `iterator`.
        pub fn deinit(self: *Self) void {
            const data_info = @typeInfo(Data).@"struct";
            inline for (data_info.field_names) |name| {
                @field(self.data, name).deinit();
            }
            self.entities.deinit();
            self.* = undefined;
        }

        /// Stores `value` as `entity`'s `component`, overwriting any old payload.
        /// Adding a component the entity did not physically have may move it to
        /// another archetype, which can reallocate and invalidates pointers
        /// previously obtained from `Iterator.next`.
        ///
        /// # Args
        /// - `entity`: the entity that gains the component.
        /// - `component`: the component to write; must be known at compile time.
        /// - `value`: the payload, stored by copy.
        ///
        /// # Return
        /// `error.OutOfMemory` if the target archetype cannot grow, otherwise
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
            const bit = comptime bitOf(component);
            const record = self.entities.get(entity) orelse Record{
                .logical = 0,
                .physical = 0,
            };
            const new_logical = record.logical | bit;

            // The physical archetype already owns this column, so the payload can
            // be written in place even if the component is logically absent.
            if (record.physical & bit != 0) {
                inline for (1..MaskCount) |physical| {
                    if ((physical & bit) != 0) {
                        if (record.physical == physical) {
                            const store = self.storePtr(physical);
                            const row = store.rowOf(entity) orelse unreachable;
                            @field(store.columns, @tagName(component)).items[row] = value;
                            self.entities.getPtr(entity).?.logical = new_logical;
                            return;
                        }
                    }
                }
                unreachable;
            }

            // The entity has to move (or be created) in a larger archetype.
            if (record.physical == 0) {
                try self.insertFresh(entity, bit, new_logical, component, value);
                return;
            }

            inline for (1..MaskCount) |physical| {
                if (record.physical == physical) {
                    return self.moveRow(
                        physical,
                        physical | bit,
                        entity,
                        new_logical,
                        component,
                        value,
                    );
                }
            }
            unreachable;
        }

        /// Removes `entity`'s `component`, if it has one.
        ///
        /// # Args
        /// - `entity`: the entity that loses the component.
        /// - `component`: the component to remove; must be known at compile time.
        ///
        /// # Return
        /// Nothing; removing a component the entity does not own is a no-op.
        ///
        /// # Note
        /// Only the logical component bit is cleared, so the call never
        /// allocates, never fails and never moves the entity. Iteration filters
        /// rows through the logical mask, so a component that was removed is no
        /// longer observable. Use `has` to test presence, or `get` to read the
        /// payload before removing it.
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
            const bit = comptime bitOf(component);
            const record = self.entities.getPtr(entity) orelse return;
            if (record.logical & bit == 0) return;

            const new_logical = record.logical & ~bit;
            if (new_logical != 0) {
                record.logical = new_logical;
                return;
            }

            // Without any logical component the row is dead; drop it. This is a
            // swap-remove in the entity's last archetype and never allocates.
            const physical = record.physical;
            inline for (1..MaskCount) |mask| {
                if (physical == mask) {
                    self.storePtr(mask).removeRow(entity);
                }
            }
            _ = self.entities.remove(entity);
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
            const bit = comptime bitOf(component);
            const record = self.entities.get(entity) orelse return null;
            if (record.logical & bit == 0) return null;

            inline for (1..MaskCount) |physical| {
                if ((physical & bit) != 0) {
                    if (record.physical == physical) {
                        const store = self.storePtr(physical);
                        const row = store.rowOf(entity) orelse unreachable;
                        const slot = &@field(store.columns, @tagName(component)).items[row];
                        const old = slot.*;
                        slot.* = value;
                        return old;
                    }
                }
            }
            unreachable;
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
            const bit = comptime bitOf(component);
            const record = self.entities.get(entity) orelse return null;
            if (record.logical & bit == 0) return null;

            inline for (1..MaskCount) |physical| {
                if ((physical & bit) != 0) {
                    if (record.physical == physical) {
                        const store = self.storePtr(physical);
                        const row = store.rowOf(entity) orelse unreachable;
                        return @field(store.columns, @tagName(component)).items[row];
                    }
                }
            }
            unreachable;
        }

        /// Tests whether `entity` currently owns `component`.
        ///
        /// # Args
        /// - `entity`: the entity to test.
        /// - `component`: the component to test for; must be known at compile
        ///   time.
        ///
        /// # Return
        /// `true` when the component is logically present for `entity`,
        /// otherwise `false`.
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
            const bit = comptime bitOf(component);
            const record = self.entities.get(entity) orelse return false;
            return record.logical & bit != 0;
        }

        /// Iterates the entities that own every requested component. The returned
        /// iterator borrows the value: it must outlive it and it must not be
        /// mutated through `register`, `unregister` or `deinit` until the loop
        /// ends, otherwise the yielded pointers and the cursor become invalid.
        ///
        /// # Args
        /// - `components`: the components an entity must own to be yielded; must
        ///   be known at compile time. The iterator scans the archetypes that
        ///   physically contain all of them and filters rows by the logical mask.
        ///
        /// # Return
        /// An `Iterator(components)`; each `next` call yields
        /// `.{ entity, Item(components) }` while the iterator is not exhausted.
        ///
        /// # Note
        /// Yield order follows archetype order and the packed rows inside each
        /// archetype; the order is not specified. Entities that miss any
        /// requested component are skipped, so the loop visits a subset rather
        /// than stopping early.
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
                .archetype = 0,
                .row = 0,
            };
        }

        /// Builds the item type yielded by `Iterator(components)`. Its fields are
        /// pointers into the archetype columns of the iterated value, so they
        /// stay valid only until that value is mutated.
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
            const meta_info = structInfo(Meta);
            comptime var names: [components.len][:0]const u8 = undefined;
            comptime var types: [components.len]type = undefined;
            comptime var i = 0;
            inline for (
                meta_info.field_names,
                meta_info.field_types,
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

        /// The iterator type returned by `iterator`. It keeps a pointer to the
        /// value it scans plus a cursor into the archetype list and the packed
        /// rows of the current archetype, so the value must outlive the iterator
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
                const masks = candidateMasks(components);
                const required = requiredBits(components);

                ref: *Self,
                archetype: usize,
                row: usize,

                /// Advances to the next entity that owns every requested component.
                /// The data returned by a call aliases the archetype columns it was
                /// taken from, so it stays valid only until the next mutation of the
                /// iterated value; edit payloads through the yielded pointers
                /// instead of calling `register` again.
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
                    scan: while (self.archetype < masks.len) {
                        const physical = masks[self.archetype];
                        inline for (masks) |mask| {
                            if (physical == mask) {
                                const store = self.ref.storePtr(mask);
                                if (self.row >= store.entities.items.len) {
                                    self.archetype += 1;
                                    self.row = 0;
                                    continue :scan;
                                }

                                const row = self.row;
                                self.row += 1;
                                const entity = store.entities.items[row];
                                const record = self.ref.entities.get(entity) orelse continue :scan;
                                if (record.logical & required != required) continue :scan;

                                var item: Item(components) = undefined;
                                inline for (@typeInfo(Item(components)).@"struct".field_names) |name| {
                                    @field(item, name) = &@field(store.columns, name).items[row];
                                }
                                return .{ entity, item };
                            }
                        }
                        unreachable;
                    }
                    return null;
                }
            };
        }

        /// Fails compilation when `components` cannot drive an iteration.
        fn validateComponents(comptime components: []const Component) void {
            if (components.len == 0) {
                @compileError("Archtype: `components` must contain at least one component, " ++
                    "but it is empty; pass at least one component, e.g. `&.{.pos}`");
            }
            for (components, 0..) |lhs, i| {
                for (components[i + 1 ..]) |rhs| {
                    if (lhs == rhs) {
                        @compileError("Archtype: duplicate component `." ++ @tagName(lhs) ++
                            "` in `components`; list each component at most once, " ++
                            "e.g. `&.{.pos, .vel}`");
                    }
                }
            }
        }

        /// Returns the mask bit of one component.
        fn bitOf(comptime component: Component) usize {
            return @as(usize, 1) << @backingInt(component);
        }

        /// Returns the union of the mask bits of `components`.
        fn requiredBits(comptime components: []const Component) usize {
            var bits: usize = 0;
            for (components) |component| {
                bits |= bitOf(component);
            }
            return bits;
        }

        /// Returns how many archetypes contain every requested component.
        fn candidateCount(comptime components: []const Component) usize {
            const required = requiredBits(components);
            var count: usize = 0;
            for (1..MaskCount) |mask| {
                if (mask & required == required) count += 1;
            }
            return count;
        }

        /// Returns the masks of the archetypes that contain every requested
        /// component, in ascending order.
        fn candidateMasks(comptime components: []const Component) [candidateCount(components)]usize {
            const required = requiredBits(components);
            var masks: [candidateCount(components)]usize = undefined;
            var i: usize = 0;
            for (1..MaskCount) |mask| {
                if (mask & required == required) {
                    masks[i] = mask;
                    i += 1;
                }
            }
            return masks;
        }

        /// Returns the storage of the archetype encoded by `mask`.
        fn Store(comptime mask: usize) type {
            return ArchetypeStore(Entity, Meta, mask);
        }

        fn storePtr(self: *Self, comptime mask: usize) *Store(mask) {
            return &@field(self.data, @typeInfo(Data).@"struct".field_names[mask - 1]);
        }

        /// Creates the row of a new entity in the archetype that only contains
        /// the freshly registered component.
        fn insertFresh(
            self: *Self,
            entity: Entity,
            comptime mask: usize,
            logical: usize,
            comptime component: Component,
            value: TypeOf(component),
        ) Allocator.Error!void {
            const store = self.storePtr(mask);
            try self.entities.ensureUnusedCapacity(1);
            try store.ensureRowCapacity(store.allocator);

            const row = store.entities.items.len;
            store.entities.appendAssumeCapacity(entity);
            inline for (info.field_names, 0..) |name, i| {
                if (mask & (@as(usize, 1) << i) != 0) {
                    if (comptime std.mem.eql(u8, name, @tagName(component))) {
                        @field(store.columns, name).appendAssumeCapacity(value);
                    } else {
                        unreachable;
                    }
                }
            }
            store.rows.putAssumeCapacity(entity, row);
            self.entities.putAssumeCapacity(entity, .{
                .logical = logical,
                .physical = mask,
            });
        }

        /// Moves an entity from `old_mask` to `new_mask`, carrying every
        /// physically stored column over and writing `value` into the newly
        /// added component.
        fn moveRow(
            self: *Self,
            comptime old_mask: usize,
            comptime new_mask: usize,
            entity: Entity,
            logical: usize,
            comptime component: Component,
            value: TypeOf(component),
        ) Allocator.Error!void {
            const old = self.storePtr(old_mask);
            const new = self.storePtr(new_mask);
            const old_row = old.rowOf(entity) orelse unreachable;
            const new_row = new.entities.items.len;

            // Reserve everything first; past this point the move cannot fail.
            try new.ensureRowCapacity(new.allocator);

            new.entities.appendAssumeCapacity(entity);
            inline for (info.field_names, 0..) |name, i| {
                const bit = @as(usize, 1) << i;
                if (new_mask & bit != 0) {
                    if (old_mask & bit != 0) {
                        @field(new.columns, name).appendAssumeCapacity(
                            @field(old.columns, name).items[old_row],
                        );
                    } else {
                        if (comptime !std.mem.eql(u8, name, @tagName(component))) {
                            unreachable;
                        }
                        @field(new.columns, name).appendAssumeCapacity(value);
                    }
                }
            }
            new.rows.putAssumeCapacity(entity, new_row);

            old.removeRow(entity);
            self.entities.getPtr(entity).?.* = .{
                .logical = logical,
                .physical = new_mask,
            };
        }
    };
}

test "init/deinit" {
    var world: Archtype(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();
}

test "register/unregister" {
    var world: Archtype(u32, struct {
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
    }, world.get(0, .pos).?);

    world.unregister(0, .pos);

    try std.testing.expect(!world.has(0, .pos));
}

test "get/has" {
    var world: Archtype(u32, struct {
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
    var world: Archtype(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{
        .x = 20,
        .y = 30,
    });

    const ret = world.replace(0, .pos, .{
        .x = 40,
        .y = 50,
    });

    try std.testing.expectEqual(@TypeOf(world).TypeOf(.pos){
        .x = 40,
        .y = 50,
    }, world.get(0, .pos).?);
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
    var world: Archtype(u32, struct {
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

test "unregister keeps the other components observable" {
    var world: Archtype(u32, struct {
        pos: struct { x: f32, y: f32 },
        vel: struct { x: f32, y: f32 },
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{ .x = 1, .y = 2 });
    try world.register(0, .vel, .{ .x = 3, .y = 4 });

    world.unregister(0, .vel);

    try std.testing.expect(world.has(0, .pos));
    try std.testing.expect(!world.has(0, .vel));
    try std.testing.expectEqual(@as(f32, 1), world.get(0, .pos).?.x);
    try std.testing.expectEqual(@as(?@TypeOf(world).TypeOf(.vel), null), world.get(0, .vel));

    // Re-registering after a removal overwrites the stale physical column.
    try world.register(0, .vel, .{ .x = 7, .y = 8 });
    try std.testing.expectEqual(@as(f32, 7), world.get(0, .vel).?.x);

    var iter = world.iterator(&.{ .pos, .vel });
    var visited: u32 = 0;
    while (iter.next()) |item| {
        try std.testing.expectEqual(@as(u32, 0), item[0]);
        visited += 1;
    }
    try std.testing.expectEqual(@as(u32, 1), visited);

    // Losing every component drops the row again, without leaking.
    world.unregister(0, .pos);
    world.unregister(0, .vel);
    try std.testing.expect(!world.has(0, .pos));
    try std.testing.expect(!world.has(0, .vel));
}

test "three components move between archetypes" {
    const A = struct { v: i32 };
    const B = struct { v: i32 };
    const C = struct { v: i32 };
    var world: Archtype(u32, struct { a: A, b: B, c: C }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .a, .{ .v = 1 });
    try world.register(0, .b, .{ .v = 2 });
    try world.register(0, .c, .{ .v = 3 });
    try world.register(1, .a, .{ .v = 10 });
    try world.register(1, .c, .{ .v = 30 });

    // Removing the middle component keeps the surviving columns observable.
    world.unregister(0, .b);
    try std.testing.expectEqual(@as(i32, 1), world.get(0, .a).?.v);
    try std.testing.expectEqual(@as(i32, 3), world.get(0, .c).?.v);
    try std.testing.expect(!world.has(0, .b));

    var iter = world.iterator(&.{ .a, .c });
    var sum: i32 = 0;
    var count: u32 = 0;
    while (iter.next()) |item| {
        sum += item[1].a.v + item[1].c.v;
        count += 1;
    }
    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(i32, 44), sum);

    // Re-adding `b` leaves entity 0 as the only owner of all three.
    try world.register(0, .b, .{ .v = 20 });
    var all = world.iterator(&.{ .a, .b, .c });
    var rows: u32 = 0;
    while (all.next()) |item| {
        try std.testing.expectEqual(@as(u32, 0), item[0]);
        try std.testing.expectEqual(@as(i32, 20), item[1].b.v);
        rows += 1;
    }
    try std.testing.expectEqual(@as(u32, 1), rows);
}

test "meta with default values" {
    const Pos = struct { x: f32, y: f32 };
    var world: Archtype(u32, struct {
        pos: Pos = .{ .x = 0, .y = 0 },
        vel: Pos,
    }) = .init(std.testing.allocator);
    defer world.deinit();

    try world.register(0, .pos, .{ .x = 1, .y = 2 });
    var iter = world.iterator(&.{.pos});
    try std.testing.expectEqual(@as(f32, 1), iter.next().?[1].pos.x);
}
