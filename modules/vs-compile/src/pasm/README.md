# PASM Dialect

An MLIR dialect to represent the control and data flow of a program with explicit indexing information.

## The `.td` files (TableGen definitions)

These are the "schema" of the dialect. MLIR's TableGen tool auto-generates C++ boilerplate from them.

| File | Purpose |
|---|---|
| **`Dialect.td`** | Declares the `pasm` dialect itself (name, C++ namespace `vesyla::pasm`) |
| **`Types.td`** | Defines custom types. Currently just a placeholder `DummyType` |
| **`Ops.td`** | Defines all the **operations** (the IR nodes). This is the most important file |
| **`Passes.td`** | Declares the **compiler passes** that transform pasm IR |

### The operations in `Ops.td`

These represent the instruction set of the pasm IR:

- **`InstrOp`** — a generic instruction with an id, type, and parameter dictionary
- **`EpochOp`** — a time-bounded region (contains a body of other ops)
- **`LoopOp`** — a for-loop with an iteration count, containing a body
- **`CondOp`** — an if/else with `then` and `else` regions
- **`RopOp`** — a **resource operation**, positioned at a specific (row, col, slot, port) in the architecture
- **`CopOp`** — a **control operation**, positioned at (row, col)
- **`RawOp`** — a **raw operation**, also at (row, col), with a body
- **`CstrOp`** — a **constraint** (type + expression string)
- **`YieldOp`** — a terminator that ends region bodies (required by MLIR)

## The `.hpp`/`.cpp` pairs

| File pair | What it does |
|---|---|
| **`Dialect.cpp/hpp`** | Initializes the dialect (registers ops and types). Includes the TableGen-generated `Dialect.cpp.inc` |
| **`Ops.cpp/hpp`** | Any custom C++ logic for operations beyond what TableGen generates |
| **`Types.cpp/hpp`** | Same for types |
| **`Config.cpp/hpp`** | A singleton holding architecture and ISA JSON configs (`arch_json`, `isa_json`) |

## The passes

Each pass has a `.cpp/.hpp` pair:

| Pass | What it does |
|---|---|
| **`ScheduleEpochPass`** | The main scheduler — assigns timing to epoch regions using a constraint solver. Uses `component_path` and `tmp_path` options |
| **`Passes.cpp` (ReplaceLoopOp)** | Unrolls `LoopOp` into `EpochOp` (replaces loops with flat epoch regions) |
| **`MergeRawOp`** | Merges multiple `RawOp`s into one for optimization |
| **`AddHaltPass`** | Inserts halt instructions to terminate execution |
| **`AddSlotPortPass`** | Fills in slot/port information on operations |
| **`AddDefaultValuePass`** | Populates default parameter values |

## Reading order suggestion

1. **`Dialect.td`** → **`Ops.td`** → **`Types.td`** — understand what the IR looks like
2. **`Passes.td`** — see what transformations exist
3. **`Config.hpp`** — understand the external inputs (arch/ISA JSONs)
4. The simpler passes (`AddHaltPass`, `AddSlotPortPass`, `MergeRawOp`) — see how the IR is manipulated
5. **`ScheduleEpochPass`** last — it's the core scheduling algorithm and by far the most complex

## Overall flow

Higher-level IR gets lowered into pasm ops (instructions, epochs, resource/control ops placed on an architecture grid), then these passes schedule, merge, and finalize them into something that maps onto hardware.
