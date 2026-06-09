# `vs-compile` — pass internals (Generated using Claude Code)

This document is the deep companion to
[`modules/vs-compile/README.md`](../../../README.md), which is the high-level
overview of the executable, its CLI, environment, output layout, and build.

What follows here is the per-pass technical detail: how each pass in the
pipeline rewrites the IR, and — for `ScheduleEpochPass` — the full chain from
the textual timing-model expression language down to the MiniZinc model and
the decoded solver result.

The top-level overview lists the six passes in order
(`AddSlotPortPass` → `AddDefaultValuePass` → `ScheduleEpochPass` →
`ReplaceLoopOp` → `MergeRawOp` → `AddHaltPass`); each section below assumes
that ordering and only documents what the pass *does*.

Every pass is implemented as an MLIR `OpRewritePattern` driven by
`applyPatternsGreedily`, so the framework keeps re-applying the patterns until
a fixed point is reached. Each pass corresponds to one source file under
`src/pasm/`.

## 1. `AddSlotPortPass` — propagate position into instruction params
*File: `src/pasm/AddSlotPortPass.cpp`. Pattern matches: `RopOp`.*

A `RopOp` (resource op) carries `(row, col, slot, port)` as MLIR attributes. The contained `InstrOp`s have a free-form `param` dictionary, and downstream code expects `slot` and `port` to be present in that dictionary on every instruction. The pass:

1. Walks each `InstrOp` inside the matched `RopOp`'s body region.
2. Looks at the `param` dictionary attribute:
   - If `slot` is already there but disagrees with the parent's `slot`, prints a warning and `exit(EXIT_FAILURE)` — the IR is inconsistent.
   - If `slot` is missing, adds it as an `i32` with the parent's value. Same for `port`.
3. Replaces `param` with the updated `DictionaryAttr` if anything changed.

Effect: by the end, every `InstrOp` inside a `RopOp` has `slot` and `port` populated, matching the enclosing op.

## 2. `AddDefaultValuePass` — fill in ISA-defined defaults
*File: `src/pasm/AddDefaultValuePass.cpp`. Pattern matches: `InstrOp`.*

ISA instructions consist of named segments (bit-fields). The PASM input only mentions the segments the programmer wrote; the rest must default to whatever the ISA JSON says. The pass:

1. Walks up from the `InstrOp` to its parent, which must be a `RopOp` / `CopOp` / `RawOp`.
2. Builds a position label from the parent — `"row_col_slot_port"` for `RopOp`, `"row_col"` for `CopOp` / `RawOp`.
3. Looks up `component_map_json[label]` to find the **resource kind** for that cell (the kind of component instantiated there).
4. Walks `isa_json["components"][*]` to find the entry whose `kind` matches, then iterates its `instructions` to find the one whose `name` matches the `InstrOp`'s type.
5. For every `segment` in that ISA instruction that the `InstrOp` does **not** already declare, appends a new attribute with the segment's `default_val` (or `0` if none specified).

Effect: every `InstrOp` ends up with the full set of segments expected by the ISA, ready for the scheduler and code generator.

## 3. `ScheduleEpochPass` — the actual scheduler
*File: `src/pasm/ScheduleEpochPass.cpp`. Pattern matches: `EpochOp`.*

This is the heavy one. An `EpochOp` is a time-bounded region; the pass turns it from a *set of resource/control ops with constraints* into a *concretely scheduled program* of activation and wait instructions.

For each `EpochOp`:

1. **Build a timing model** (`tm::TimingModel`).
   - For every child `RopOp`: serialise it to JSON, write the JSON to a temp file under `tmp_path`, then `system()`-execute the per-component helper

     ```
     <component_path>/resources/<resource_kind>/compile_util get_timing_model <in.json> <out.txt>
     ```

     The helper returns a textual timing-model expression for that op. The expression is added to the model as a `tm::Operation`.
   - For every child `CstrOp`: forward `(type, expr)` to the model via `tm::Constraint`.
   - `RawOp` children cause the pattern to bail out (raw ops belong in a different lowering path; mixing them with rop/cop is rejected).
2. **Add built-in conflict constraints.** For each cell `(row, col)` that hosts more than one resource op, add `linear` constraints saying "this rop's anchor `!=` every control-op anchor in the same cell" — i.e. ops sharing a cell cannot fire on the same cycle.
3. **Solve** the model with `tm::Solver::solve(model)` (the solver is configured with `tmp_path` for its own scratch files). The result is a map from variable name to value:
   - Numeric values are decoded into a `schedule_table` (op-id → cycle).
   - Special keys `use_act_mode_0` / `use_act_mode_1` toggle which ACT-instruction encoding is used (see below).
   - Vector values (starting with `[`) are also handled.
   - If `result.empty()` the pass aborts the whole compile with "No solution found".
4. **Generate ACT instructions.** ACT instructions tell the hardware which (slot, port) combinations to activate on a given cycle. Three encodings are tried in order, picking the first that fits:
   - **Mode 0** — all selected ports lie within four consecutive slots; encodes `min_slot` and a 16-bit port-mask.
   - **Mode 1** — every active slot has the same port pattern; encodes a single 4-bit port mask + a 16-bit slot mask.
   - **Mode 2** — fully general 64-bit port-vector (one bit per `slot * 4 + port`).
   If none fit, the compile aborts. WAIT instructions are also generated for explicit cycle gaps.
5. **Materialise the schedule** in three steps inside the IR:
   - `replace_time_in_instr_param` — write the solved cycle counts back into the relevant `InstrOp` `param` dictionaries.
   - `reshape_instr` — rebuild the `EpochOp` body, replacing the original `RopOp`s/`CopOp`s with the time-ordered sequence of ACT/WAIT and concrete instructions.
   - `synchronize` — insert any cross-cell synchronisation needed.

Pass options (set in `Scheduler.cpp` when adding it to the `PassManager`):

- `component_path` — comes from `VESYLA_SUITE_PATH_COMPONENTS`, used to find the per-component `compile_util` helper.
- `tmp_path` — `vesyla::util::SysPath::temp_dir()`, scratch space for JSON exchange and the solver.
- `allow_unsafe` — relaxes safety checks (forwarded to the solver).

Effect: each `EpochOp` is rewritten so that its body is a concrete, scheduled instruction sequence per cell rather than abstract resource/control ops with constraints.

### 3a. The timing-model expression language
*Files: `src/tm/Operation.{cpp,hpp}`.*

Each `RopOp` is described by a textual expression in three primitives. `tm::OperationExpr` is the parsed AST; `tm::Operation` wraps it with a `name` and the cell coordinates `(row, col, slot, port)` set by the scheduler.

| Kind | Syntax | Meaning |
|---|---|---|
| **EVENT** | `e<id>` (e.g. `e1`) | A single 1-cycle marker the user can refer to as `<op>.e<id>` |
| **TRANSIT** | `T<delay>(left, right)` | Run `left`, wait `delay` cycles, then run `right` |
| **REPEAT** | `R<iter, delay>(child)` | Run `child` `iter` times with `delay` cycles between iterations |

`delay` and `iter` can be either integer literals or identifiers; identifiers automatically become MiniZinc decision variables (`compile()` collects them in `TimingModel::variables`). An entire operation is parsed from text like:

```
operation op0 T<3>(e1, e2)
```

Per-component helper. The string above is **produced** by an external program — `<component_path>/resources/<resource_kind>/compile_util get_timing_model <input.json> <output.txt>`. The `RopOp` (with id, row/col/slot/port and its child `InstrOp`s) is serialised to JSON via `op2json`, written to `<input.json>`, and the helper writes the timing-model string to `<output.txt>`. Different components (e.g. memory vs. arithmetic units) thus describe their own latencies/event structure outside the compiler.

Anchors. References from constraints take the form `op_name.e<id>[i][j]…`. Each `[k]` index corresponds to one enclosing `R<...>` repeat — `op0.e1[2]` means "the e1 event of op0 in iteration 2 of its outermost repeat". `Operation::get_all_anchors()` enumerates these names.

### 3b. `TimingModel::compile()`

`compile()` runs once on the model before MiniZinc is invoked. It does three things:

1. **Variable extraction.** Walk every `OperationExpr`. For each TRANSIT/REPEAT node, classify the `delay` parameter: if it's a numeric literal, ignore it; if it's an identifier, add it to `TimingModel::variables` (these will become `var 0..MAX_LATENCY` in MiniZinc — i.e. delays the solver is free to choose).
2. **Anchor lowering.** For each user `Constraint` (kind must be `"linear"`), regex-find every substring matching `<op>.e<id>[<idx>]…`, register a `tm::Anchor` for it, and **rewrite the constraint expression in place** so the `op.eN[…]` token is replaced by the synthetic anchor variable name. After this step the constraint expression is plain MiniZinc.
3. **Symbolic start/duration computation per operation.** Each `OperationExpr` is converted to a binary tree (TRANSIT → two children, REPEAT → one child, EVENT → leaf), then traversed twice:

   **Post-order (LRC) — duration up the tree:**

   | Node kind | Duration string |
   |---|---|
   | EVENT | `(1)` |
   | TRANSIT | `(left_dur + right_dur + delay)` |
   | REPEAT | `(left_dur*iter + delay*(iter-1))` |

   **Pre-order (CLR) — start time down the tree:**
   - root → `start = (0)`,
   - left child inherits parent's start,
   - right child → `start = (parent.start + left.duration + delay)`.

   Then for every anchor that lives in this operation, walk from its EVENT leaf up to the root collecting REPEAT ancestors (innermost-first, then reversed so outermost becomes index `0`). The anchor's absolute timing expression becomes:

   ```
   timing_expr = op_name + event.start + Σ_i (R_i.left.duration + R_i.delay) * indices[i]
   ```

   Range-checked at compile time: each `indices[i]` must be `< iter` of the corresponding repeat, and the number of indices must not exceed the depth of repeat nesting.

   Finally `op.duration_expr = root.duration` is the symbolic length of the whole operation.

After `compile()`, the model has: a `variables` set, a fully-populated `anchors` map (each with a numeric `timing_expr`), and a `duration_expr` per operation — all as MiniZinc-ready strings.

### 3c. `TimingModel::to_mzn()` — model emission

`to_mzn(mzn_stream, dzn_stream, allow_act_mode_2)` writes a MiniZinc model and a matching data file:

**DZN — fixed per-op data**

```minizinc
op_cell_row = array1d(0..NUM_OPS-1, [...]);
op_cell_col = array1d(0..NUM_OPS-1, [...]);
op_slot     = array1d(0..NUM_OPS-1, [...]);
op_port     = array1d(0..NUM_OPS-1, [...]);
```

(Note: `op_slot` and `op_port` are typed as `var` in the model, so the values in the DZN constrain them to specific decision values.)

**MZN — constants**: `MAX_LATENCY = 10_000_000`, `MAX_SLOTS = 16`, `NUM_PORTS_PER_RESOURCE = 4`, `NUM_OPS = <count>`.

**MZN — decision variables**:
- `var 0..MAX_LATENCY: total_latency;`
- per-op start/end vectors `op_start_vec[i]`, `op_end_vec[i]`,
- per-op `var: <op_name>` (the start time of the op — referenced by name in anchor expressions),
- one `var 0..MAX_LATENCY` per identifier collected in `variables` (free delays),
- one `var 0..MAX_LATENCY` per anchor.

**MZN — constraints**:
- Bounding box: `min(op_start_vec) == 0` and `max(op_end_vec) == total_latency`.
- For each operation: `op_start_vec[idx] == <name>` and `op_end_vec[idx] == <name> + <duration_expr>`.
- For each anchor: `<anchor_name> == <timing_expr>`.
- For each user constraint: `constraint <expr>;` verbatim (the `compile()` step already substituted anchor names into it).

**MZN — ACT-mode booleans**: `var bool: use_act_mode_0;`, `var bool: use_act_mode_1;`. With `allow_act_mode_2 == false` the solver must pick exactly one (`= 1`); with `true` it may pick neither (`<= 1`), which is the only way mode 2 is allowed. Each boolean gates a big quantified `forall` over all op pairs in the same cell:

- **Mode 0**: any two ops in the same cell whose `op_slot`s differ by ≥ 4 must have **different** start times.
- **Mode 1**: any two ops in the same cell that *do* start at the same time and use different slots must have **identical** port-usage patterns across all `NUM_PORTS_PER_RESOURCE` ports.

These directly mirror the encoding capacity of the corresponding ACT instruction modes (see `create_act_0_instr` / `create_act_1_instr` in `ScheduleEpochPass.cpp`).

**Objective**: `solve minimize total_latency;`.

### 3d. `Solver::solve()` — invocation and result decoding
*File: `src/tm/Solver.cpp`.*

1. Generate three random temp filenames in `tmp_path`: `<rand>.mzn`, `<rand>.dzn`, `<rand>.json`.
2. Call `tm.to_mzn(...)` to populate the first two files.
3. `system("minizinc --json-stream --solver cp-sat <mzn> <dzn> > <json>")`. The CP-SAT (Google OR-Tools) solver is what actually finds a schedule.
4. The MiniZinc `--json-stream` output is multiple JSON objects concatenated, not a JSON array. `turn_to_valid_json` rewrites it: trim, wrap in `[...]`, and replace each `}<whitespace>{` with `}, {`. Then parse.
5. `check_mzn_solution_valid` walks the resulting array. Any entry with `type == "solution"` causes the helper to extract the `output.dzn` text as the solution string. Any entry with `status == "UNSATISFIABLE"` means no solution.
6. **First-attempt vs. retry**. The first call passes `allow_act_mode_2 = false` (modes 0 and 1 only — these are the cheap encodings). If the solver returns UNSAT, the model is regenerated with `allow_act_mode_2 = true` and re-solved — only then is the general 64-bit port-vector (mode 2) admissible. Failing the second attempt is a fatal error.
7. The solution string is a list of `name = value;` lines (DZN syntax). The solver splits on `\n`, then on `=`, trims whitespace, drops the trailing `;`, and returns an `unordered_map<string,string>`.

Keys you'll see in the result map:
- `<op_name>` — solved start time (cycle) for each operation.
- `<anchor_name>` — solved cycle for each anchor referenced in a constraint.
- `op_slot[..]`, `op_port[..]` — final slot/port assignment per op (as `[a,b,c,...]` arrays — these are the keys with values starting with `[`).
- `total_latency` — overall epoch length.
- `use_act_mode_0`, `use_act_mode_1` — the chosen ACT encoding (`true`/`false`).
- Plus any free delay identifier from `variables`.

Back in `ScheduleEpochPass::matchAndRewrite`, the per-op start times are extracted into `schedule_table : op_id → cycle`, the `use_act_mode_*` flags select the encoder (`create_act_0_instr` / `create_act_1_instr` / `create_act_2_instr`), and the IR is rewritten as described in step 5 of §3 above.

### 3e. Where the helper comes from, and a worked example

The per-component `compile_util` is **not** hand-written compiler code — it's a small Rust binary, one per component, generated from a Jinja template in the `vs-component` module:

- Template: `modules/vs-component/template/compile_util/src/main.rs.jinja`.
- Stamping: `vs-component`'s `ComponentGenerator::copy_and_render_files` renders the template against the component's `arch.json` + `isa.json`. For each ISA instruction it emits a `match` arm in `get_timing_model(op: Op) -> String`. The template leaves the body as `todo!(...)` — the component author fills in the arms with the real per-instruction latency / event structure.
- Output layout per component: `compile_util/Cargo.toml`, `compile_util/src/main.rs`. Once built (`cargo build`), the binary is invoked by `ScheduleEpochPass` as:

  ```
  <VESYLA_SUITE_PATH_COMPONENTS>/resources/<resource_kind>/compile_util <subcommand> <in.json> <out.txt>
  ```

  Two subcommands exist: `get_timing_model` (used by `ScheduleEpochPass`) and `reshape_instr` (used later by the same pass during code emission to rewrite per-instruction fields).

The template's algorithm for `get_timing_model` is straightforward:

1. Start with `expr = "e0"` (one event marker).
2. Walk every `Instr` in the op body. The author's per-instruction code populates two maps:
   - `t : i64 → String` — at index `i`, a transit delay that follows event `e<i-1>`.
   - `r : i64 → (String, String)` — at index `i`, a `(iter, delay)` pair for a repeat enclosing the current expression.
3. After the walk, fold `t` into the expression by sorting keys and wrapping:
   `expr = T<delay>(expr, e<event_counter>)` for each non-zero entry.
4. Then fold `r` similarly: `expr = R<iter, delay>(expr)` for each entry, outermost last.
5. Return the final expression string.

#### Example: a DPU `add` instruction

Imagine a hypothetical "DPU" component placed at `(row=0, col=1, slot=2, port=0)` whose ISA defines an `add` instruction with segments `iter` and `delay`. Suppose the PASM input contains:

```mlir
pasm.epoch "ep0" {
  pasm.rop "rop0" 0 1 2 0 {
    pasm.instr "i0" "add" {iter = 4 : i32, delay = 2 : i32}
  }
  pasm.cstr "linear" "rop0.e1[0] >= 5"
}
```

**Step 1 — `op2json` writes this temp input file** (`<tmp>/<rand>.json`):

```json
{
  "kind": "rop",
  "id": "rop0",
  "row": 0, "col": 1, "slot": 2, "port": 0,
  "body": [
    {
      "id": "i0",
      "kind": "add",
      "params": [
        {"name": "iter",  "value": "4"},
        {"name": "delay", "value": "2"},
        {"name": "slot",  "value": "2"},
        {"name": "port",  "value": "0"}
      ]
    }
  ]
}
```

**Step 2 — the scheduler runs**:

```bash
$VESYLA_SUITE_PATH_COMPONENTS/resources/dpu/compile_util \
    get_timing_model <tmp>/abc.json <tmp>/abc_out.txt
```

**Step 3 — what the author's `get_timing_model` does for `add`**. The hand-filled match arm reads `iter` and `delay` from `instr.params`, then writes:

```rust
"add" => {
    let iter  = instr_segments.get_value("iter");   // "4"
    let delay = instr_segments.get_value("delay");  // "2"
    // one TRANSIT after e0 with a 1-cycle pipeline latency:
    t.insert(0, "1".to_string());
    // wrapped in a repeat across `iter` iterations with `delay` between them:
    r.insert(0, (iter, delay));
}
```

The template's epilogue then folds `t` and `r` into the final expression:

```
expr := "e0"
        → T<1>(e0, e1)              // from t[0] = "1"
        → R<4, 2>(T<1>(e0, e1))     // from r[0] = ("4", "2")
```

**Step 4 — output file content** (`<tmp>/abc_out.txt`):

```
R<4,2>(T<1>(e0, e1))
```

**Step 5 — back in `ScheduleEpochPass`**:

```cpp
tm::Operation operation =
    tm::Operation(rop_json["id"].get<std::string>(),  // "rop0"
                  output_str);                        // "R<4,2>(T<1>(e0, e1))"
operation.row = 0; operation.col = 1; operation.slot = 2; operation.port = 0;
model.add_operation(operation);
```

The `OperationExpr` parser turns the string into a tree:

```
REPEAT  iter=4, delay=2
└── TRANSIT  delay=1
    ├── EVENT  e0
    └── EVENT  e1
```

Anchors `rop0.e0`, `rop0.e0[0..3]`, `rop0.e1`, `rop0.e1[0..3]` are all referenceable from constraints.

**Step 6 — the user's constraint `rop0.e1[0] >= 5`** is added as a `tm::Constraint("linear", "rop0.e1[0] >= 5")`. During `compile()`:

- Anchor `rop0.e1[0]` is registered with `op_name=rop0`, `event_id=1`, `indices=[0]`.
- Constraint expression rewritten to `<anchor_var> >= 5` (where `<anchor_var>` is a flattened identifier).
- Per-op tree traversal computes (with `iter=4, delay=2, child_dur=(1+1+1)=3`):
  - `event e1`'s `start = (0 + (1) + (1)) = 2` cycles into the inner transit,
  - operation's `duration_expr = (3*4 + 2*3) = 18`,
  - anchor's `timing_expr = (rop0 + 2 + (3 + 2)*0) = (rop0 + 2)` (index 0 contributes nothing).

**Step 7 — what the MiniZinc model looks like (excerpt):**

```minizinc
int: NUM_OPS = 1;
var 0..MAX_LATENCY: total_latency;
array [0..0] of var 0..MAX_LATENCY: op_start_vec;
array [0..0] of var 0..MAX_LATENCY: op_end_vec;
op_cell_row = [0]; op_cell_col = [1];
array [0..0] of var 0..MAX_SLOTS-1: op_slot;
array [0..0] of var 0..NUM_PORTS_PER_RESOURCE-1: op_port;

var 0..MAX_LATENCY: rop0;
constraint op_start_vec[0] == rop0;
constraint op_end_vec[0]   == rop0 + (((1)+(1)+(1))*4 + (2)*(4-1));

var 0..MAX_LATENCY: rop0_e1_0;     % synthesised anchor
constraint rop0_e1_0 == (rop0 + 2);

constraint rop0_e1_0 >= 5;          % the user constraint, rewritten

constraint min(op_start_vec) == 0;
constraint max(op_end_vec)   == total_latency;
solve minimize total_latency;
```

The solver picks `rop0 = 3` (smallest start that satisfies `rop0 + 2 >= 5`), so `total_latency = 3 + 18 = 21`. The result map sent back to `ScheduleEpochPass` therefore contains `rop0 = 3`, `total_latency = 21`, and the chosen ACT mode boolean — and the schedule table records cycle `3` for `rop0`, after which the IR is rewritten as described in step 5 of §3 above.

## 4. `ReplaceLoopOp` — unroll loops into epochs
*File: `src/pasm/Passes.cpp`. Pattern matches: `LoopOp`.*

After scheduling, `LoopOp`s are flattened into `EpochOp`s so that downstream passes only need to deal with one container kind:

1. Get the `LoopOp`'s parent block.
2. Create a new `EpochOp` with id `epoch_derived_from_loop_<original-id>`.
3. Walk every block in the loop body and `rewriter.clone()` each non-terminator op into the new epoch's entry block.
4. `rewriter.replaceOp(op, epoch_op)` — substitute the loop with the new epoch.

Note: the pattern returns `failure()` even after rewriting, because the iteration count (`getIter()`) isn't yet acted on — the pattern reports "no further match" so the greedy driver doesn't loop forever on the freshly-created epoch. The current code handles a single-block loop body and explicitly notes that multi-block bodies / branches need richer logic.

Effect: every `LoopOp` is replaced by a single `EpochOp` whose body holds a clone of the loop body. (Iteration counts aren't materialised here — they'd come from a follow-up unroller.)

## 5. `MergeRawOp` — fuse all RawOp epochs into one
*File: `src/pasm/MergeRawOp.cpp`. Pattern matches: `EpochOp`.*

By this stage every `EpochOp` in the module contains only `RawOp`s (one raw op per `(row, col)` cell). Multiple epochs back-to-back with the same cells means the hardware would receive disjointed instruction streams per cell. The pass merges them:

1. Walk the module-level block; the **first** `EpochOp` is the **target**, the rest are **sources**. Verify every source contains only `RawOp`s.
2. Build a `target_insertion_raw_op_map: "row_col" → RawOp*` from the target's existing raw ops.
3. For each source `EpochOp`, for each `RawOp` it contains:
   - If the same `"row_col"` label exists in the target, clone every `InstrOp` from the source `RawOp` to just before the target `RawOp`'s terminator (preserving order).
   - Otherwise, clone the whole `RawOp` into the target `EpochOp` and remember it in the map.
4. Erase the source `EpochOp`s.

Effect: the module collapses to a single `EpochOp` containing exactly one `RawOp` per cell, with all instructions for that cell concatenated.

## 6. `AddHaltPass` — terminate every per-cell program
*File: `src/pasm/AddHaltPass.cpp`. Pattern matches: `RawOp`.*

Each per-cell instruction stream needs an explicit halt to stop the local sequencer. The pass:

1. Look at the `RawOp`'s body block.
2. The last op is the MLIR `YieldOp` terminator; check the **second-to-last** op — if it's already an `InstrOp` of type `"halt"`, return `failure()` (idempotent).
3. Otherwise, insert a new `InstrOp(type="halt")` with a randomly-named id and an empty `param` dictionary just before the terminator.

Effect: every `RawOp`'s instruction list ends with `halt`, ready for the code generator.
