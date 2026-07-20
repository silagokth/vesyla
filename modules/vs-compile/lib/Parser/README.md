# `src/schedule/` — pipeline driver, PASM front-end, code generator

This directory contains the `vesyla::schedule` namespace: the glue that
`main.cpp` hands over to. It is responsible for three things, in order:

1. **Parsing** a `.pasm` text file into an MLIR `ModuleOp` that uses the
   `pasm` dialect (Flex/Bison front-end).
2. **Running the fixed pass pipeline** that lowers and schedules the IR.
3. **Generating the final outputs** — `instr.asm` and `instr.bin`.

The companion docs are
[`../../README.md`](../../README.md) (top-level overview of the executable,
its CLI, env, build) and
[`../exec/vs-compile/README.md`](../exec/vs-compile/README.md) (deep per-pass
internals, including the timing-model solver).

---

## Pass pipeline

`Scheduler::run` builds an `MLIRContext`, registers the `pasm` dialect, parses
the `.pasm` file into an empty `ModuleOp`, then runs the following passes.
The IR after each pass is dumped to `<output_dir>/debug/schedule/<n>.mlir`.

| # | Pass | Purpose |
|---|---|---|
| 0 | — | Initial parsed IR snapshot |
| 1 | `AddSlotPortPass` | Propagate `(slot, port)` from `RopOp` into each child `InstrOp`'s `param` dictionary |
| 2 | `AddDefaultValuePass` | Fill in missing instruction segments from the ISA JSON (per-component defaults) |
| 3 | `ScheduleEpochPass` | Build a timing model per `EpochOp`, solve it with MiniZinc/CP-SAT, rewrite the IR with concrete cycles plus generated `ACT` and `WAIT` instructions |
| 4 | `ReplaceLoopOp` | Replace each `LoopOp` with an equivalent `EpochOp` so downstream passes only deal with one container kind |
| 5 | `MergeRawOp` | Merge per-cell `RawOp`s across epochs into a single `EpochOp` containing exactly one `RawOp` per `(row, col)` cell |
| 6 | `AddHaltPass` | Append a `halt` `InstrOp` to every per-cell program |

Pass implementations live under `src/pasm/`. Each is implemented as an MLIR
`OpRewritePattern` driven by `applyPatternsGreedily`. If any pass fails,
`Scheduler::run` logs `LOG_FATAL` and exits.

## Code generation

After the pipeline, `vesyla::schedule::Generator::generate` walks the final
module and writes:

- `<output_dir>/instr.asm` — human-readable assembly,
- `<output_dir>/instr.bin` — packed binary instructions ready for the
  hardware sequencer.

## Output layout

```
<output_dir>/
├── instr.asm                     final assembly
├── instr.bin                     final binary
└── debug/
    └── schedule/
        ├── 0.mlir                after parsing
        ├── 1.mlir                after AddSlotPortPass
        ├── 2.mlir                after AddDefaultValuePass
        ├── 3.mlir                after ScheduleEpochPass
        ├── 4.mlir                after ReplaceLoopOp
        ├── 5.mlir                after MergeRawOp
        └── 6.mlir                after AddHaltPass
```

The `debug/schedule/*.mlir` snapshots are the primary tool for diagnosing
miscompiles — each shows the IR between two specific passes.

---

## Files in this directory

### Pipeline driver

#### `Scheduler.{hpp,cpp}`
The top-level orchestrator. Defines `vesyla::schedule::Scheduler` with two
overloads of `run`:

- `run(std::string pasm_file, std::string output_dir, bool allow_unsafe)` —
  parse the file first, then schedule it.
- `run(mlir::ModuleOp &module, std::string output_dir, bool allow_unsafe)` —
  schedule an already-parsed module.

The second overload is the heart of the file: it reads
`VESYLA_SUITE_PATH_COMPONENTS` from the environment (aborts if unset),
creates `<output_dir>/debug/schedule/`, and runs the six passes one at a time
through an MLIR `PassManager`, calling `save_mlir` between each. After the
last pass it calls `Generator::generate` to emit the asm + bin files.

`save_mlir` is a private helper that dumps a module to a file via
`module.print(ofs)` and `LOG_FATAL`s on I/O failure.

### PASM front-end

#### `pasm.l`
The Flex lexer. Tokenises PASM input: integer literals (decimal, binary,
hex, octal, with optional sign), string literals, and the keyword set
(`epoch`, `rop`, `cop`, `raw`, `for`, `if`, `else`, `cstr`, …). Tracks the
current source line in `vesyla::schedule::global_source_line` so that error
messages can point at the right line. Compiled at build time by
`flex_target(PasmLexer ...)` in `CMakeLists.txt` to
`${CMAKE_BINARY_DIR}/PasmLexer.cpp`.

#### `pasm.y`
The Bison grammar. Defines the PASM concrete syntax and emits MLIR ops
directly from the reduce actions — building `EpochOp`, `RopOp`, `CopOp`,
`RawOp`, `LoopOp`, `InstrOp`, `CstrOp`, etc. into the module passed in by
`Parser`. Compiled at build time by `bison_target(PasmParser ...)` to
`${CMAKE_BINARY_DIR}/PasmParser.cpp` (and a header `PasmParser.hpp`).

#### `Parser.{hpp,cpp}`
Thin C++ wrapper that owns the file handle and the temporary state needed
by the Bison reduce actions. `Parser::parse(filename, module)` does:

1. `fopen` the input file and assign it to `yyin`.
2. Stash the filename in `global_input_file_name` (used by error messages).
3. Create a temporary "holder" `EpochOp` named `__temp__` at the end of the
   module and expose it to the Bison actions via the global `temp_epoch_op`
   — this gives the grammar a known insertion point for ops parsed at the
   top level.
4. Call `yyparse()`.
5. Erase the temp epoch op and close the file.

The two globals (`module`, `temp_epoch_op`) are declared `extern` here and
defined inside the generated `pasm.y` translation unit — the grammar
actions read and append to them.

### Front-end support utilities

#### `BisonUtil.{hpp,cpp}`
Shared helpers used from inside Bison reduce actions:

- `print_error(const char *message)` — formats a parse error with the
  current `global_source_line` and `yylineno`.
- `print_grammar(const std::string &grammar, bool printLineNum)` — debug
  log of which grammar rule fired (used to trace parses).
- Re-declares the `module` and `temp_epoch_op` externs alongside the MLIR
  builder includes the grammar needs.

#### `FlexUtil.{hpp,cpp}`
A single helper, `print_lex_token(const string &token)`, used from Flex
rules to log every matched token at debug level. Useful when diagnosing
mis-tokenisation.

#### `GlobalUtil.{hpp,cpp}`
Holds the two globals shared across the lexer, parser, and error helpers:

- `char global_input_file_name[80]` — the path being parsed (set by
  `Parser::parse`).
- `unsigned int global_source_line` — incremented by the lexer on every
  newline.

These are intentionally plain globals because Flex/Bison need C-callable
state.

### Code generator

#### `Generator.{hpp,cpp}`
The final emission stage. `Generator::generate(module, output_dir, basename)`
calls two private helpers in order:

- `gen_asm(...)` — writes `<basename>.asm`. Walks the module: each top-level
  `EpochOp` → each `RawOp` (one per `(row, col)` cell) → each `InstrOp`,
  printing a human-readable line per instruction with its segments.
- `gen_bin(...)` — writes `<basename>.bin`. Same walk, but each instruction
  is encoded into the binary layout dictated by the ISA JSON (using the
  `int2bin` helper at the top of the file to render integer segments as
  fixed-width bit strings).

The `Generator` is stateless — instances are constructed on the stack and
discarded after one `generate` call.
