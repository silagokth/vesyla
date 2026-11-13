#ifndef VESYLA_CONVERSION_FUNC_TO_PASM_PASSES
#define VESYLA_CONVERSION_FUNC_TO_PASM_PASSES

include "mlir/Pass/PassBase.td"

def FuncToEpochPass : Pass<"func-to-epoch-pass", "::mlir::ModuleOp"> {
  let summary = "Turn a C kernel function into the epoch form vesyla compiles.";
  let description = [{
    The C front end (Polygeist `cgeist`) emits a kernel as a `func.func` whose
    memref arguments are the io buffers. The rest of the pipeline wants what a
    hand-written program has: the buffers as module-level `memref.alloc` ops,
    with `input_buffer` and `output_buffer` named by their `id`, and the body in
    `pasm.epoch` ops directly in the module. A module without a `func.func` is
    left as it is.

    Rewrite:
      - the argument that is only read becomes
        `memref.alloc() {id = @input_buffer}`, the one that is written
        `memref.alloc() {id = @output_buffer}`. A dynamic dimension (a C array
        parameter decays, so cgeist emits `memref<?x16xi16>`) becomes 65536,
        the whole io address space (IO_ADDR_WIDTH = 16); nothing reads the
        size, only the rank and the access maps;
      - each `memref.alloca` / `memref.alloc` of the function body becomes a
        module-level `memref.alloc`, keeping its `id` if it has one;
        `memref.dealloc` is dropped;
      - the body moves into the module, wrapped in one `pasm.epoch` "rb0"
        unless it already has epochs; the `return` is dropped.

    Preconditions (an error stops the compile):
      1. the module has at most one `func.func`; with none, the pass does
         nothing;
      2. the function has a body, and its `return` carries no values;
      3. every argument is a memref, used only by affine loads and stores --
         any other use means cgeist could not raise the code to affine;
      4. exactly one argument is only read and exactly one is written;
      5. every `memref.alloca` / `memref.alloc` sits directly in the function
         body, not inside a loop.

    Postconditions (checked after the rewrite; a failure is a bug here):
      1. no `func.func` or `func.return` is left;
      2. there is exactly one `memref.alloc {id = @input_buffer}` and one
         `memref.alloc {id = @output_buffer}`;
      3. no `memref.alloca` or `memref.dealloc` is left;
      4. the module body holds only `memref.alloc` and `pasm.epoch` ops.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::memref::MemRefDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_CONVERSION_FUNC_TO_PASM_PASSES
