#ifndef VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES
#define VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES

include "mlir/Pass/PassBase.td"

def AssignIdsPass : Pass<"assign-ids-pass", "::mlir::ModuleOp"> {
  let summary = "Give every unnamed buffer, access and arith op an id.";
  let description = [{
    The passes below key everything on the `id` symbol of an op: instruction
    selection names the storage of an access after its `memref.alloc`, and
    copies the id of the op a match is rooted on onto the `drra.rop` it makes,
    which design space exploration, constraint generation and the pasm lowering
    then look up. A hand-written program names them by hand; a kernel lowered
    from C has none.

    Every `memref.alloc`, `affine.load` / `affine.store` /
    `affine.vector_load` / `affine.vector_store` and every arith op (whatever a
    pattern may be rooted on, now or once the library grows) that has no `id`
    gets a fresh one, `@<op name>_<n>`, numbered per op name in program order:
    `@alloc_0`, `@load_0`, `@load_1`, `@muli_0`. A name some op already uses
    is skipped. Ids already there are kept, so a hand-written program is left
    as it is. An id on an op that never becomes a `drra.rop` (an index
    constant, an `arith.extsi`) is never read.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES
