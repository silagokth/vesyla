#ifndef VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES
#define VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES

include "mlir/Pass/PassBase.td"

def SelectInstructionsPass : Pass<"select-instructions-pass", "::mlir::ModuleOp"> {
  let summary = "Select drra instructions from affine + arith ops.";
  let description = [{
    Instruction selection driven by the component library. Each resource ships a
    functional description of itself as ordinary MLIR --
    `<component-path>/resources/<kind>/<kind>.mlir` -- with one `func.func` per
    ISA mode or port. The function's arguments are the wildcards, its body is
    the shape to match, and attributes on it say which instruction a match
    becomes (`conf` / `evt`, keyed by `isa.json` segment name) and how strongly
    to prefer it (`benefit`).

    Matching walks the pattern body and the program at the same time, one
    operand pair at a time. Patterns are applied highest benefit first, and
    everything at one benefit is tried together; ops covered by an earlier match
    are off limits to later ones, which is what makes it maximal munch. A fused
    `mac` therefore takes the multiply and the add as a unit before the plain
    `add` / `mult` patterns can take them apart.

    This pass currently only *annotates*: a match records `select.kind`,
    `select.pattern`, `select.benefit` and the emission dictionary on the root,
    and `select.covered_by` on the rest of the match. Rewriting into the drra
    dialect comes later. Ops that no pattern claimed are reported as warnings
    rather than failing the pass, so one run shows every gap.

    Adding a resource is a matter of dropping a `.mlir` file into the component
    library -- nothing here has to be recompiled.
  }];

  let options = [
    Option<"componentPath", "component-path", "std::string", /*default=*/"\"\"",
           "Root of the built component library. Defaults to "
           "$VESYLA_SUITE_PATH_COMPONENTS.">,
  ];

  let dependentDialects = ["::vesyla::drra::DrraDialect",
                           "::vesyla::pasm::PasmDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES
