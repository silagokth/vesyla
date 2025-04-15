#ifndef VESYLA_IROPT_PASM_OPS
#define VESYLA_IROPT_PASM_OPS

include "pasm/Dialect.td"
include "pasm/Types.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

class Pasm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pasm_Dialect, mnemonic, traits>;

def MakeInstrOp : Pasm_Op<"make_instr"> {
  let arguments = (ins StrAttr:$id, StrAttr:$type, DictionaryAttr:$param);
  let assemblyFormat = [{
    `<` attr-dict `>`
  }];
}

def MakeEpochOp : Pasm_Op<"make_epoch"> {
  let summary = "Make epoch";
  let description = [{
    Make epoch
  }];

  let arguments = (ins StrAttr:$id);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def MakeRopOp : Pasm_Op<"make_rop"> {
  let summary = "Make resource operation";
  let description = [{
    Make resource operation.
  }];

  let arguments = (ins StrAttr:$id, I32Attr:$row, I32Attr:$col,
                   I32Attr:$port, I32Attr:$slot);
  let results = (outs RopType: $result);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def MakeCopOp : Pasm_Op<"make_cop"> {
  let summary = "Make control operation";
  let description = [{
    Make control operation.
  }];
  let arguments = (ins StrAttr:$id, I32Attr:$row, I32Attr:$col);
  let results = (outs CopType: $result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def MakeRawOp : Pasm_Op<"make_raw"> {
  let summary = "Make raw operation";
  let description = [{
    Make raw operation.
  }];
  let arguments = (ins StrAttr:$id, I32Attr:$row, I32Attr:$col);
  let results = (outs RawType: $result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def MakeBinaryConstraintOp : Pasm_Op<"make_binary_constraint"> {
  let summary = "Make constraint";
  let description = [{
    Make constraint.
  }];

  let arguments = (ins StrAttr:$type,
                   DictionaryAttr:$param, AnchorType:$anchor1, AnchorType:$anchor2);

  let assemblyFormat = [{
    `<` attr-dict `>` `:` `(` $anchor1 `,` $anchor2 `)`
  }];
}


def MakeAnchorOp : Pasm_Op<"make_anchor"> {
  let summary = "Make constraint";
  let description = [{
    Make constraint.
  }];

  let arguments = (ins OprType:$op, I32Attr:$event, ArrayAttr:$slice);
  let results = (outs AnchorType:$result);

  let assemblyFormat = [{
    `<` attr-dict `>` `(` $op `:` type($op) `)`
  }];
}


def YieldOp : Pasm_Op<"yield", [Terminator,
    ParentOneOf<["MakeEpochOp", "MakeRopOp", "MakeCopOp", "MakeRawOp" ]>]> {
  let summary = "Represents the default branching behaviour of a region";
  let description = [{
    The `cir.yield` operation terminates regions on different CIR operations,
    and it is used to represent the default branching behaviour of a region.
    Said branching behaviour is determinted by the parent operation. For
    example, a yield in a `switch-case` region implies a fallthrough, while
    a yield in a `cir.if` region implies a branch to the exit block, and so
    on.

    In some cases, it might yield an SSA value and the semantics of how the
    values are yielded is defined by the parent operation. For example, a
    `cir.ternary` operation yields a value from one of its regions.

    As a general rule, `cir.yield` must be explicitly used whenever a region has
    more than one block and no terminator, or within `cir.switch` regions not
    `cir.return` terminated.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

#endif // VESYLA_IROPT_PASM_OPS
