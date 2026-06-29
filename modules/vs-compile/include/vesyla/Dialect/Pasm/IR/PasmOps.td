#ifndef VESYLA_PASM_OPS
#define VESYLA_PASM_OPS

include "vesyla/Dialect/Pasm/IR/PasmDialect.td"
include "vesyla/Dialect/Pasm/IR/PasmTypes.td"
include "vesyla/Dialect/Pasm/IR/PasmAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Pasm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pasm_Dialect, mnemonic, traits>;

def InstrOp : Pasm_Op<"instr"> {
  let arguments = (ins StrAttr:$id, StrAttr:$type, DictionaryAttr:$param);
  let assemblyFormat = [{
    `<` attr-dict `>`
  }];
}

def EpochOp : Pasm_Op<"epoch", [SymbolTable]> {
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

def LoopOp : Pasm_Op<"loop"> {
  let summary = "Make for loop";
  let description = [{
    Make for loop
  }];
  let arguments = (ins StrAttr:$id, I32Attr:$iter);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def CondOp : Pasm_Op<"cond"> {
  let summary = "Make if condition";
  let description = [{
    Make if condition
  }];
  let arguments = (ins StrAttr:$id, StrAttr:$comparator, I32Attr:$var1, I32Attr:$var2);
  let regions = (region SizedRegion<1>:$then, 
                 SizedRegion<1>:$else);
  let assemblyFormat = [{
    `<` attr-dict `>`
    `then` $then
    `else` $else
  }];
}

def RopOp : Pasm_Op<"rop"> {
  let summary = "Make resource operation";
  let description = [{
    Make resource operation. May optionally carry an affine map describing
    per-iteration index math; the map is consumed and erased by the
    affine-to-rep lowering pass.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                   I32Attr:$row, I32Attr:$col,
                   I32Attr:$slot, I32Attr:$port,
                   OptionalAttr<AffineMapAttr>:$map);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body
  }];
}

def CopOp : Pasm_Op<"cop"> {
  let summary = "Make control operation";
  let description = [{
    Make control operation.
  }];
  let arguments = (ins StrAttr:$id, I32Attr:$row, I32Attr:$col);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def RawOp : Pasm_Op<"raw"> {
  let summary = "Make raw operation";
  let description = [{
    Make raw operation.
  }];
  let arguments = (ins StrAttr:$id, I32Attr:$row, I32Attr:$col);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body 
  }];
}

def CellOp : Pasm_Op<"cell", [SymbolTable, NoTerminator]> {
  let summary = "Make cell";
  let description = [{
    Make cell located at a given (row, col) position in the architecture.
  }];
  let arguments = (ins I32Attr:$row, I32Attr:$col);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` attr-dict `>` $body
  }];
}

def CstrOp : Pasm_Op<"cstr"> {
  let summary = "Delay constraint between two ROPs";
  let description = [{
    Constrains the delay from src ROP to dst ROP, applied over the
    given index ranges of each, to lie within the optional bounds
    carried by `delay = #pasm.delay<[<min>, <max>]>` (either bound
    may be absent).
  }];

  let arguments = (ins
    AnchorRangeAttr:$src,
    AnchorRangeAttr:$dst,
    DelayAttr:$delay,
    DefaultValuedAttr<BoolAttr, "false">:$is_neq);

  let assemblyFormat = [{
    `<` attr-dict `>`
  }];

  let hasVerifier = 1;
}

def IcDepOp : Pasm_Op<"icdep"> {
  let summary = "Interconnect dependency edge between two resource endpoints";
  let description = [{
    Represents an interconnect dependency edge from a source `ResourceAttr`
    to a target `ResourceAttr`. The edge carries two anchors describing the
    first and last use of the interconnect, each given as
    (instruction, event, indices) together with a delay value.
  }];

  let arguments = (ins
    ResourceAttr:$src,
    ResourceArrayAttr:$dst,
    StrAttr:$kind,
    AnchorAttr:$first,
    AnchorAttr:$last,
    OptionalAttr<StrAttr>:$dir);

  let assemblyFormat = [{
    `<` attr-dict `>`
  }];

  let hasVerifier = 1;
}



def YieldOp : Pasm_Op<"yield", [Terminator,
    ParentOneOf<["LoopOp", "CondOp", "EpochOp", "RopOp", "CopOp", "RawOp" ]>]> {
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

#endif // VESYLA_PASM_OPS
