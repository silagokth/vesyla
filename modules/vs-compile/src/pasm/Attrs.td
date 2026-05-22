#ifndef VESYLA_PASM_ATTRS
#define VESYLA_PASM_ATTRS

include "pasm/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class Pasm_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Pasm_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ResourceAttr : Pasm_Attr<"Resource", "resource"> {
  let summary = "A (row, col, slot, port) location in the architecture";
  let description = [{
    Identifies a resource endpoint by its position on the architecture grid
    (row, col) and its slot/port within that tile.
  }];
  let parameters = (ins
    "int32_t":$row,
    "int32_t":$col,
    "int32_t":$slot,
    "int32_t":$port);
  let assemblyFormat = "`<` struct(params) `>`";
}

def ResourceArrayAttr : TypedArrayAttrBase<ResourceAttr,
    "array of ResourceAttr"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def AnchorRangeAttr : Pasm_Attr<"AnchorRange", "anchor_range"> {
  let summary = "Symbolic instr, event, and per-dimension index range";
  let description = [{
    Bundles a constraint endpoint's instruction symbol, event name, and
    a multi-dimensional index range. The `instr` field is required; the
    rest default to empty. Printed as
    `#pasm.anchor_range<@instr[, "event"[, [lo0:hi0][lo1:hi1]...]]>`.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$instr,
    StringRefParameter<>:$event,
    ArrayRefParameter<"uint32_t">:$idx_lo,
    ArrayRefParameter<"uint32_t">:$idx_hi);
  let hasCustomAssemblyFormat = 1;
}

def AnchorAttr : Pasm_Attr<"Anchor", "anchor"> {
  let summary = "Single anchor (instr + optional event/indices + delay) for an icdep";
  let description = [{
    Bundles an icdep anchor's instruction symbol, optional event name, optional
    multi-dimensional indices, and delay. Printed as
    `#pasm.anchor<@instr[, "event", [idx0, idx1, ...]][, delay]>`.
    Empty event and zero delay are omitted on print.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$instr,
    StringRefParameter<>:$event,
    ArrayRefParameter<"int32_t">:$idx,
    "int32_t":$delay);
  let hasCustomAssemblyFormat = 1;
}

def DelayAttr : Pasm_Attr<"Delay", "delay"> {
  let summary = "Optional [min, max] delay pair";
  let description = [{
    Holds an optional min and an optional max delay bound. Either bound may
    be absent. Printed as `[<min>, <max>]` with an empty slot for any
    missing bound, e.g. `[, 3]`, `[0, ]`, `[, ]`.
  }];
  let parameters = (ins
    "std::optional<int32_t>":$min,
    "std::optional<int32_t>":$max);
  let hasCustomAssemblyFormat = 1;
}

#endif // VESYLA_PASM_ATTRS
