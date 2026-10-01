#ifndef VESYLA_PASM_ATTRS
#define VESYLA_PASM_ATTRS

include "vesyla/Dialect/Pasm/IR/PasmDialect.td"
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
  let hasCustomAssemblyFormat = 1;
}

def ResourceArrayAttr : TypedArrayAttrBase<ResourceAttr,
    "array of ResourceAttr"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def AnchorRangeAttr : Pasm_Attr<"AnchorRange", "anchor_range"> {
  let summary = "Constraint endpoint as a low/high anchor pair (OR.MT.IR)";
  let description = [{
    A constraint endpoint spanning a range between a low and a high anchor that
    differ only in the innermost IR index. Both anchors share the `instr`
    symbol; each stores its own OR indices, MT event id, and IR indices.
    Printed as `#pasm.anchor_range<@instr[, "<or>.<mt>.<ir[:range]>"]>`, where
    the index text follows the ::vesyla::Anchor grammar with a compact
    `[lo:hi]` range on the innermost IR bracket.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$instr,
    ArrayRefParameter<"uint32_t">:$or_lo,
    "uint32_t":$mt_lo,
    ArrayRefParameter<"uint32_t">:$ir_lo,
    ArrayRefParameter<"uint32_t">:$or_hi,
    "uint32_t":$mt_hi,
    ArrayRefParameter<"uint32_t">:$ir_hi);
  let hasCustomAssemblyFormat = 1;
}

def AnchorAttr : Pasm_Attr<"Anchor", "anchor"> {
  let summary = "Single icdep anchor (instr + OR.MT.IR indices + delay)";
  let description = [{
    Bundles an icdep anchor's instruction symbol, OR indices, MT event id, IR
    indices, and delay. Printed as
    `#pasm.anchor<@instr[, "<or>.<mt>.<ir>"][, delay]>`, where the index text
    follows the ::vesyla::Anchor grammar. An empty index text and zero delay
    are omitted on print.
  }];
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$instr,
    ArrayRefParameter<"int32_t">:$or_idx,
    "int32_t":$mt,
    ArrayRefParameter<"int32_t">:$ir_idx,
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
