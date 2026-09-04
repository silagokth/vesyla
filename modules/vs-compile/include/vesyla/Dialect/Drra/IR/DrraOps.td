#ifndef VESYLA_DRRA_OPS
#define VESYLA_DRRA_OPS

include "vesyla/Dialect/Drra/IR/DrraDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class Drra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Drra_Dialect, mnemonic, traits>;

def RopOp : Drra_Op<"rop"> {
  let summary = "Generic resource operation with variadic inputs and outputs";
  let description = [{
    A rop carries zero or more input values and produces zero or more
    output values.

    `conf` and `evt` hold the instruction segments -- keyed by segment name, as
    the resource's isa.json declares them -- that the later lowering needs to
    build a pasm.instr. Either may be absent, meaning that resource needs no
    instruction of that kind for this operation: a DPU mode is a conf with no
    evt, an RF access is an evt with no conf, and an operation that has to be
    configured and then triggered carries both. Note that the two do not stay
    together downstream: DrraToPasmPass hoists a conf out of the enclosing loop
    to before the earliest producer of its operands, while an evt stays where
    the rop was.

    `uses` names the parts of the resource this operation holds while it is
    active -- an array of strings such as `["conf_reg:0", "input_narrow:0",
    "input_narrow:1", "output_narrow:1"]` for a dpu mac, or `["agu:1"]` for a
    dpu rst.
    A resource is not indivisible, and two operations on one instance only
    contend when they want the same part of it, so this is what the conflict
    graph in design-space exploration intersects to decide whether they can
    share. The strings are opaque to the compiler: two parts are the same part
    when they are spelled the same, so the resource author owns the vocabulary.
    It is written by the resource's own description in the component library --
    the `@replace` rop declares it and instruction selection carries it over --
    and an operation that declares none is read as holding the whole resource,
    which conflicts with everything of its kind.

    `delay` is the cycles from the event that starts the operation to its result
    being there to be taken. CreateConstraintsPass accumulates it along a
    dataflow chain to time each operation against the next, and GenerateIcdepPass
    copies it onto the icdep anchors built from the operation. It is a property
    of the resource, so the library states it and instruction selection carries
    it over; an operation that declares none is read as taking no time, which
    times its consumer as simultaneous with it.

    `endpoints` says which slot of the resource each of the operation's operands
    and results travels on, as an offset from the first slot the instance
    occupies -- `endpoints = {results = [0 : i32], operands = [0 : i32, 1 : i32]}`
    for a dpu whose two narrow inputs are two different slots. A resource wider
    than one slot spreads its endpoints across them, and the switchbox addresses
    a channel by slot and knows nothing of ports, so two operands left on one
    slot would share one channel and only one of them would arrive. `uses`
    cannot answer this: those names belong to the resource author and are
    compared only against each other. Design-space exploration reads it to write
    the `resource` attribute as the [results..., operands...] array
    GenerateIcdepPass indexes, rather than as a single endpoint. Absent means
    every endpoint sits on the instance's first slot, which is what a
    single-slot resource wants; an operation with several endpoints on a wider
    one has to declare it, and is an error if it does not.

    `storage` names the memref this operation lands in, for a resource that
    holds its own -- an rf's registers, not an io's external buffer. It is the
    `id` symbol of the memref's alloc, lifted by instruction selection, which is
    the only place it is still known: the replacement leaves the memref argument
    unused, because the AGU sweeps the address and the memref is not an
    instruction operand. Design-space exploration binds every operation sharing
    a `storage` to one instance, since every access to one register file has to
    land on one register file. Absent on a resource that computes rather than
    stores.

    Every other attribute is free-form and lives in the op's attribute
    dictionary.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       OptionalAttr<DictionaryAttr>:$conf,
                       OptionalAttr<DictionaryAttr>:$evt);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
}

#endif // VESYLA_DRRA_OPS
