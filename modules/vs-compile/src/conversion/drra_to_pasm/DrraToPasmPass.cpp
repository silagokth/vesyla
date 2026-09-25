#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

#include "DrraToPasmPass.hpp"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

namespace vesyla::conversion::drra_to_pasm {
#define GEN_PASS_DEF_DRRATOPASMPASS
#include "conversion/drra_to_pasm/Passes.hpp.inc"

namespace {

// Lift an op out to its outermost enclosing affine.for, if any. Returns the op
// itself when it is not nested in any affine.for.
mlir::Operation *lift_out_of_loops(mlir::Operation *op) {
  mlir::Operation *result = op;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<mlir::affine::AffineForOp>(parent)) {
      result = parent;
    }
  }
  return result;
}

// The earliest producer of the rop's operands, lifted out of any enclosing
// affine.for. Returns null when the rop has no operand with a defining op.
mlir::Operation *earliest_producer(drra::RopOp rop) {
  mlir::Operation *target = nullptr;
  for (mlir::Value input : rop.getInputs()) {
    if (!input) {
      continue;
    }
    mlir::Operation *def = input.getDefiningOp();
    if (!def) {
      continue;
    }
    mlir::Operation *anchor = lift_out_of_loops(def);
    if (!target) {
      target = anchor;
    } else if (anchor->getBlock() == target->getBlock() &&
               anchor->isBeforeInBlock(target)) {
      target = anchor;
    }
  }
  return target;
}

// Rebuild an affine.for without its loop-carried values: drop the iter_args,
// the loop results, and the affine.yield operands. The data flow they carried
// is irrelevant in pasm.
void strip_iter_args(mlir::affine::AffineForOp for_op) {
  if (for_op.getNumIterOperands() == 0) {
    return;
  }

  mlir::OpBuilder builder(for_op);
  auto new_for = mlir::affine::AffineForOp::create(
      builder, for_op.getLoc(), for_op.getLowerBoundOperands(),
      for_op.getLowerBoundMap(), for_op.getUpperBoundOperands(),
      for_op.getUpperBoundMap(), for_op.getStepAsInt());

  mlir::Block *old_body = for_op.getBody();
  mlir::Block *new_body = new_for.getBody();

  // Redirect the induction variable; drop everything the iter_args fed.
  old_body->getArgument(0).replaceAllUsesWith(new_body->getArgument(0));
  for (unsigned i = 1, e = old_body->getNumArguments(); i < e; ++i) {
    old_body->getArgument(i).dropAllUses();
  }

  // Move the body over, leaving the old affine.yield (with operands) behind to
  // be erased with the old loop. The new loop keeps its operand-less yield.
  mlir::Operation *new_terminator = new_body->getTerminator();
  new_body->getOperations().splice(new_terminator->getIterator(),
                                   old_body->getOperations(), old_body->begin(),
                                   std::prev(old_body->end()));

  for (mlir::Value result : for_op.getResults()) {
    result.dropAllUses();
  }
  for_op.erase();
}

// The symbol the pasm.rop lowered from one instruction of `rop` carries.
//
// A drra.rop that lowers to a single instruction keeps its own id, which is
// what the anchors GenerateIcdepPass and CreateConstraintsPass have already
// built refer to. One carrying both a conf and an evt lowers to two pasm.rops
// that are placed differently and so cannot share a symbol; there the conf
// takes a "_conf" suffix and the evt keeps the plain id, because the evt is the
// instruction dataflow and timing anchor at. CreateConstraintsPass names the
// conf the same way.
std::string rop_symbol(mlir::FlatSymbolRefAttr id, llvm::StringRef type,
                       bool splits) {
  if (splits && type == "conf") {
    return id.getValue().str() + "_conf";
  }
  return id.getValue().str();
}

// Build the pasm.rop for one instruction of `rop`, at the builder's current
// insertion point. `type` is "conf" or "evt" -- the instruction names the
// resource's isa.json declares -- and `segments` is the dictionary the
// component library wrote for it, keyed by segment name.
void build_instr_rop(mlir::OpBuilder &builder, drra::RopOp rop,
                     mlir::FlatSymbolRefAttr id, llvm::StringRef type,
                     mlir::DictionaryAttr segments, pasm::ResourceAttr resource,
                     mlir::AffineMapAttr map, bool splits) {
  mlir::Location loc = rop.getLoc();

  // A resource wider than one slot addresses its ports through several of
  // them, and the library says which one this instruction is issued to as an
  // offset from the instance's base slot. Absent on a single-slot resource,
  // which is every resource whose ports all live in one slot.
  int32_t slot = resource.getSlot();
  if (auto offset = rop->getAttrOfType<mlir::IntegerAttr>("slot_offset")) {
    slot += static_cast<int32_t>(offset.getInt());
  }

  // Build the pasm.rop with the resource location and the symbol above. The
  // port is not carried on the rop; it is attached to the instr below so it
  // only ends up on instructions that actually use it (see AddSlotPortPass).
  auto sym_name = builder.getStringAttr(rop_symbol(id, type, splits));
  auto pasm_rop = pasm::RopOp::create(
      builder, loc, sym_name, builder.getI32IntegerAttr(resource.getRow()),
      builder.getI32IntegerAttr(resource.getCol()),
      builder.getI32IntegerAttr(slot), map);

  // The segments become the instr's param dict as they stand. The resource
  // port is added when the library did not write one; AddSlotPortPass later
  // drops it from any instruction whose ISA definition has no port segment.
  llvm::SmallVector<mlir::NamedAttribute> instr_params(segments.begin(),
                                                       segments.end());
  if (!segments.get("port")) {
    instr_params.push_back(builder.getNamedAttr(
        "port", builder.getI32IntegerAttr(resource.getPort())));
  }

  // Build the pasm.instr inside the new pasm.rop body. The instr id is the
  // drra.rop id, an underscore, and the instruction name, which keeps the two
  // instructions of a split rop apart whichever symbol their rops took.
  mlir::Block *body = builder.createBlock(&pasm_rop.getBody());
  builder.setInsertionPointToEnd(body);
  pasm::InstrOp::create(
      builder, loc,
      builder.getStringAttr(id.getValue().str() + "_" + type.str()),
      builder.getStringAttr(type), builder.getDictionaryAttr(instr_params));
  pasm::YieldOp::create(builder, loc);
}

// Lower a single drra.rop into one pasm.rop per instruction it carries.
mlir::LogicalResult convert_rop(drra::RopOp rop, mlir::OpBuilder &builder) {
  // id is a symbol reference (e.g. @input_r_b); use its bare name.
  auto id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  // map is optional on the drra.rop; copy it through when present.
  auto map = rop->getAttrOfType<mlir::AffineMapAttr>("map");

  // The instruction segments instruction selection carried over from the
  // component library. Either may be absent: a DPU mode is a conf with no evt,
  // an RF access an evt with no conf, and an operation that has to be
  // configured and then triggered carries both.
  mlir::DictionaryAttr conf = rop.getConfAttr();
  mlir::DictionaryAttr evt = rop.getEvtAttr();

  // resource is a single ResourceAttr, which is what design-space exploration
  // writes, or an array of them on hand-written input; either way the pasm.rop
  // takes its location from the first resource.
  auto resource = rop->getAttrOfType<pasm::ResourceAttr>("resource");
  if (!resource) {
    if (auto resources = rop->getAttrOfType<mlir::ArrayAttr>("resource");
        resources && !resources.empty()) {
      resource = mlir::dyn_cast<pasm::ResourceAttr>(resources[0]);
    }
  }

  if (!id || !resource) {
    rop.emitError("drra.rop is missing id or resource attribute");
    return mlir::failure();
  }
  if (!conf && !evt) {
    rop.emitError("drra.rop carries neither a conf nor an evt, so there is no "
                  "instruction to lower it to");
    return mlir::failure();
  }
  const bool splits = conf && evt;

  // A conf that consumes values is placed before the earliest producer of its
  // operands (lifted out of any enclosing affine.for); a conf with no such
  // producer stays in place. A conf that is hoisted out of its enclosing loop
  // drops its map: the index math belongs to the iterations it no longer runs
  // in.
  //
  // TODO: this places each conf on its own, without looking at the confs
  // already placed on the same (row, col, slot). Design-space exploration only
  // keeps apart operations whose `uses` overlap, so two that want different
  // config registers are put on one resource on purpose, and two that want the
  // same one are never bound together -- but nothing here checks either, so a
  // second conf on a resource is emitted as if it were the first. What it
  // should do depends on what it finds: a conf that differs only in which
  // config register it writes can stand alongside the one already there, one
  // identical to it is redundant and should be dropped rather than issued
  // twice, and one that would overwrite a register still in use has to be
  // re-issued between the uses instead of hoisted past them -- which is a
  // scheduling question, so it may not belong in this pass at all.
  if (conf) {
    mlir::Operation *target = earliest_producer(rop);
    builder.setInsertionPoint(target ? target : rop.getOperation());
    mlir::AffineMapAttr conf_map = map;
    if (target && rop->getParentOfType<mlir::affine::AffineForOp>()) {
      conf_map = mlir::AffineMapAttr();
    }
    build_instr_rop(builder, rop, id, "conf", conf, resource, conf_map, splits);
  }

  // An evt stays where the drra.rop was.
  if (evt) {
    builder.setInsertionPoint(rop);
    build_instr_rop(builder, rop, id, "evt", evt, resource, map, splits);
  }

  // Outputs are ignored; drop any dangling uses before erasing.
  for (mlir::Value result : rop->getResults()) {
    result.dropAllUses();
  }
  rop.erase();
  return mlir::success();
}

class DrraToPasmPass : public impl::DrraToPasmPassBase<DrraToPasmPass> {
public:
  using impl::DrraToPasmPassBase<DrraToPasmPass>::DrraToPasmPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    // First drop loop-carried values so no affine.for/affine.yield references a
    // drra.rop result. walk() is post-order, so inner loops are handled first.
    llvm::SmallVector<mlir::affine::AffineForOp> loops;
    module.walk([&](mlir::affine::AffineForOp for_op) {
      loops.push_back(for_op);
    });
    for (mlir::affine::AffineForOp for_op : loops) {
      strip_iter_args(for_op);
    }

    // Convert the rops in reverse program order: a consumer is lowered (and its
    // operand uses removed) before its producer, so each rop is use-free when
    // erased.
    llvm::SmallVector<drra::RopOp> rops;
    module.walk([&](drra::RopOp rop) { rops.push_back(rop); });
    mlir::OpBuilder builder(&getContext());
    for (auto it = rops.rbegin(); it != rops.rend(); ++it) {
      if (mlir::failed(convert_rop(*it, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace vesyla::conversion::drra_to_pasm
