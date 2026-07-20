#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

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

// Lower a single drra.rop into a pasm.rop holding one pasm.instr.
mlir::LogicalResult convert_rop(drra::RopOp rop, mlir::OpBuilder &builder) {
  // id is a symbol reference (e.g. @input_r_b); use its bare name.
  auto id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  auto instr = rop->getAttrOfType<mlir::StringAttr>("instr");
  auto param = rop->getAttrOfType<mlir::DictionaryAttr>("param");
  // map is optional on the drra.rop; copy it through when present.
  auto map = rop->getAttrOfType<mlir::AffineMapAttr>("map");

  // resource is a single ResourceAttr (evt) or an array of them (conf); either
  // way the pasm.rop takes its location from the first resource.
  auto resource = rop->getAttrOfType<pasm::ResourceAttr>("resource");
  if (!resource) {
    if (auto resources = rop->getAttrOfType<mlir::ArrayAttr>("resource");
        resources && !resources.empty()) {
      resource = mlir::dyn_cast<pasm::ResourceAttr>(resources[0]);
    }
  }

  if (!id || !instr || !resource) {
    rop.emitError("drra.rop is missing id, instr, or resource attribute");
    return mlir::failure();
  }
  if (!param) {
    param = builder.getDictionaryAttr({});
  }

  // An evt stays where the drra.rop was. A conf that consumes values is placed
  // before the earliest producer of its operands (lifted out of any enclosing
  // affine.for); a conf with no such producer stays in place too. A conf that
  // is hoisted out of its enclosing loop drops its map.
  if (instr.getValue() == "conf") {
    mlir::Operation *target = earliest_producer(rop);
    builder.setInsertionPoint(target ? target : rop.getOperation());
    if (target && rop->getParentOfType<mlir::affine::AffineForOp>()) {
      map = mlir::AffineMapAttr();
    }
  } else {
    builder.setInsertionPoint(rop);
  }

  mlir::Location loc = rop.getLoc();

  // Build the pasm.rop with the resource location and the drra.rop's id. The
  // port is no longer carried on the rop; it is attached to the instr below so
  // it only ends up on instructions that actually use it (see AddSlotPortPass).
  auto sym_name = builder.getStringAttr(id.getValue());
  auto pasm_rop = pasm::RopOp::create(
      builder, loc, sym_name, builder.getI32IntegerAttr(resource.getRow()),
      builder.getI32IntegerAttr(resource.getCol()),
      builder.getI32IntegerAttr(resource.getSlot()), map);

  // Carry the resource port onto the instr's param dict. AddSlotPortPass later
  // drops it from any instruction whose ISA definition has no port segment.
  llvm::SmallVector<mlir::NamedAttribute> instr_params(param.begin(),
                                                       param.end());
  if (!param.get("port")) {
    instr_params.push_back(builder.getNamedAttr(
        "port", builder.getI32IntegerAttr(resource.getPort())));
  }

  // Build the single pasm.instr inside the new pasm.rop body. The instr id is
  // the drra.rop id, an underscore, and the instr type.
  mlir::Block *body = builder.createBlock(&pasm_rop.getBody());
  builder.setInsertionPointToEnd(body);
  pasm::InstrOp::create(
      builder, loc,
      builder.getStringAttr(id.getValue().str() + "_" + instr.str()), instr,
      builder.getDictionaryAttr(instr_params));
  pasm::YieldOp::create(builder, loc);

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
