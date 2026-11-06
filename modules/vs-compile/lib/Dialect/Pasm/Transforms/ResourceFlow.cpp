#include "vesyla/Dialect/Pasm/Transforms/ResourceFlow.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace vesyla::pasm {

// Read the resource for a given result of a producer op. Ops with a single
// endpoint carry a plain ResourceAttr; ops with multiple inputs and an output
// carry an array laid out as [outputs..., inputs...], so result `r` lives at
// index `r`.
ResourceAttr get_result_resource(mlir::Operation *op, unsigned result_index) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<ResourceAttr>(attr)) {
    return res;
  }
  if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    if (result_index < arr.size()) {
      return mlir::dyn_cast<ResourceAttr>(arr[result_index]);
    }
  }
  return nullptr;
}

// Read the resource for a given operand of a consumer op. In the array layout
// [outputs..., inputs...], operand `o` lives at index `num_results + o`.
ResourceAttr get_operand_resource(mlir::Operation *op,
                                  unsigned operand_index) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<ResourceAttr>(attr)) {
    return res;
  }
  if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    unsigned idx = op->getNumResults() + operand_index;
    if (idx < arr.size()) {
      return mlir::dyn_cast<ResourceAttr>(arr[idx]);
    }
  }
  return nullptr;
}

// Forward-traverse the SSA value `value`, collecting the resource of every
// resourced consumer reachable from it. Resourced consumers terminate a chain.
// affine.for iter-args and affine.yield carry no resource attribute, so they are
// "looked through" to follow loop-carried data flow:
//   - a value used as an affine.for iter-arg init flows into the matching region
//     iter-arg (iteration 0 and, via the yield back-edge, every later one);
//   - a value yielded by affine.yield flows into both the matching loop result
//     (the final value) and the matching region iter-arg (the next iteration).
// `visited` guards against the iter-arg <-> yield cycle.
void collect_consumer_resources(mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Attribute> &dst,
                                llvm::SmallPtrSetImpl<void *> &visited) {
  auto follow = [&](mlir::Value next) {
    if (visited.insert(next.getAsOpaquePointer()).second) {
      collect_consumer_resources(next, dst, visited);
    }
  };

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *consumer = use.getOwner();
    if (consumer->hasAttr("resource")) {
      ResourceAttr dres =
          get_operand_resource(consumer, use.getOperandNumber());
      if (dres) {
        dst.push_back(dres);
      }
      continue;
    }
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(consumer)) {
      unsigned ctrl = loop.getNumControlOperands();
      if (use.getOperandNumber() >= ctrl) {
        unsigned i = use.getOperandNumber() - ctrl;
        if (i < loop.getRegionIterArgs().size()) {
          follow(loop.getRegionIterArgs()[i]);
        }
      }
      continue;
    }
    if (auto yield = mlir::dyn_cast<mlir::affine::AffineYieldOp>(consumer)) {
      if (auto loop =
              mlir::dyn_cast<mlir::affine::AffineForOp>(yield->getParentOp())) {
        unsigned i = use.getOperandNumber();
        if (i < loop.getNumResults()) {
          follow(loop.getResult(i));
          follow(loop.getRegionIterArgs()[i]);
        }
      }
      continue;
    }
  }
}

// The data kind a value travels as: a vector is a bulk transfer, a scalar a
// word transfer.
llvm::StringRef kind_of(mlir::Value value) {
  return mlir::isa<mlir::ShapedType>(value.getType()) ? "bulk" : "word";
}

} // namespace vesyla::pasm
