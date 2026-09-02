#include "DesignSpaceExplorationPass.hpp"

#include "Architecture.hpp"
#include "Strategy.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"

#include <memory>

namespace vesyla::transformation::dse {
#define GEN_PASS_DEF_DESIGNSPACEEXPLORATIONPASS
#include "transformation/design_space_exploration/Passes.hpp.inc"

namespace {

// Does this operation carry a resource downstream can use? Either form is
// accepted: a single endpoint, which is what an operation with one port has,
// or the [results..., operands...] array an operation with several endpoints
// needs. GenerateIcdepPass reads both.
bool has_resource(mlir::Operation *op) {
  mlir::Attribute attr = op->getAttr("resource");
  if (mlir::isa_and_nonnull<pasm::ResourceAttr>(attr)) {
    return true;
  }
  auto array = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr);
  if (!array || array.empty()) {
    return false;
  }
  return llvm::all_of(array, [](mlir::Attribute entry) {
    return mlir::isa<pasm::ResourceAttr>(entry);
  });
}

// The check the pass ends on: nothing may leave here unbound.
//
// Every drra.rop is walked rather than stopping at the first gap, so one run
// reports every operation that needs attention instead of one per rebuild.
mlir::LogicalResult verify_all_bound(mlir::ModuleOp module) {
  bool ok = true;
  module.walk([&](drra::RopOp rop) {
    if (has_resource(rop)) {
      return;
    }
    mlir::InFlightDiagnostic diag =
        rop.emitError("design-space-exploration: operation was not bound to a "
                      "resource");
    if (auto id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id")) {
      diag << " (id @" << id.getValue() << ")";
    }
    if (auto kind = rop->getAttrOfType<mlir::StringAttr>("kind")) {
      diag << " (kind '" << kind.getValue() << "')";
    }
    ok = false;
  });
  return mlir::success(ok);
}

class DesignSpaceExplorationPass
    : public impl::DesignSpaceExplorationPassBase<DesignSpaceExplorationPass> {
public:
  using impl::DesignSpaceExplorationPassBase<
      DesignSpaceExplorationPass>::DesignSpaceExplorationPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    // Allocation. Read, not decided: the architecture file passed to the
    // executable already fixes which resources exist and where.
    mlir::FailureOr<Architecture> arch = Architecture::from_config(module);
    if (mlir::failed(arch)) {
      return signalPassFailure();
    }

    // Each pasm.epoch is handled on its own: an epoch is the region downstream
    // scheduling treats as a unit, so it is also the scope a solution is
    // searched over. A program with no epochs is handled as a whole.
    llvm::SmallVector<mlir::Operation *> scopes;
    module.walk([&](pasm::EpochOp epoch) { scopes.push_back(epoch); });
    if (scopes.empty()) {
      scopes.push_back(module.getOperation());
    }

    // Scheduling comes first, because it is what makes binding a question with
    // an answer: two operations may share an instance exactly when their
    // relative schedules keep them apart. Not written yet, so binding below
    // falls back on an order it does not have to justify.

    std::unique_ptr<Binder> binder = create_placeholder_binder();
    for (mlir::Operation *scope : scopes) {
      if (mlir::failed(binder->bind(scope, *arch))) {
        return signalPassFailure();
      }
    }

    if (mlir::failed(verify_all_bound(module))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::transformation::dse
