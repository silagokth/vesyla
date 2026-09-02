#include "Strategy.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"

#include <memory>

namespace vesyla {
namespace transformation {
namespace dse {

namespace {

// The port the operation drives. Selection already fixed it: an rf word read is
// evt port 1, a bulk write evt port 2, a dpu rst evt port 1, and so on -- the
// port says which behaviour of the resource was selected, so it is not a
// binding decision. Binding picks (row, col, slot) and leaves the port alone.
int resolve_port(drra::RopOp rop) {
  for (const char *name : {"evt", "conf"}) {
    auto dict = rop->getAttrOfType<mlir::DictionaryAttr>(name);
    if (!dict) {
      continue;
    }
    if (auto port = dict.getAs<mlir::IntegerAttr>("port")) {
      return static_cast<int>(port.getInt());
    }
  }
  return 0;
}

class PlaceholderBinder : public Binder {
public:
  mlir::LogicalResult bind(mlir::Operation *scope,
                           const Architecture &arch) final {
    // Cache the lookup so a scope full of rf accesses does not rescan the
    // instance table for each one.
    llvm::DenseMap<mlir::StringAttr, unsigned> first_of_kind;
    bool failed = false;

    scope->walk([&](drra::RopOp rop) {
      auto kind = rop->getAttrOfType<mlir::StringAttr>("kind");
      if (!kind) {
        rop.emitError("design-space-exploration: operation carries no `kind`, "
                      "so there is nothing to bind it to -- instruction "
                      "selection should have written one");
        failed = true;
        return;
      }

      auto cached = first_of_kind.find(kind);
      if (cached == first_of_kind.end()) {
        llvm::SmallVector<unsigned> candidates =
            arch.instances_of_kind(kind.getValue());
        if (candidates.empty()) {
          rop.emitError() << "design-space-exploration: the architecture "
                             "allocates no resource of kind '"
                          << kind.getValue() << "'";
          failed = true;
          return;
        }
        cached = first_of_kind.insert({kind, candidates.front()}).first;
      }

      const ResourceInstance &instance = arch.at(cached->second);
      rop->setAttr("resource", pasm::ResourceAttr::get(
                                   rop.getContext(), instance.row, instance.col,
                                   instance.slot, resolve_port(rop)));
    });

    return mlir::failure(failed);
  }
};

} // namespace

std::unique_ptr<Binder> create_placeholder_binder() {
  return std::make_unique<PlaceholderBinder>();
}

} // namespace dse
} // namespace transformation
} // namespace vesyla
