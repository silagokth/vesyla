#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "SelectInstructionsPass.hpp"

#include "Matcher.hpp"
#include "PatternLibrary.hpp"
#include "Rewriter.hpp"

#include <cstdlib>

namespace vesyla::conversion::select_instructions {
#define GEN_PASS_DEF_SELECTINSTRUCTIONSPASS
#include "conversion/select_instructions/Passes.hpp.inc"

namespace {

// The upstream ops this pass is responsible for selecting away. One left
// unclaimed is not fatal -- it is reported so a single run shows every gap.
bool is_selectable_source(mlir::Operation *op) {
  return llvm::isa<mlir::arith::MulIOp, mlir::arith::AddIOp,
                   mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                   mlir::affine::AffineVectorLoadOp,
                   mlir::affine::AffineVectorStoreOp>(op);
}

class SelectInstructionsPass
    : public impl::SelectInstructionsPassBase<SelectInstructionsPass> {
public:
  using impl::SelectInstructionsPassBase<
      SelectInstructionsPass>::SelectInstructionsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    // Where the resource descriptions live. The pass option wins so a test can
    // point at a fixture; otherwise the installed component library.
    std::string path = componentPath;
    if (path.empty()) {
      const char *env = std::getenv("VESYLA_SUITE_PATH_COMPONENTS");
      path = env ? env : "";
    }
    if (path.empty()) {
      module.emitError("select-instructions: no component path -- set "
                       "VESYLA_SUITE_PATH_COMPONENTS or pass component-path");
      return signalPassFailure();
    }

    PatternLibrary library;
    if (mlir::failed(library.load(path, &getContext(), llvm::errs()))) {
      return signalPassFailure();
    }
    for (const auto &[what, why] : library.skipped()) {
      llvm::errs() << "select-instructions: not matching " << what << " -- "
                   << why << "\n";
    }

    llvm::DenseSet<mlir::Operation *> claimed;

    // Highest benefit first, and everything at one benefit is tried together.
    // A match is replaced as soon as it is found, so the bindings it hands the
    // rewriter name values that are live at that moment.
    //
    // That ordering works out on its own. A pattern that matched earlier built
    // a rop which is an ordinary IR user, so when the op producing one of its
    // operands is itself replaced later, replaceAllUsesWith rewires it along
    // with everything else -- the multiply-accumulate rop picks up the load
    // rops that replace its operands without anyone tracking it.
    //
    // The claimed set is what makes this maximal munch: @mac (20) takes the
    // multiply and the add as one, so @mult and @add (10) cannot take them
    // apart on a later tier, and an IO-buffer access goes to io (12) before
    // rf (10) ever sees it.
    for (const std::vector<const Pattern *> &tier : library.tiers()) {
      // Snapshot the tier's candidates before rewriting any of them: replacing
      // inserts ops, and walking a region while it grows is not safe. Nothing
      // is erased until every tier has run, so these stay valid throughout.
      llvm::SmallVector<mlir::Operation *> candidates;
      module.walk([&](mlir::Operation *op) { candidates.push_back(op); });

      for (mlir::Operation *op : candidates) {
        if (claimed.contains(op)) {
          continue;
        }
        for (const Pattern *pattern : tier) {
          MatchResult result;
          if (!Matcher(*pattern).match(op, result)) {
            continue;
          }
          // A match reaching into ops another pattern already took would
          // double-claim them. They are still in the IR at this point -- only
          // the claimed set says they are spoken for.
          if (llvm::any_of(result.cone, [&](mlir::Operation *covered) {
                return claimed.contains(covered);
              })) {
            continue;
          }
          if (mlir::failed(
                  applyReplacement(*pattern, op, result, llvm::errs()))) {
            return signalPassFailure();
          }
          for (mlir::Operation *covered : result.cone) {
            claimed.insert(covered);
          }
          break;
        }
      }
    }

    // Every match has been replaced, so the matched ops are dead. Erasing is
    // left to the end rather than done per match: it frees the operations, and
    // the candidate snapshots above still point at them.
    llvm::SmallVector<mlir::Operation *> dead;
    module.walk([&](mlir::Operation *op) {
      if (claimed.contains(op)) {
        dead.push_back(op);
      }
    });
    for (mlir::Operation *op : llvm::reverse(dead)) {
      op->dropAllUses();
      op->erase();
    }

    // affine.apply ops that only fed converted accesses are now dead; their
    // index maps were lifted onto the rops.
    llvm::SmallVector<mlir::affine::AffineApplyOp> deadApplies;
    module.walk([&](mlir::affine::AffineApplyOp apply) {
      if (apply.use_empty()) {
        deadApplies.push_back(apply);
      }
    });
    for (mlir::affine::AffineApplyOp apply : deadApplies) {
      apply.erase();
    }

    // Report rather than fail: one run should show every gap, not just the
    // first.
    module.walk([&](mlir::Operation *op) {
      if (is_selectable_source(op)) {
        op->emitWarning("select-instructions: no pattern claimed this "
                        "operation");
      }
    });
  }
};

} // namespace
} // namespace vesyla::conversion::select_instructions
