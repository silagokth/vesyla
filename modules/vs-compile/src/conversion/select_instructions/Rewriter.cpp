#include "Rewriter.hpp"
#include "NativeHelpers.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"

namespace vesyla {
namespace conversion {
namespace select_instructions {

mlir::LogicalResult applyReplacement(const Pattern &pattern,
                                     mlir::Operation *root,
                                     const MatchResult &match,
                                     llvm::raw_ostream &diag) {
  // MLIR op handles are value types whose accessors are not const-qualified, so
  // take copies rather than reaching through the const Pattern &.
  mlir::func::FuncOp replaceFn = pattern.replace;
  mlir::func::FuncOp matchFn = pattern.match;

  mlir::Block &body = replaceFn.getBody().front();
  mlir::Operation *terminator = body.getTerminator();

  // Argument i of the replacement stands for argument i of the match, so it
  // maps to whatever that argument bound to in the program. An argument the
  // match never bound -- the accumulate register, an AGU-swept address -- has
  // no counterpart, which is fine as long as the replacement does not use it.
  // Using one that was never bound means the pattern is asking for a value the
  // program does not have.
  mlir::IRMapping mapping;
  for (auto [replaceArg, matchArg] :
       llvm::zip(body.getArguments(), matchFn.getArguments())) {
    auto bound = match.bindings.find(matchArg);
    if (bound != match.bindings.end()) {
      mapping.map(replaceArg, bound->second);
    } else if (!replaceArg.use_empty()) {
      return diag << "select-instructions: " << pattern.kind << " @"
                  << pattern.name << ": replacement uses argument "
                  << replaceArg.getArgNumber()
                  << ", which the match never bound\n",
             mlir::failure();
    }
  }

  mlir::OpBuilder builder(root);
  auto id = root->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  mlir::AffineMapAttr map = vesyla::sel::lift_affine_map(root);

  for (mlir::Operation &op : body.without_terminator()) {
    mlir::Operation *clone = builder.clone(op, mapping);
    clone->setLoc(root->getLoc());
    // Per-match, so the library cannot state them: which program operation this
    // came from, and the address the AGU sweeps for it.
    if (llvm::isa<drra::RopOp>(clone)) {
      if (id) {
        clone->setAttr("id", id);
      }
      if (map) {
        clone->setAttr("map", map);
      }
      clone->setAttr("kind", builder.getStringAttr(pattern.kind));
    }
  }

  // What the replacement returns stands in for what the matched root produced.
  llvm::SmallVector<mlir::Value> results;
  for (mlir::Value operand : terminator->getOperands()) {
    results.push_back(mapping.lookupOrDefault(operand));
  }
  if (results.size() != root->getNumResults())
    return diag << "select-instructions: " << pattern.kind << " @"
                << pattern.name << ": replacement returns " << results.size()
                << " value(s) for a root producing " << root->getNumResults()
                << "\n",
           mlir::failure();

  // Everything still using the root moves to the replacement: ops outside every
  // cone (an affine.yield, a loop's iter_args) and rops built by earlier
  // matches that took this root's result as an operand.
  root->replaceAllUsesWith(results);
  return mlir::success();
}

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla
