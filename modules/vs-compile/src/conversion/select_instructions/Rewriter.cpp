#include "Rewriter.hpp"
#include "NativeHelpers.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"

namespace vesyla {
namespace conversion {
namespace select_instructions {

namespace {

// The storage this match touched, when the resource holds it itself.
//
// Binding has to put every access to one register file on one register file,
// and the rop alone cannot say which: the replacement leaves the memref
// argument unused, because the AGU sweeps the address and the memref is not an
// instruction operand. So the identity is lifted here, while the match that
// knows it is still in hand, and travels on the rop as `storage`.
//
// A pattern's memref argument names storage the resource owns when it is in the
// default address space. Address space 1 is the external io subsystem, which
// the io resource addresses but does not hold -- io.mlir says so in as many
// words -- so two io transfers to different buffers are still the same io. A
// rank-0 memref is the resource's own accumulate register, which is state
// rather than addressable storage and is what findAccumulator already picks
// out. Neither is storage to bind.
//
// Returns null when the pattern names no such argument, which is the ordinary
// case for a resource that computes rather than stores.
mlir::Value liftStorage(const Pattern &pattern, const MatchResult &match) {
  mlir::func::FuncOp matchFn = pattern.match;
  for (mlir::BlockArgument arg : matchFn.getArguments()) {
    auto memref = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
    if (!memref || memref.getRank() == 0 ||
        memref.getMemorySpaceAsInt() != 0) {
      continue;
    }
    auto bound = match.bindings.find(arg);
    if (bound != match.bindings.end()) {
      return bound->second;
    }
  }
  return {};
}

// What that storage is called. The program names its buffers with an id on the
// alloc, because MLIR drops SSA value names at parse time and %rf2 would
// otherwise be gone by the time any pass runs.
mlir::FlatSymbolRefAttr storageName(mlir::Value storage) {
  auto alloc = llvm::dyn_cast_or_null<mlir::memref::AllocOp>(
      storage.getDefiningOp());
  if (!alloc) {
    return {};
  }
  return alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
}

} // namespace

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

  // Which storage this access lands in, for a resource that holds its own.
  // Without a name there is no telling two of them apart, and binding would be
  // free to send two accesses to one array off to different instances -- wrong
  // silently, and in a way that only shows up as wrong results, so it is
  // reported here instead.
  mlir::FlatSymbolRefAttr storage;
  if (mlir::Value memref = liftStorage(pattern, match)) {
    storage = storageName(memref);
    if (!storage) {
      return diag << "select-instructions: " << pattern.kind << " @"
                  << pattern.name
                  << ": the memref this access lands in carries no `id`, so "
                     "two of them cannot be told apart when binding -- give "
                     "the memref.alloc an id attribute\n",
             mlir::failure();
    }
  }

  for (mlir::Operation &op : body.without_terminator()) {
    mlir::Operation *clone = builder.clone(op, mapping);
    clone->setLoc(root->getLoc());
    // Per-match, so the library cannot state them: which program operation this
    // came from, the address the AGU sweeps for it, and the storage it lands
    // in.
    if (llvm::isa<drra::RopOp>(clone)) {
      if (id) {
        clone->setAttr("id", id);
      }
      if (map) {
        clone->setAttr("map", map);
      }
      if (storage) {
        clone->setAttr("storage", storage);
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
