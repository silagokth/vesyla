#include "tm/AffineForWithPasmRopPattern.hpp"
#include "util/Common.hpp"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/IR/AffineMap.h"

bool isEmptyAffineForOp(affine::AffineForOp forOp) {
  return forOp.getBody()->getOperations().empty();
}

PasmRopPresence containsPasmRop(affine::AffineForOp forOp) {
  if (isEmptyAffineForOp(forOp)) {
    return PasmRopPresence::Empty;
  }
  for (auto &op : forOp.getBody()->getOperations()) {
    if (llvm::isa<vesyla::pasm::RopOp>(&op)) {
      return PasmRopPresence::Present;
    }
    // Recursively check nested AffineForOps
    if (auto nestedForOp = llvm::dyn_cast<affine::AffineForOp>(&op)) {
      if (containsPasmRop(nestedForOp) == PasmRopPresence::Present) {
        return PasmRopPresence::Present;
      } else if (isEmptyAffineForOp(nestedForOp)) {
        // If the nested loop is empty, we can skip it
        return PasmRopPresence::Empty;
      }
    }
  }
  return PasmRopPresence::None;
}

RepOp extractRepOp(affine::AffineForOp forOp) {
  auto count = affine::getConstantTripCount(forOp);
  if (!count.has_value()) {
    throw std::runtime_error("Dynamic loop iteration count is not supported.");
  }

  RepOp repOp;
  repOp.step = forOp.getStepAsInt();
  repOp.iter = count.value();

  return repOp;
}

std::vector<RepOp> extractRepOps(affine::AffineForOp forOp,
                                 SmallVector<int64_t, 4> &flattenedExpr) {
  std::vector<RepOp> repOps;
  repOps.push_back(extractRepOp(forOp));

  for (auto &op : forOp.getBody()->getOperations()) {
    flattenedExpr.reserve(repOps.size());
    if (auto nestedForOp = llvm::dyn_cast<affine::AffineForOp>(&op)) {
      auto nestedRepOps = extractRepOps(nestedForOp, flattenedExpr);
      repOps.insert(repOps.end(), nestedRepOps.begin(), nestedRepOps.end());
    } else if (auto applyOp = llvm::dyn_cast<affine::AffineApplyOp>(&op)) {
      auto map = applyOp.getAffineMap();
      auto result = getFlattenedAffineExpr(map.getResult(0), map.getNumDims(),
                                           map.getNumSymbols(), &flattenedExpr);

      llvm::errs() << "Found affine map: " << map << "\n";
    }
  }

  return repOps;
}

LogicalResult
updateRepOpsWithFlattenedExpr(std::vector<RepOp> &repOps,
                              const SmallVector<int64_t, 4> &flattenedExpr) {
  if (flattenedExpr.size() != repOps.size() + 1)
    throw std::runtime_error(
        "Mismatch between flattened expression size and repOps size.");

  for (int idx = 0; idx < flattenedExpr.size(); ++idx) {
    auto val = flattenedExpr[idx];
    if (idx < repOps.size())
      repOps[idx].step = val;
  }

  return success();
}

std::vector<pasm::RopOp> collectPasmRopOps(affine::AffineForOp forOp) {
  std::vector<pasm::RopOp> pasmRopOps;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto ropOp = llvm::dyn_cast<vesyla::pasm::RopOp>(&op)) {
      pasmRopOps.push_back(ropOp);
    }

    // Recursively check nested AffineForOps
    if (auto nestedForOp = llvm::dyn_cast<affine::AffineForOp>(&op)) {
      auto nestedPasmRopOps = collectPasmRopOps(nestedForOp);
      pasmRopOps.insert(pasmRopOps.end(), nestedPasmRopOps.begin(),
                        nestedPasmRopOps.end());
    }
  }
  return pasmRopOps;
}

LogicalResult injectRepInstructions(PatternRewriter &rewriter,
                                    affine::AffineForOp forOp,
                                    std::vector<RepOp> &repOps,
                                    std::vector<pasm::RopOp> &pasmRopOps) {
  rewriter.setInsertionPoint(forOp);

  for (auto &ropOp : pasmRopOps) {
    auto clonedRopOp =
        cast<vesyla::pasm::RopOp>(rewriter.clone(*ropOp.getOperation()));
    mlir::Block &bodyBlock = clonedRopOp.getBody().front();
    if (!bodyBlock.empty() &&
        bodyBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(&bodyBlock.back());
    } else {
      rewriter.setInsertionPointToEnd(&bodyBlock);
    }

    for (const auto &repOp : repOps) {
      rewriter.create<vesyla::pasm::InstrOp>(
          clonedRopOp.getLoc(),
          rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8)),
          rewriter.getStringAttr("rep"),
          rewriter.getDictionaryAttr({
              rewriter.getNamedAttr("iter",
                                    rewriter.getI32IntegerAttr(repOp.iter)),
              rewriter.getNamedAttr("step",
                                    rewriter.getI32IntegerAttr(repOp.step)),
              rewriter.getNamedAttr("delay",
                                    rewriter.getI32IntegerAttr(repOp.delay)),
          }));
    }
    rewriter.setInsertionPointAfter(clonedRopOp);
  }

  return success();
}

LogicalResult rewriteAffineForOp(affine::AffineForOp forOp,
                                 PatternRewriter &rewriter) {
  // Extract repetition levels and steps
  SmallVector<int64_t, 4> flattenedExpr;
  std::vector<RepOp> repOps = extractRepOps(forOp, flattenedExpr);

  if (!flattenedExpr.empty())
    if (!updateRepOpsWithFlattenedExpr(repOps, flattenedExpr).succeeded())
      return failure();

  // Collect rop operations and inject rep instructions
  std::vector<pasm::RopOp> pasmRopOps = collectPasmRopOps(forOp);

  if (failed(injectRepInstructions(rewriter, forOp, repOps, pasmRopOps)))
    return failure();

  rewriter.eraseOp(forOp);

  return success();
}

LogicalResult
AffineForWithPasmRopPattern::matchAndRewrite(affine::AffineForOp forOp,
                                             PatternRewriter &rewriter) const {
  PasmRopPresence hasPasmRop = containsPasmRop(forOp);
  switch (hasPasmRop) {
  case PasmRopPresence::Empty:
    rewriter.eraseOp(forOp);
    return success();
    break;
  case PasmRopPresence::Present:
    return rewriteAffineForOp(forOp, rewriter);
    break;
  case PasmRopPresence::None:
    return failure();

  default:
    break;
  }

  return failure();
}
