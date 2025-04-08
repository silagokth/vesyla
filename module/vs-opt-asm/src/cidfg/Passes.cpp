//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cidfg/Passes.hpp"

#include <iostream>

using namespace mlir;

namespace vesyla::cidfg {
#define GEN_PASS_DEF_CIDFGCONSTANTFOLDING
#include "cidfg/Passes.hpp.inc"

namespace {
class CidfgConstantFoldingRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    std::cout << "op: " << op.getSymName() << std::endl;
    if (op.getSymName() == "scalar_binop") {
      auto constantAttr = op.getArgAttrOfType<IntegerAttr>(0, "op");
      std::cout << "constantAttr: " << constantAttr.getInt() << std::endl;
      return success();
    }
    return failure();
  }
};

class CidfgConstantFolding
    : public impl::CidfgConstantFoldingBase<CidfgConstantFolding> {
public:
  using impl::CidfgConstantFoldingBase<
      CidfgConstantFolding>::CidfgConstantFoldingBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CidfgConstantFoldingRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace vesyla::cidfg
