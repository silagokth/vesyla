#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "vesyla/Dialect/Pasm/Transforms/ExtractCellsPass.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_EXTRACTCELLSPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

class ExtractCellsPassRewriter : public mlir::OpRewritePattern<EpochOp> {
public:
  using mlir::OpRewritePattern<EpochOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EpochOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::Block &epoch_block = op.getBody().front();

    // Snapshot top-level rops and icdeps so iteration isn't invalidated by
    // moves/erases.
    llvm::SmallVector<RopOp> rops;
    for (RopOp rop : epoch_block.getOps<RopOp>()) {
      rops.push_back(rop);
    }
    llvm::SmallVector<IcDepOp> word_icdeps;
    llvm::SmallVector<IcDepOp> bulk_icdeps;
    for (IcDepOp icdep : epoch_block.getOps<IcDepOp>()) {
      if (icdep.getKind() == "word") {
        word_icdeps.push_back(icdep);
      } else if (icdep.getKind() == "bulk") {
        bulk_icdeps.push_back(icdep);
      }
    }
    if (rops.empty() && word_icdeps.empty() && bulk_icdeps.empty()) {
      return mlir::failure();
    }

    // Index any cells already present in this epoch by their (row, col).
    llvm::DenseMap<std::pair<int32_t, int32_t>, CellOp> cells;
    for (CellOp cell : epoch_block.getOps<CellOp>()) {
      cells[{cell.getRow(), cell.getCol()}] = cell;
    }

    auto find_or_create_cell = [&](int32_t r, int32_t c,
                                   mlir::Location loc) -> CellOp {
      auto it = cells.find({r, c});
      if (it != cells.end()) {
        return it->second;
      }
      // New CellOp at the end of the epoch body (before its terminator).
      rewriter.setInsertionPoint(epoch_block.getTerminator());
      CellOp cell = CellOp::create(rewriter, loc, rewriter.getI32IntegerAttr(r),
                                   rewriter.getI32IntegerAttr(c));
      rewriter.createBlock(&cell.getBody());
      cells[{r, c}] = cell;
      return cell;
    };

    for (RopOp rop : rops) {
      CellOp cell =
          find_or_create_cell(rop.getRow(), rop.getCol(), rop.getLoc());
      mlir::Block &cell_block = cell.getBody().front();
      rop->moveBefore(&cell_block, cell_block.end());
    }

    for (IcDepOp icdep : word_icdeps) {
      auto src = mlir::dyn_cast<ResourceAttr>(icdep.getSrc());
      if (!src) {
        continue;
      }
      CellOp cell =
          find_or_create_cell(src.getRow(), src.getCol(), icdep.getLoc());
      mlir::Block &cell_block = cell.getBody().front();
      icdep->moveBefore(&cell_block, cell_block.end());
    }

    mlir::StringAttr kind_bulk = rewriter.getStringAttr("bulk");
    mlir::StringAttr dir_send = rewriter.getStringAttr("send");
    mlir::StringAttr dir_recv = rewriter.getStringAttr("recv");
    for (IcDepOp bulk : bulk_icdeps) {
      auto src_res = mlir::dyn_cast<ResourceAttr>(bulk.getSrc());
      auto dst_arr = mlir::dyn_cast<mlir::ArrayAttr>(bulk.getDst());
      if (!src_res || !dst_arr) {
        continue;
      }
      int32_t sr = src_res.getRow();
      int32_t sc = src_res.getCol();
      for (mlir::Attribute d : dst_arr) {
        auto dst_res = mlir::dyn_cast<ResourceAttr>(d);
        if (!dst_res) {
          continue;
        }
        int32_t rr = dst_res.getRow();
        int32_t rc = dst_res.getCol();

        AnchorAttr first = bulk.getFirst();
        AnchorAttr last = bulk.getLast();
        mlir::ArrayAttr dst_one = rewriter.getArrayAttr({dst_res});

        CellOp send_cell = find_or_create_cell(sr, sc, bulk.getLoc());
        rewriter.setInsertionPointToEnd(&send_cell.getBody().front());
        IcDepOp::create(rewriter, bulk.getLoc(), src_res, dst_one, kind_bulk,
                        first, last, dir_send);

        CellOp recv_cell = find_or_create_cell(rr, rc, bulk.getLoc());
        rewriter.setInsertionPointToEnd(&recv_cell.getBody().front());
        IcDepOp::create(rewriter, bulk.getLoc(), src_res, dst_one, kind_bulk,
                        first, last, dir_recv);
      }
      rewriter.eraseOp(bulk);
    }
    return mlir::success();
  }
};

class ExtractCellsPass : public impl::ExtractCellsPassBase<ExtractCellsPass> {
public:
  using impl::ExtractCellsPassBase<ExtractCellsPass>::ExtractCellsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ExtractCellsPassRewriter>(&getContext());
    mlir::FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
