#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "MergeRawOp.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_MERGERAWOP
#include "pasm/Passes.hpp.inc"

namespace {
//===----------------------------------------------------------------------===//
class MergeRawOpRewriter : public OpRewritePattern<EpochOp> {
public:
  using OpRewritePattern<EpochOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(EpochOp op,
                                PatternRewriter &rewriter) const final {
    llvm::outs() << "MergeRawOpRewriter triggered on EpochOp.\n";

    // get parent block
    mlir::Block *moduleBlock = op.getOperation()->getBlock();
    if (!moduleBlock) {
      llvm::outs() << "Error: EpochOp is not inside a block.\n";
      exit(EXIT_FAILURE);
    }

    // Iterate through all operations in the module's block
    Operation *target_epoch_op = nullptr;
    std::vector<Operation *> epoch_op_list;
    llvm::outs() << "size of moduleBlock: "
                 << moduleBlock->getOperations().size() << "\n";
    for (mlir::Operation &child_op : moduleBlock->getOperations()) {
      // check if it's a EpochOp
      if (auto epoch_op = llvm::dyn_cast<EpochOp>(&child_op)) {
        llvm::outs() << "Found EpochOp: " << epoch_op.getId() << "\n";
        if (target_epoch_op == nullptr) {
          target_epoch_op = &child_op;
        } else {
          epoch_op_list.push_back(&child_op);
        }
        mlir::Region &child_op_region = epoch_op.getBody();
        if (child_op_region.empty()) {
          llvm::outs() << "Error: EpochOp has no body.\n";
          exit(EXIT_FAILURE);
        }
        mlir::Block *child_op_block = &child_op_region.front();
        // Iterate through all operations in the child operation's block
        for (mlir::Operation &child_op_child : *child_op_block) {
          if (llvm::dyn_cast<RawOp>(&child_op_child) ||
              llvm::dyn_cast<YieldOp>(&child_op_child)) {
            // DO NOTHING
          } else {
            llvm::outs() << "Error: Illegal operation type in EpochOp: "
                         << child_op_child.getName() << "\n";
            exit(EXIT_FAILURE);
          }
        }
      } else {
        llvm::outs() << "Error: Illegal operation type in EpochOp: "
                     << child_op.getName() << "\n";
        return mlir::failure();
      }
    }

    llvm::outs() << "Found " << epoch_op_list.size()
                 << " EpochOps in the module.\n";
    for (auto &child_op : epoch_op_list) {
      llvm::outs() << "EpochOp: " << llvm::dyn_cast<EpochOp>(child_op).getId()
                   << "\n";
    }

    if (epoch_op_list.size() == 0) {
      return mlir::failure();
    }

    // create a map for insertion location
    std::unordered_map<std::string, mlir::Operation *>
        target_insertion_raw_op_map;
    // get all the raw ops in the target epoch op
    for (auto &child_op : *target_epoch_op->getRegion(0).begin()) {
      if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
        int row = raw_op.getRow();
        int col = raw_op.getCol();
        std::string label = std::to_string(row) + "_" + std::to_string(col);
        if (target_insertion_raw_op_map.find(label) ==
            target_insertion_raw_op_map.end()) {
          target_insertion_raw_op_map[label] = &child_op;
        } else {
          llvm::outs() << "Error: RawOp already exists in the target epoch op: "
                       << label << "\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }

    // Iterate through all operations in the module's block
    for (auto &child_op : epoch_op_list) {
      if (auto epoch_op = llvm::dyn_cast<EpochOp>(child_op)) {
        mlir::Region &child_op_region = epoch_op.getBody();
        if (child_op_region.empty()) {
          continue;
        }
        mlir::Block *child_op_block = &child_op_region.front();
        // Iterate through all operations in the child operation's block
        for (mlir::Operation &child_op_child : *child_op_block) {
          if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op_child)) {
            int row = raw_op.getRow();
            int col = raw_op.getCol();
            std::string label = std::to_string(row) + "_" + std::to_string(col);
            if (target_insertion_raw_op_map.find(label) !=
                target_insertion_raw_op_map.end()) {
              // set the insertion point to the body of the target raw op, just
              // before the terminator
              Operation *target_raw_op = target_insertion_raw_op_map[label];
              mlir::Region &target_raw_op_region =
                  llvm::dyn_cast<RawOp>(target_raw_op).getBody();
              mlir::Block *target_raw_op_block;
              if (target_raw_op_region.empty()) {
                target_raw_op_block =
                    rewriter.createBlock(&target_raw_op_region);
              } else {
                target_raw_op_block = &target_raw_op_region.front();
              }

              // clone the child op to the target raw op
              mlir::Region &raw_op_body = raw_op.getBody();
              mlir::Block *raw_op_entry_block;
              if (raw_op_body.empty()) {
                raw_op_entry_block = rewriter.createBlock(&raw_op_body);
              } else {
                raw_op_entry_block = &raw_op_body.front();
              }
              for (mlir::Operation &raw_op_child : *raw_op_entry_block) {
                if (auto instr_op = llvm::dyn_cast<InstrOp>(&raw_op_child)) {
                  rewriter.setInsertionPoint(
                      target_raw_op_block->getTerminator());
                  rewriter.clone(raw_op_child);
                } else if (auto yield_op =
                               llvm::dyn_cast<YieldOp>(&raw_op_child)) {
                  // DO NOTHING
                } else {
                  llvm::outs() << "Error: Illegal operation type in RawOp: "
                               << raw_op_child.getName() << "\n";
                  exit(EXIT_FAILURE);
                }
              }
            } else {
              // clone this raw op to the target epoch op
              mlir::Region &target_epoch_op_region =
                  llvm::dyn_cast<EpochOp>(target_epoch_op).getBody();
              mlir::Block *target_epoch_op_block =
                  &target_epoch_op_region.front();
              rewriter.setInsertionPoint(
                  target_epoch_op_block->getTerminator());
              auto new_raw_op = rewriter.clone(child_op_child);
              // add the raw op to the target insertion map
              target_insertion_raw_op_map[label] = new_raw_op;
            }
          }
        }
      }
    }

    // remove all the epoch ops
    for (auto &child_op : epoch_op_list) {
      if (auto epoch_op = llvm::dyn_cast<EpochOp>(child_op)) {
        rewriter.eraseOp(epoch_op);
      }
    }

    // print the whole block
    llvm::outs() << "Block contents after merging:\n";
    for (mlir::Operation &child_op : *moduleBlock) {
      llvm::outs() << "Operation type: " << child_op.getName() << "\n";
      if (auto epoch_op = llvm::dyn_cast<EpochOp>(&child_op)) {
        llvm::outs() << "EpochOp ID: " << epoch_op.getId() << "\n";
        mlir::Region &epochBodyRegion = epoch_op.getBody();
        if (epochBodyRegion.empty()) {
          llvm::outs() << "Error: EpochOp has no body.\n";
          exit(EXIT_FAILURE);
        }
        mlir::Block *epochEntryBlock = &epochBodyRegion.front();
        // Iterate through all operations in the child operation's block
        for (mlir::Operation &epoch_child_op : *epochEntryBlock) {
          if (auto raw_op = llvm::dyn_cast<RawOp>(&epoch_child_op)) {
            llvm::outs() << "RawOp ID: " << raw_op.getId() << "\n";
          } else if (auto yield_op = llvm::dyn_cast<vesyla::pasm::YieldOp>(
                         &epoch_child_op)) {
            // DO NOTHING
          } else {
            llvm::outs() << "Error: Illegal operation type in EpochOp: "
                         << epoch_child_op.getName() << "\n";
            exit(EXIT_FAILURE);
          }
        }
      }
    }

    return mlir::success();
  }
};

class MergeRawOp : public impl::MergeRawOpBase<MergeRawOp> {
public:
  using impl::MergeRawOpBase<MergeRawOp>::MergeRawOpBase;

  void runOnOperation() {
    llvm::outs() << "Running MergeRawOp pass...\n";

    // Get the current module
    mlir::ModuleOp module = getOperation();

    // Create a pattern set
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeRawOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    // Apply the patterns to the module
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
