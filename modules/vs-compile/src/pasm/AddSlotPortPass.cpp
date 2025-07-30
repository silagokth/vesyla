#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "AddSlotPortPass.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_ADDSLOTPORTPASS
#include "pasm/Passes.hpp.inc"

namespace {
//===----------------------------------------------------------------------===//
class AddSlotPortPassRewriter : public OpRewritePattern<RopOp> {
public:
  using OpRewritePattern<RopOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(RopOp op,
                                PatternRewriter &rewriter) const final {
    int32_t slot = op.getSlot();
    int32_t port = op.getPort();
    mlir::Region &op_region = op.getBody();
    mlir::Block *op_block = nullptr;
    if (op_region.empty()) {
      op_block = rewriter.createBlock(&op_region);
    } else {
      op_block = &op_region.front();
    }
    bool flag = false;
    for (auto &inst : op_block->getOperations()) {
      if (auto instr_op = mlir::dyn_cast<InstrOp>(inst)) {
        mlir::DictionaryAttr current_instr_params = instr_op.getParam();
        llvm::SmallVector<mlir::NamedAttribute> updated_attrs;
        bool found_slot = false;
        bool found_port = false;
        bool param_changed = false;
        for (const mlir::NamedAttribute &named_attr_entry :
             current_instr_params) {
          auto attr_name = named_attr_entry.getName();
          auto attr_value = named_attr_entry.getValue();

          if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr_value)) {
            if (attr_name == "slot") {
              int32_t instr_slot = int_attr.getInt();
              if (instr_slot != slot) {
                llvm::outs()
                    << "Warning: Slot mismatch in InstrOp: " << instr_slot
                    << " != " << slot << "\n";
                exit(EXIT_FAILURE);
              } else {
                // Keep the original attribute if it matches the slot
                updated_attrs.push_back(named_attr_entry);
              }
              found_slot = true;
            } else if (attr_name == "port") {
              int32_t instr_port = int_attr.getInt();
              if (instr_port != port) {
                llvm::outs()
                    << "Warning: Port mismatch in InstrOp: " << instr_port
                    << " != " << port << "\n";
                exit(EXIT_FAILURE);
              } else {
                // Keep the original attribute if it matches the port
                updated_attrs.push_back(named_attr_entry);
              }
              found_port = true;
            } else {
              // Keep the original attribute
              updated_attrs.push_back(named_attr_entry);
            }

          } else {
            // If the attribute is not an IntegerAttr, keep it as is
            updated_attrs.push_back(named_attr_entry);
          }
        }

        if (!found_slot) {
          // add slot attribute if not found
          updated_attrs.push_back(
              rewriter.getNamedAttr("slot", rewriter.getI32IntegerAttr(slot)));
          param_changed = true;
        }
        if (!found_port) {
          // add port attribute if not found
          updated_attrs.push_back(
              rewriter.getNamedAttr("port", rewriter.getI32IntegerAttr(port)));
          param_changed = true;
        }

        // If any attributes were changed, create a new DictionaryAttr and
        // update the operation
        if (param_changed) {
          mlir::DictionaryAttr new_instr_params =
              rewriter.getDictionaryAttr(updated_attrs);
          instr_op->setAttr("param", new_instr_params);
          flag = true;
        }
      }
    }

    if (flag) {
      return success();
    } else {
      return failure();
    }
  }
};

class AddSlotPortPass : public impl::AddSlotPortPassBase<AddSlotPortPass> {
public:
  using impl::AddSlotPortPassBase<AddSlotPortPass>::AddSlotPortPassBase;

  void runOnOperation() {
    // Get the current module
    mlir::ModuleOp module = getOperation();

    // Create a pattern set
    RewritePatternSet patterns(&getContext());
    patterns.add<AddSlotPortPassRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    // Apply the patterns to the module
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
