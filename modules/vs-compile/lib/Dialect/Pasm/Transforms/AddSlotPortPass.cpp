#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "vesyla/Dialect/Pasm/Transforms/AddSlotPortPass.hpp"

#include <optional>

namespace vesyla::pasm {
#define GEN_PASS_DEF_ADDSLOTPORTPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// An instruction needs a port iff its ISA definition carries a "port" segment.
// Port-ness is a property of the instruction name (it is consistent across the
// components that define a given instruction), so it is looked up by name
// across all components, including any variant segment lists.
bool instr_needs_port(const nlohmann::json &isa_json,
                      llvm::StringRef instr_name) {
  auto has_port_segment = [](const nlohmann::json &segments) {
    for (const auto &segment : segments) {
      if (segment.contains("name") && segment["name"] == "port") {
        return true;
      }
    }
    return false;
  };
  for (const auto &component : isa_json["components"]) {
    for (const auto &instr : component["instructions"]) {
      if (!instr.contains("name") || instr["name"] != instr_name.str()) {
        continue;
      }
      if (instr.contains("segments") && has_port_segment(instr["segments"])) {
        return true;
      }
      if (instr.contains("variants")) {
        for (const auto &variant : instr["variants"]) {
          if (variant.contains("segments") &&
              has_port_segment(variant["segments"])) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
class AddSlotPortPassRewriter : public OpRewritePattern<RopOp> {
public:
  using OpRewritePattern<RopOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(RopOp op,
                                PatternRewriter &rewriter) const final {
    int32_t slot = op.getSlot();

    vesyla::pasm::Config cfg;
    nlohmann::json isa_json = cfg.get_isa_json();

    mlir::Region &op_region = op.getBody();
    mlir::Block *op_block = nullptr;
    if (op_region.empty()) {
      op_block = rewriter.createBlock(&op_region);
    } else {
      op_block = &op_region.front();
    }

    // All port-bearing instructions in a rop act on the same resource port. It
    // may be given on only some of them (e.g. on the evt but not the rep), so
    // pick it up from whichever instruction carries it and reuse it for the
    // others that need one.
    std::optional<int32_t> reference_port;
    for (auto &inst : op_block->getOperations()) {
      if (auto instr_op = mlir::dyn_cast<InstrOp>(inst)) {
        if (auto port_attr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
                instr_op.getParam().get("port"))) {
          reference_port = static_cast<int32_t>(port_attr.getInt());
          break;
        }
      }
    }

    bool flag = false;
    for (auto &inst : op_block->getOperations()) {
      if (auto instr_op = mlir::dyn_cast<InstrOp>(inst)) {
        mlir::DictionaryAttr current_instr_params = instr_op.getParam();
        bool needs_port = instr_needs_port(isa_json, instr_op.getType());
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
              // Keep the port only when the ISA defines a port segment for this
              // instruction; otherwise drop it so port lives only where it is
              // required.
              if (needs_port) {
                updated_attrs.push_back(named_attr_entry);
                found_port = true;
              } else {
                param_changed = true;
              }
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

        if (needs_port && !found_port && reference_port.has_value()) {
          // an instruction that needs a port but did not carry one inherits the
          // rop's resource port from its siblings.
          updated_attrs.push_back(rewriter.getNamedAttr(
              "port", rewriter.getI32IntegerAttr(*reference_port)));
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
