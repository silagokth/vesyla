#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "AddDefaultValuePass.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_ADDDEFAULTVALUEPASS
#include "pasm/Passes.hpp.inc"

namespace {
//===----------------------------------------------------------------------===//
class AddDefaultValuePassRewriter : public OpRewritePattern<InstrOp> {
public:
  using OpRewritePattern<InstrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(InstrOp op,
                                PatternRewriter &rewriter) const final {
    bool flag = false;

    std::string instr_name = op.getType().str();

    vesyla::pasm::Config cfg;
    nlohmann::json component_map_json = cfg.get_component_map_json();
    nlohmann::json isa_json = cfg.get_isa_json();

    // get the parent of the current operation
    mlir::Operation *parent_op = op->getParentOp();
    if (!parent_op) {
      llvm::outs()
          << "Error: InstrOp must be inside a RopOp or CopOp or RawOp.\n";
      exit(-1);
    }
    std::string label = "";
    if (auto rop_op = llvm::dyn_cast<RopOp>(parent_op)) {
      int row = rop_op.getRow();
      int col = rop_op.getCol();
      int slot = rop_op.getSlot();
      int port = rop_op.getPort();
      label = std::to_string(row) + "_" + std::to_string(col) + "_" +
              std::to_string(slot) + "_" + std::to_string(port);
    } else if (auto cop_op = llvm::dyn_cast<CopOp>(parent_op)) {
      int row = cop_op.getRow();
      int col = cop_op.getCol();
      label = std::to_string(row) + "_" + std::to_string(col);
    } else if (auto raw_op = llvm::dyn_cast<RawOp>(parent_op)) {
      int row = raw_op.getRow();
      int col = raw_op.getCol();
      label = std::to_string(row) + "_" + std::to_string(col);
    } else {
      llvm::outs()
          << "Error: InstrOp must be inside a RopOp or CopOp or RawOp.\n";
      exit(-1);
    }

    if (component_map_json.find(label) == component_map_json.end()) {
      llvm::outs() << "Error: Cannot find the component for label: " << label
                   << "\n";
      std::exit(-1);
    }
    std::string resource_kind = component_map_json[label].get<std::string>();
    // save the dictionary attributes of the InstrOp to a small vector
    std::set<std::string> instr_params_name_set;
    mlir::DictionaryAttr current_instr_params = op.getParam();
    llvm::SmallVector<mlir::NamedAttribute> updated_attrs;
    bool attr_changed = false;
    for (const mlir::NamedAttribute &named_attr_entry : current_instr_params) {
      auto attr_name = named_attr_entry.getName();
      updated_attrs.push_back(named_attr_entry);
      instr_params_name_set.insert(attr_name.str());
    }

    // get the complete field name and default value from the ISA JSON
    for (auto component : isa_json["components"]) {
      if (component["kind"] == resource_kind) {
        for (auto instr : component["instructions"]) {
          if (instr["name"] == instr_name) {
            for (auto segment : instr["segments"]) {
              if (instr_params_name_set.find(
                      segment["name"].get<std::string>()) ==
                  instr_params_name_set.end()) {
                int default_val = 0;
                if (segment.contains("default_val")) {
                  default_val = segment["default_val"].get<int>();
                }

                updated_attrs.push_back(rewriter.getNamedAttr(
                    segment["name"].get<std::string>(),
                    rewriter.getI32IntegerAttr(default_val)));
                attr_changed = true;
              }
            }
          }
        }
      }
    }

    if (attr_changed) {
      // If any attributes were changed, create a new DictionaryAttr and
      // update the operation
      mlir::DictionaryAttr new_instr_params =
          rewriter.getDictionaryAttr(updated_attrs);
      op->setAttr("param", new_instr_params);
      flag = true;
    }

    if (flag) {
      return success();
    } else {
      return failure();
    }
  }
};

class AddDefaultValuePass
    : public impl::AddDefaultValuePassBase<AddDefaultValuePass> {
public:
  using impl::AddDefaultValuePassBase<
      AddDefaultValuePass>::AddDefaultValuePassBase;

  void runOnOperation() {
    // Get the current module
    mlir::ModuleOp module = getOperation();

    // Create a pattern set
    RewritePatternSet patterns(&getContext());
    patterns.add<AddDefaultValuePassRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    // Apply the patterns to the module
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm