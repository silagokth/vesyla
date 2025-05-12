#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pasm/Passes.hpp"
#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include "util/Common.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_REPLACELOOPOP
#define GEN_PASS_DEF_REPLACEEPOCHOP
#include "pasm/Passes.hpp.inc"

namespace {

//===----------------------------------------------------------------------===//
class ReplaceEpochOpRewriter : public OpRewritePattern<EpochOp> {
public:
  nlohmann::json component_map;
  std::string component_path;
  std::string tmp_path;

public:
  ReplaceEpochOpRewriter(MLIRContext *context, std::string component_map,
                         std::string component_path, std::string tmp_path)
      : OpRewritePattern<EpochOp>(context),
        component_map(nlohmann::json::parse(component_map)),
        component_path(std::move(component_path)),
        tmp_path(std::move(tmp_path)) {}

private:
  // Function to generate a random string of given length
  std::string get_random_string(size_t length) const {
    const std::string characters =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      result += characters[rand() % characters.size()];
    }
    return result;
  }

public:
  using OpRewritePattern<EpochOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(EpochOp op,
                                PatternRewriter &rewriter) const final {

    tm::TimingModel model;
    tm::Solver solver(tmp_path);
    // Get the EpochOp's ID
    std::string originalIdStr = op.getId().str();
    LOG(INFO) << "Original ID: " << originalIdStr;
    mlir::Region &epochBodyRegion = op.getBody();
    mlir::Block *entryBlock;
    if (epochBodyRegion.empty()) {
      entryBlock = rewriter.createBlock(&epochBodyRegion);
    } else {
      entryBlock = &epochBodyRegion.front();
    }
    // print the contents of the entry block
    llvm::outs() << "Entry block contents:\n";
    std::set<std::string> operation_type_set;
    for (mlir::Operation &child_op : *entryBlock) {
      // get the operation type
      llvm::outs() << "Operation type: " << child_op.getName() << "\n";
      // cast to the correct type
      if (auto rop_op = llvm::dyn_cast<RopOp>(&child_op)) {
        llvm::outs() << "RopOp ID: " << rop_op.getId() << "\n";
        std::string id = rop_op.getId().str();
        uint32_t row = rop_op.getRow();
        uint32_t col = rop_op.getCol();
        uint32_t slot = rop_op.getSlot();
        uint32_t port = rop_op.getPort();

        // get its internal block
        mlir::Region &ropBodyRegion = rop_op.getBody();
        mlir::Block *ropEntryBlock;
        if (ropBodyRegion.empty()) {
          ropEntryBlock = rewriter.createBlock(&ropBodyRegion);
        } else {
          ropEntryBlock = &ropBodyRegion.front();
        }

        nlohmann::json rop_json;
        rop_json["op_name"] = id;
        rop_json["row"] = row;
        rop_json["col"] = col;
        rop_json["slot"] = slot;
        rop_json["port"] = port;
        rop_json["instr_list"] = nlohmann::json::array();
        for (mlir::Operation &rop_child_op : *ropEntryBlock) {
          if (auto instr_op = llvm::dyn_cast<InstrOp>(&rop_child_op)) {
            std::string id = instr_op.getId().str();
            std::string type = instr_op.getType().str();
            std::unordered_map<std::string, std::string> param_map;
            for (auto &param : instr_op.getParam()) {
              std::string param_name = param.getName().str();
              mlir::Attribute param_value = param.getValue();
              if (auto str_attr =
                      llvm::dyn_cast<mlir::StringAttr>(param_value)) {
                param_map[param_name] = str_attr.getValue().str();
              } else if (auto int_attr =
                             llvm::dyn_cast<mlir::IntegerAttr>(param_value)) {
                param_map[param_name] = std::to_string(int_attr.getInt());
              } else {
                llvm::outs()
                    << "Unsupported parameter type in InstrOp: " << param_value
                    << "\n";
                std::exit(-1);
              }
            }

            nlohmann::json instr_json;
            instr_json["name"] = type;
            instr_json["fields"] = nlohmann::json::array();
            for (auto &param : param_map) {
              nlohmann::json param_json;
              param_json["name"] = param.first;
              param_json["value"] = param.second;
              instr_json["fields"].push_back(param_json);
            }
            rop_json["instr_list"].push_back(instr_json);
          } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&rop_child_op)) {
            // DO NOTHING
          } else {
            llvm::outs() << "Illegal operation type in RopOp: "
                         << rop_child_op.getName() << "\n";
            std::exit(-1);
          }
        }

        // output the json to stdout
        llvm::outs() << "Rop JSON: " << rop_json.dump(4) << "\n";

        // create a temporary file in the tmp directory with random name and
        // call the timing model builder to generate the timing model
        // expression.
        std::string random_str = get_random_string(10);
        std::string input_filename = tmp_path + "/" + random_str + ".json";
        std::string output_filename = tmp_path + "/" + random_str + ".txt";
        std::string label = std::to_string(row) + "_" + std::to_string(col) +
                            "_" + std::to_string(slot) + "_" +
                            std::to_string(port);
        if (component_map.find(label) == component_map.end()) {
          llvm::outs() << "Error: Cannot find the component : " << label
                       << "\n";
          std::exit(-1);
        }
        std::string command = component_path + "/resources/" +
                              component_map[label].get<std::string>() +
                              "/timing_model " + input_filename + " " +
                              output_filename;

        llvm::outs() << "Command: " << command << "\n";
        std::ofstream file(input_filename);
        if (!file.is_open()) {
          llvm::outs() << "Error: Failed to create temporary file.\n";
          std::exit(-1);
        }
        file << rop_json.dump(4);
        file.close();
        int result = system(command.c_str());
        if (result != 0) {
          llvm::outs() << "Error: Command failed with error code: " << result
                       << "\n";
          std::exit(-1);
        }

        // read the output file
        std::ifstream output_file(output_filename);
        if (!output_file.is_open()) {
          llvm::outs() << "Error: Failed to open output file.\n";
          std::exit(-1);
        }
        std::string output_str((std::istreambuf_iterator<char>(output_file)),
                               std::istreambuf_iterator<char>());
        output_file.close();

        model.add_operation(tm::Operation(id, output_str));

        // delete the temporary files
        remove(input_filename.c_str());
        remove(output_filename.c_str());

      } else if (auto cop_op = llvm::dyn_cast<CopOp>(&child_op)) {
        llvm::outs() << "CopOp ID: " << cop_op.getId() << "\n";
      } else if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
        if ((operation_type_set.find("pasm.rop") != operation_type_set.end()) ||
            operation_type_set.find("pasm.cop") != operation_type_set.end()) {
          llvm::outs() << "Error: RawOp cannot be used with RopOp or CopOp.\n";
          std::exit(-1);
        }
        return failure();
      } else if (auto constraint_op = llvm::dyn_cast<ConstraintOp>(&child_op)) {
        std::string type = constraint_op.getType().str();
        std::string expr = constraint_op.getExpr().str();
        model.add_constraint(tm::Constraint(type, expr));
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&child_op)) {
        // DO NOTHING
      } else {
        llvm::outs() << "Illegal operation type in EpochOp: "
                     << child_op.getName() << "\n";
        std::exit(-1);
      }
      operation_type_set.insert(child_op.getName().getStringRef().str());
    }
    LOG(DEBUG) << "Timing model: \n" << model.to_string();

    // solve the timing model
    std::unordered_map<std::string, std::string> result = solver.solve(model);
    if (result.empty()) {
      llvm::outs() << "Error: No solution found.\n";
      std::exit(-1);
    }

    // output the result
    llvm::outs() << "Result: \n";
    for (auto it = result.begin(); it != result.end(); ++it) {
      llvm::outs() << it->first << ": " << it->second << "\n";
    }

    return failure();
  }
};
class ReplaceEpochOp : public impl::ReplaceEpochOpBase<ReplaceEpochOp> {
public:
  using impl::ReplaceEpochOpBase<ReplaceEpochOp>::ReplaceEpochOpBase;
  void runOnOperation() override {
    // convert json string to json object
    std::string component_map = this->component_map;
    std::string component_path = this->component_path;
    std::string tmp_path = this->tmp_path;

    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceEpochOpRewriter>(&getContext(), component_map,
                                         component_path, tmp_path);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
//===----------------------------------------------------------------------===//
class ReplaceLoopOpRewriter : public OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern<LoopOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const final {

    int iter = op.getIter();
    auto &body = op.getBody();
    // return failure();
    // Get the block containing the LoopOp itself
    mlir::Block *parentBlock = op.getOperation()->getBlock();

    if (parentBlock) {
      llvm::outs() << "LoopOp is in block: " << parentBlock << "\n";
      std::string originalIdStr = op.getId().str();
      std::string newId = "epoch_derived_from_loop_" + originalIdStr;
      mlir::StringAttr newIdAttr = rewriter.getStringAttr(newId);
      EpochOp epoch_op = rewriter.create<EpochOp>(op.getLoc(), newIdAttr);
      mlir::Region &epochBodyRegion = epoch_op.getBody();
      mlir::Block *entryBlock;
      if (epochBodyRegion.empty()) {
        entryBlock = rewriter.createBlock(&epochBodyRegion);
      } else {
        entryBlock = &epochBodyRegion.front();
      }
      rewriter.setInsertionPointToEnd(entryBlock);
      // Move the operations from the original loop body's blocks
      // into the new epoch's entry block, before the terminator.
      for (mlir::Block &block :
           body) { // Iterate through blocks in the LoopOp's region
        for (mlir::Operation &child_op :
             block) { // Iterate ops excluding the terminator
          // Clone the operation at the current insertion point
          rewriter.clone(child_op);
        }
        // Note: This simple loop assumes a single block in the LoopOp body
        // and doesn't handle block arguments or complex control flow
        // transfer. If LoopOp body can have multiple blocks or branches, a
        // more sophisticated merging/cloning logic (like inlineRegionBefore)
        // is needed.
      }
      rewriter.replaceOp(op, epoch_op);

    } else {
      // This usually means the op is top-level (like a FuncOp)
      llvm::outs() << "LoopOp is not inside a block.\n";
    }

    // create a new
    return failure();
  }
};

class ReplaceLoopOp : public impl::ReplaceLoopOpBase<ReplaceLoopOp> {
public:
  using impl::ReplaceLoopOpBase<ReplaceLoopOp>::ReplaceLoopOpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceLoopOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace vesyla::pasm