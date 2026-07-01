#include "vesyla/Dialect/Pasm/Transforms/ScheduleEpochPass.hpp"
#include "ScheduleEpochPassDetail.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vesyla/Analysis/TimingModel/Solver.hpp"
#include "vesyla/Analysis/TimingModel/TimingModel.hpp"
#include "vesyla/Support/Config.hpp"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace vesyla::pasm {
#define GEN_PASS_DEF_SCHEDULEEPOCHPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"
} // namespace vesyla::pasm

namespace vesyla::pasm::schedule_epoch_detail {

// Defined in TimetableDump.cpp. Writes the resolved per-epoch schedule to a
// JSON timetable under the configured directory for visualization. The
// min_shift_time (from synchronize) is added so absolute cycles match the
// emitted code.
void dump_schedule_table(
    const std::string &tmp_path, const std::string &epoch_id,
    const std::unordered_map<std::string, int> &schedule_table,
    const std::unordered_map<std::string, std::string> &result,
    const ::vesyla::tm::TimingModel &model, int min_shift_time,
    const std::vector<CtrlInstr> &controller_instrs);

::mlir::LogicalResult ScheduleEpochPassRewriter::matchAndRewrite(
    ::vesyla::pasm::EpochOp op, ::mlir::PatternRewriter &rewriter) const {

  struct OpExprTuple {
    std::string id;
    std::string kind;
    int row;
    int col;
    int slot;
    int port;
    std::string expr;
  };
  std::vector<OpExprTuple> op_exprs;
  ::vesyla::tm::TimingModel model;
  ::vesyla::tm::Solver solver(tmp_path);

  std::string originalIdStr = op.getId().str();
  ::mlir::Block *entryBlock = getEpochBodyEntryBlock(op, rewriter);
  std::set<std::string> operation_type_set;
  for (::mlir::Operation &child_op : *entryBlock) {
    if (auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(&child_op)) {

      nlohmann::json rop_json = op2json(rop_op.getOperation());

      std::string random_str = ::vesyla::util::Common::gen_random_string(10);
      LOG_DEBUG << "Random string for temporary file: " << random_str;
      std::string input_filename = tmp_path + "/" + random_str + ".json";
      std::string output_filename = tmp_path + "/" + random_str + "_out.txt";
      LOG_DEBUG << "Input filename: " << input_filename;
      LOG_DEBUG << "Output filename: " << output_filename;
      std::string label = std::to_string(rop_json["row"].get<int>()) + "_" +
                          std::to_string(rop_json["col"].get<int>()) + "_" +
                          std::to_string(rop_json["slot"].get<int>());
      LOG_DEBUG << "Label: " << label;
      if (component_map.find(label) == component_map.end()) {
        llvm::outs() << "Error: Cannot find the component : " << label << "\n";
        std::exit(EXIT_FAILURE);
      }
      ::vesyla::pasm::Config cfg;
      std::string command = component_path + "/resources/" +
                            component_map[label].get<std::string>() + "/" +
                            cfg.tool_name("compile_util") +
                            " get_timing_model " + input_filename + " " +
                            output_filename;

      llvm::outs() << "Executing command: " << command << "\n";

      std::ofstream file(input_filename);
      if (!file.is_open()) {
        llvm::outs() << "Error: Failed to create temporary file.\n";
        std::exit(EXIT_FAILURE);
      }
      file << rop_json.dump(4);
      file.close();
      int result = system(command.c_str());
      if (result != 0) {
        llvm::outs() << "Error: Command failed with error code: " << result
                     << "\n";
        std::exit(EXIT_FAILURE);
      }

      std::ifstream output_file(output_filename);
      if (!output_file.is_open()) {
        llvm::outs() << "Error: Failed to open output file.\n";
        std::exit(EXIT_FAILURE);
      }
      std::string output_str((std::istreambuf_iterator<char>(output_file)),
                             std::istreambuf_iterator<char>());
      output_file.close();

      ::vesyla::tm::Operation operation = ::vesyla::tm::Operation(
          rop_json["id"].get<std::string>(), output_str);
      operation.col = rop_json["col"].get<int>();
      operation.row = rop_json["row"].get<int>();
      operation.slot = rop_json["slot"].get<int>();
      operation.port = rop_json["port"].get<int>();
      model.add_operation(operation);

      std::filesystem::remove(input_filename);
      std::filesystem::remove(output_filename);

      op_exprs.push_back(OpExprTuple{
          rop_json["id"].get<std::string>(),
          rop_json["kind"].get<std::string>(), rop_json["row"].get<int>(),
          rop_json["col"].get<int>(), rop_json["slot"].get<int>(),
          rop_json["port"].get<int>(), output_str});

    } else if (auto cop_op = llvm::dyn_cast<::vesyla::pasm::CopOp>(&child_op)) {
      llvm::outs() << "CopOp ID: " << cop_op.getId() << "\n";
    } else if (auto raw_op = llvm::dyn_cast<::vesyla::pasm::RawOp>(&child_op)) {
      if ((operation_type_set.find("pasm.rop") != operation_type_set.end()) ||
          operation_type_set.find("pasm.cop") != operation_type_set.end()) {
        llvm::outs() << "Error: RawOp cannot be used with RopOp or CopOp.\n";
        std::exit(EXIT_FAILURE);
      }
      return ::mlir::failure();
    } else if (auto cstr_op =
                   llvm::dyn_cast<::vesyla::pasm::CstrOp>(&child_op)) {
      for (auto &c : ::vesyla::tm::Constraint::from_cstr_op(cstr_op)) {
        model.add_constraint(c);
      }
    } else if (auto yield_op =
                   llvm::dyn_cast<::vesyla::pasm::YieldOp>(&child_op)) {
      // DO NOTHING
    } else {
      llvm::outs() << "Illegal operation type in EpochOp: "
                   << child_op.getName() << "\n";
      std::exit(EXIT_FAILURE);
    }
    operation_type_set.insert(child_op.getName().getStringRef().str());
  }

  // add built-in constraints
  std::unordered_map<std::string, std::vector<std::string>> all_resource_op;
  struct CopAnchorRef {
    std::string op_name;
    ::vesyla::tm::Constraint::Anchor anchor;
  };
  std::unordered_map<std::string, std::vector<CopAnchorRef>>
      all_control_op_anchors;
  for (auto &op_expr : op_exprs) {
    if (op_expr.kind == "rop") {
      std::string label =
          std::to_string(op_expr.row) + "_" + std::to_string(op_expr.col);
      if (all_resource_op.find(label) == all_resource_op.end()) {
        all_resource_op[label] = std::vector<std::string>();
      }
      all_resource_op[label].push_back(op_expr.id);
    } else if (op_expr.kind == "cop") {
      std::string label =
          std::to_string(op_expr.row) + "_" + std::to_string(op_expr.col);
      // OperationExpr::get_all_anchors yields (event_id, idx) pairs with
      // indices in innermost-first order, matching tm::Constraint::Anchor.
      auto cop_op_info = model.get_operation(op_expr.id);
      for (auto &p : cop_op_info.expr.get_all_anchors()) {
        CopAnchorRef ref;
        ref.op_name = op_expr.id;
        ref.anchor.event_id = p.first;
        ref.anchor.idx = p.second;
        all_control_op_anchors[label].push_back(std::move(ref));
      }
    } else if (op_expr.kind == "raw") {
      // DO NOTHING
    } else if (op_expr.kind == "cstr") {
      // DO NOTHING
    } else if (op_expr.kind == "yield") {
      // DO NOTHING
    } else {
      llvm::outs() << "Error: Illegal operation kind in EpochOp: "
                   << op_expr.kind << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  for (auto cell : all_resource_op) {
    std::string label = cell.first;
    std::vector<std::string> ops = cell.second;
    if (ops.size() > 1) {
      for (size_t i = 0; i < ops.size(); i++) {
        int slot = -1;
        int port = -1;

        for (auto &op : op_exprs) {
          if (op.id == ops[i]) {
            slot = op.slot;
            port = op.port;
            break;
          }
        }

        if (all_control_op_anchors.find(label) !=
            all_control_op_anchors.end()) {
          for (auto &ref : all_control_op_anchors[label]) {
            ::vesyla::tm::Constraint c(/*src_id=*/ops[i],
                                       /*dst_id=*/ref.op_name,
                                       /*min_delay=*/0, /*max_delay=*/0,
                                       /*src_anchor=*/std::nullopt,
                                       /*dst_anchor=*/ref.anchor);
            c.is_neq = true;
            c.kind = "linear";
            model.add_constraint(std::move(c));
          }
        }
      }
    }
  }

  llvm::outs().flush();

  std::unordered_map<std::string, std::string> result = solver.solve(model);
  if (result.empty()) {
    llvm::outs() << "Error: No solution found.\n";
    std::exit(EXIT_FAILURE);
  }

  std::unordered_map<std::string, int> schedule_table;
  for (auto it = result.begin(); it != result.end(); ++it) {
    std::string key = it->first;
    std::string value = it->second;

    if (key == "use_act_mode_0") {
      llvm::outs() << "Using ACT mode 0: " << value << "\n";
      continue;
    }
    if (key == "use_act_mode_1") {
      llvm::outs() << "Using ACT mode 1: " << value << "\n";
      continue;
    }

    if (value[0] != '[') {
      schedule_table[key] = std::stoi(value);
    }
  }

  replace_time_in_instr_param(op, schedule_table, rewriter);
  reshape_instr(op, rewriter);
  int min_shift_time = synchronize(op, schedule_table, rewriter, allow_unsafe);

  // After synchronize the epoch body is just one pasm.raw block per cell, each
  // holding the ordered controller instruction stream that becomes the .asm.
  // Walk them to recover what every controller issues and over which cycles.
  // Each instruction spans [cycle, cycle + advance): a wait(N) takes N+1 cycles
  // (1 issue + N stall) and every other instruction one, so the issue clock
  // advances by that amount and following instructions stay on their true
  // absolute cycle (matching the ROP starts).
  std::vector<CtrlInstr> controller_instrs;
  for (::mlir::Operation &child_op : op.getBody().front()) {
    auto raw_op = llvm::dyn_cast<::vesyla::pasm::RawOp>(&child_op);
    if (!raw_op) {
      continue;
    }
    int row = static_cast<int>(raw_op.getRow());
    int col = static_cast<int>(raw_op.getCol());
    ::mlir::Region &raw_body = raw_op.getBody();
    if (raw_body.empty()) {
      continue;
    }
    int cycle = 0;
    for (::mlir::Operation &raw_child : raw_body.front()) {
      auto instr_op = llvm::dyn_cast<::vesyla::pasm::InstrOp>(&raw_child);
      if (!instr_op) {
        continue;
      }
      std::string type = instr_op.getType().str();
      ::mlir::DictionaryAttr params = instr_op.getParam();

      // Read an i32 param, returning `dflt` if absent.
      auto read_param = [&](const char *key, int dflt) -> int {
        if (params && params.contains(key)) {
          if (auto int_attr =
                  llvm::dyn_cast<::mlir::IntegerAttr>(params.get(key))) {
            return static_cast<int>(int_attr.getInt());
          }
        }
        return dflt;
      };

      int advance = 1;
      if (type == "wait") {
        advance = read_param("cycle", 0) + 1;
      }

      // Slot-targeting control instructions (evt/conf/rep/trans, ...) carry the
      // slot (and port) they act on; act/wait/calc do not. Read generically so
      // any such instruction echoes onto its target resource and nothing has to
      // be enumerated by name. Absent -> -1 (no target).
      int slot = read_param("slot", -1);
      int port = read_param("port", -1);

      controller_instrs.push_back(
          CtrlInstr{row, col, cycle, cycle + advance, slot, port, type});
      cycle += advance;
    }
  }

  dump_schedule_table(tmp_path, originalIdStr, schedule_table, result, model,
                      min_shift_time, controller_instrs);

  return ::mlir::success();
}

} // namespace vesyla::pasm::schedule_epoch_detail

namespace vesyla::pasm {
namespace {

class ScheduleEpochPass
    : public impl::ScheduleEpochPassBase<ScheduleEpochPass> {
public:
  using impl::ScheduleEpochPassBase<ScheduleEpochPass>::ScheduleEpochPassBase;
  void runOnOperation() override {
    ::vesyla::pasm::Config cfg;
    nlohmann::json component_map_json = cfg.get_component_map_json();
    std::string component_path = this->component_path;
    std::string tmp_path = this->tmp_path;
    bool allow_unsafe = this->allow_unsafe;
    nlohmann::json arch_json = cfg.get_arch_json();
    int row = arch_json["parameters"]["ROWS"].get<int>();
    int col = arch_json["parameters"]["COLS"].get<int>();

    ::mlir::RewritePatternSet patterns(&getContext());
    patterns.add<schedule_epoch_detail::ScheduleEpochPassRewriter>(
        &getContext(), component_map_json, component_path, tmp_path,
        allow_unsafe, row, col);
    ::mlir::FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vesyla::pasm
