//===----------------------------------------------------------------------===//
// ScheduleEpochPass — documentation surface
//
// This header is a doc-only mirror of the pass classes defined inside the
// anonymous namespace in ScheduleEpochPass.cpp. Nothing else links against
// these declarations — they exist so each method has a stable place for a
// documentation comment. The active class definitions remain in the .cpp;
// when changing a signature, update both.
//
// See the long block at the top of ScheduleEpochPass.cpp (or the pipeline
// overview comments below) for what the pass does end-to-end.
//===----------------------------------------------------------------------===//

#ifndef __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__
#define __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__

#include "Config.hpp"
#include "Passes.hpp"

#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include "util/Common.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vesyla::pasm {

//===----------------------------------------------------------------------===//
// ScheduleEpochPassRewriter
//
// OpRewritePattern fired once per `pasm.epoch` op by the greedy driver. Walks
// the epoch body, builds a tm::TimingModel out of every ROP/COP/CSTR child,
// invokes the MiniZinc solver, then materializes the resulting schedule as a
// flat sequence of `pasm.instr` ops in cycle order and erases the originals.
//===----------------------------------------------------------------------===//
class ScheduleEpochPassRewriter
    : public mlir::OpRewritePattern<vesyla::pasm::EpochOp> {
public:
  // Configuration captured at construction time and reused for every epoch.
  nlohmann::json component_map;
  std::string component_path;
  std::string tmp_path;
  bool allow_unsafe;
  int _row;
  int _col;

  // Constructor. The trailing row/col are the architecture's grid dimensions
  // (used by some of the activation-mode helpers).
  ScheduleEpochPassRewriter(mlir::MLIRContext *context,
                            nlohmann::json component_map,
                            std::string component_path, std::string tmp_path,
                            bool allow_unsafe, int row_, int col_);

  // Pattern entry point. Called by the greedy driver on every EpochOp; runs
  // the full synchronize → schedule → materialize pipeline.
  mlir::LogicalResult
  matchAndRewrite(vesyla::pasm::EpochOp op,
                  mlir::PatternRewriter &rewriter) const final;

private:
  // Returns the entry block of `region`, creating an empty one if the region
  // has none yet.
  mlir::Block *getOrCreateEntryBlock(mlir::Region &region,
                                     mlir::PatternRewriter &rewriter) const;

  // Convenience wrapper around getOrCreateEntryBlock for an EpochOp's body.
  // Aborts the process if `epoch_op` is null.
  mlir::Block *getEpochBodyEntryBlock(vesyla::pasm::EpochOp epoch_op,
                                      mlir::PatternRewriter &rewriter) const;

  //===--------------------------------------------------------------------===//
  // Activation-instruction strategy builders
  //
  // Three packing strategies for combining multiple per-port activations into
  // a single hardware `act` instruction. Each returns the encoded parameter
  // map on success, or `nullopt` if the strategy can't represent the inputs.
  //===--------------------------------------------------------------------===//

  // Strategy 0: all indices fit within a 4-slot window.
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_0_instr(std::vector<int> indices) const;

  // Strategy 1: each (slot, port) gets a dedicated bit-position; succeeds
  // when all referenced ports share a common port mask.
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_1_instr(std::vector<int> indices) const;

  // Strategy 2: register-allocated activation. Falls back to using a control
  // register to fan out the activation across slots that don't fit 0 or 1.
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_2_instr(std::vector<int> indices) const;

  // Dispatcher: tries 0, then 1, then 2 in order. Aborts if all fail.
  std::unordered_map<std::string, uint64_t>
  create_act_instr(std::vector<int> indices) const;

  // Builds the parameter map for a `wait` instruction inserted to span an
  // idle cycle gap of length `cycle`.
  std::unordered_map<std::string, int> create_wait_instr(int cycle) const;

  //===--------------------------------------------------------------------===//
  // JSON / MLIR adapters
  //
  // Used to ferry RopOp / CopOp definitions across the process boundary into
  // the per-component Rust `compile_util` binary and back.
  //===--------------------------------------------------------------------===//

  // Serializes a RopOp or CopOp (with its instr body) to the JSON shape the
  // Rust compile_util expects.
  nlohmann::json op2json(mlir::Operation *op) const;

  // Inverse of op2json. Materializes a RopOp or CopOp from the JSON the Rust
  // tool returns, inserting it via `rewriter` at the current insertion point.
  void json2op(nlohmann::json op_json,
               mlir::PatternRewriter &rewriter) const;

  //===--------------------------------------------------------------------===//
  // Cycle resolution / instruction reshape
  //===--------------------------------------------------------------------===//

  // Walks every `instr` in the epoch and substitutes symbolic time references
  // (e.g. delay variables like `t4`) in the param dictionaries with the
  // concrete cycles in `schedule_table`.
  void replace_time_in_instr_param(
      vesyla::pasm::EpochOp &op,
      std::unordered_map<std::string, int> &schedule_table,
      mlir::PatternRewriter &rewriter) const;

  // For each Rop/Cop, shells out to that component's `reshape_instr` Rust
  // binary, which splits instructions that overflow encoding bitwidths (e.g.
  // `rep` → `rep + repx`) and may rewrite the body in component-specific ways.
  void reshape_instr(vesyla::pasm::EpochOp &op,
                     mlir::PatternRewriter &rewriter) const;

  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//

  // Pretty-prints the time table (cell label → cycle → op) to llvm::outs.
  // Debug aid only — not consumed by anything in the production path.
  void print_time_table(
      std::map<std::string,
               std::map<int, mlir::Operation *>> &time_table) const;

  //===--------------------------------------------------------------------===//
  // Activation-mode 2 (register-allocated) helpers
  //===--------------------------------------------------------------------===//

  // Flattens each rop's (slot, port) into a linear 0..63 index. Order matches
  // `rop_ops`.
  std::vector<int>
  get_absolute_port_indices(std::vector<mlir::Operation *> &rop_ops) const;

  // Picks which control register each act-mode-2 instruction is allocated
  // into. Returns map (cycle → list of register indices).
  std::map<int, std::vector<int>> create_reg_alloc_table() const;

  // Rewrites an act op's params to reference the assigned register at
  // `first_reg_address` once register allocation has happened.
  void changeActMode2Param(mlir::Operation *act_op, int first_reg_address,
                           mlir::PatternRewriter &rewriter) const;

  // Returns the pre-load (`prop`) instructions that need to fire before an
  // act-mode-2 activation can use registers `first_reg_address` onward.
  std::vector<std::unordered_map<std::string, int>>
  get_act_mode2_prep_instrs(int first_reg_address, uint64_t ports) const;

  // True iff this instruction is an `act` op already in mode 2.
  bool op_is_act_mode2(mlir::Operation *op) const;

  // Reads the bitmask of ports referenced by an act-mode-2 op.
  uint64_t get_ports_from_act_mode2_instr(mlir::Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Materialization helpers (called by `synchronize`)
  //===--------------------------------------------------------------------===//

  // Folds a cop's body instructions back into the parent epoch at the cycles
  // assigned by the solver.
  void insert_cop_instructions(
      mlir::Block *copEntryBlock,
      const std::unordered_map<std::string, int> &schedule_table,
      std::map<int, mlir::Operation *> &cell_time_table) const;

  // Emits the `act`/`prop`/etc. instructions for the rop_ops that all start at
  // cycle `t`. `cell_time_table` is updated to track what already lives at
  // each cycle in this cell.
  void insert_rop_instructions(
      std::vector<mlir::Operation *> &rop_ops, int t,
      mlir::PatternRewriter &rewriter,
      std::map<int, mlir::Operation *> &cell_time_table,
      bool allow_unsafe = false) const;

  //===--------------------------------------------------------------------===//
  // synchronize — the heart of the pass
  //
  // Builds the tm::TimingModel from the epoch body, runs the solver, then
  // emits act/wait/prep instructions in cycle order. `schedule_table` is
  // filled with the (mzn-var → cycle) mapping returned by MiniZinc.
  //===--------------------------------------------------------------------===//
  void synchronize(vesyla::pasm::EpochOp &op,
                   std::unordered_map<std::string, int> &schedule_table,
                   mlir::PatternRewriter &rewriter,
                   bool allow_unsafe = false) const;
};

// The `ScheduleEpochPass` class itself inherits from the TableGen-generated
// `impl::ScheduleEpochPassBase<>` template, which is only available in
// translation units that `#define GEN_PASS_DEF_SCHEDULEEPOCHPASS` before
// including "pasm/Passes.hpp.inc". It therefore lives in the cpp's anonymous
// namespace; see Passes.td for the canonical pass-level summary / options /
// description.

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__
