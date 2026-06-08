#ifndef VESYLA_PASM_SCHEDULE_EPOCH_DETAIL_HPP
#define VESYLA_PASM_SCHEDULE_EPOCH_DETAIL_HPP

#include "mlir/IR/PatternMatch.h"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Support/Common.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#define SCALAR_REGISTER_COUNT 16

namespace vesyla::pasm::schedule_epoch_detail {

class ScheduleEpochPassRewriter
    : public ::mlir::OpRewritePattern<::vesyla::pasm::EpochOp> {
public:
  nlohmann::json component_map;
  std::string component_path;
  std::string tmp_path;
  bool allow_unsafe;
  int _row;
  int _col;

  ScheduleEpochPassRewriter(::mlir::MLIRContext *context,
                            nlohmann::json component_map,
                            std::string component_path, std::string tmp_path,
                            bool allow_unsafe, int row_, int col_)
      : ::mlir::OpRewritePattern<::vesyla::pasm::EpochOp>(context),
        component_map(component_map),
        component_path(std::move(component_path)),
        tmp_path(std::move(tmp_path)), allow_unsafe(allow_unsafe), _row(row_),
        _col(col_) {}

  // ScheduleEpochPass.cpp
  ::mlir::LogicalResult
  matchAndRewrite(::vesyla::pasm::EpochOp op,
                  ::mlir::PatternRewriter &rewriter) const final;

  // TimeTable.cpp
  ::mlir::Block *
  getOrCreateEntryBlock(::mlir::Region &region,
                        ::mlir::PatternRewriter &rewriter) const;
  ::mlir::Block *
  getEpochBodyEntryBlock(::vesyla::pasm::EpochOp epoch_op,
                         ::mlir::PatternRewriter &rewriter) const;
  std::map<int, ::mlir::Operation *> &getOrCreateCellTimeTable(
      std::map<std::string, std::map<int, ::mlir::Operation *>> &time_table,
      const std::string &label) const;
  void create_time_table_entry(
      std::map<int, ::mlir::Operation *> &cell_time_table, int t,
      ::vesyla::pasm::InstrOp &instr_op) const;
  std::map<std::string, std::vector<::mlir::Operation *>> get_rop_ops_for_cycle(
      int currentCycle,
      std::unordered_map<::mlir::Operation *, int> time_table_rop) const;
  void replace_time_in_instr_param(
      ::vesyla::pasm::EpochOp &op,
      std::unordered_map<std::string, int> &schedule_table,
      ::mlir::PatternRewriter &rewriter) const;
  void print_time_table(
      std::map<std::string, std::map<int, ::mlir::Operation *>> &time_table)
      const;

  // InstrFactory.cpp
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_0_instr(std::vector<int> indices) const;
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_1_instr(std::vector<int> indices) const;
  std::optional<std::unordered_map<std::string, uint64_t>>
  create_act_2_instr(std::vector<int> indices) const;
  std::unordered_map<std::string, uint64_t>
  create_act_instr(std::vector<int> indices) const;
  std::unordered_map<std::string, int> create_wait_instr(int cycle) const;
  ::vesyla::pasm::InstrOp compose_act_mlir_op(
      ::vesyla::pasm::EpochOp op, ::mlir::PatternRewriter &rewriter,
      std::unordered_map<std::string, uint64_t> &act_instr_param_map) const;
  ::vesyla::pasm::InstrOp compose_calc_mlir_op(
      ::vesyla::pasm::EpochOp op, ::mlir::PatternRewriter &rewriter,
      std::unordered_map<std::string, int> &calc_instr_param_map) const;
  std::vector<int>
  get_absolute_port_indices(std::vector<::mlir::Operation *> &rop_ops) const;

  // RegisterAllocator.cpp
  std::map<int, std::vector<int>> create_reg_alloc_table() const;
  void changeActMode2Param(::mlir::Operation *act_op, int first_reg_address,
                           ::mlir::PatternRewriter &rewriter) const;
  int allocateAndScheduleActMode2PrepInstructions(
      std::map<int, std::vector<int>> &reg_alloc_table,
      std::map<int, ::mlir::Operation *> &cell_time_table,
      ::mlir::PatternRewriter &rewriter, ::vesyla::pasm::EpochOp op, int cycle,
      uint64_t ports) const;
  std::vector<std::unordered_map<std::string, int>>
  get_act_mode2_prep_instrs(int first_reg_address, uint64_t ports) const;
  bool op_is_act_mode2(::mlir::Operation *op) const;
  uint64_t get_ports_from_act_mode2_instr(::mlir::Operation *op) const;

  // JsonOpBridge.cpp
  nlohmann::json op2json(::mlir::Operation *op) const;
  void json2op(nlohmann::json op_json,
               ::mlir::PatternRewriter &rewriter) const;

  // Synchronize.cpp
  void
  insert_rop_instructions(std::vector<::mlir::Operation *> &rop_ops, int t,
                          ::mlir::PatternRewriter &rewriter,
                          std::map<int, ::mlir::Operation *> &cell_time_table,
                          bool allow_unsafe = false) const;
  void insert_cop_instructions(
      ::mlir::Block *copEntryBlock,
      const std::unordered_map<std::string, int> &schedule_table,
      std::map<int, ::mlir::Operation *> &cell_time_table) const;
  void synchronize(::vesyla::pasm::EpochOp &op,
                   std::unordered_map<std::string, int> &schedule_table,
                   ::mlir::PatternRewriter &rewriter,
                   bool allow_unsafe = false) const;

  // ExternalReshape.cpp
  void reshape_instr(::vesyla::pasm::EpochOp &op,
                     ::mlir::PatternRewriter &rewriter) const;
};

} // namespace vesyla::pasm::schedule_epoch_detail

#endif // VESYLA_PASM_SCHEDULE_EPOCH_DETAIL_HPP
