#include "ScheduleEpochPassDetail.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <bitset>

namespace vesyla::pasm::schedule_epoch_detail {

std::map<int, std::vector<int>>
ScheduleEpochPassRewriter::create_reg_alloc_table() const {
  std::map<int, std::vector<int>> reg_alloc_table;
  for (int reg_i = 0; reg_i < SCALAR_REGISTER_COUNT; reg_i++) {
    reg_alloc_table[reg_i] = std::vector<int>();
  }

  return reg_alloc_table;
}

void ScheduleEpochPassRewriter::changeActMode2Param(
    ::mlir::Operation *act_op, int first_reg_address,
    ::mlir::PatternRewriter &rewriter) const {
  ::mlir::DictionaryAttr new_instr_params = rewriter.getDictionaryAttr({
      rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(2)),
      rewriter.getNamedAttr("param",
                            rewriter.getI32IntegerAttr(first_reg_address)),
      rewriter.getNamedAttr("ports", rewriter.getI64IntegerAttr(0)),
  });
  act_op->setAttr("param", new_instr_params);
}

int ScheduleEpochPassRewriter::allocateAndScheduleActMode2PrepInstructions(
    std::map<int, std::vector<int>> &reg_alloc_table,
    std::map<int, ::mlir::Operation *> &cell_time_table,
    ::mlir::PatternRewriter &rewriter, ::vesyla::pasm::EpochOp op, int cycle,
    uint64_t ports) const {

  const int num_regs_needed = 4; // TODO: make this configurable
  const int max_attempts = 1000;

  // A group holds the mask from its first prep instruction to its act, which
  // reads it. Acts come in cycle order, so a group is free again for prep
  // instructions after the last cycle it was used.
  for (int first_reg : {4, 4 + num_regs_needed}) {
    const std::vector<int> &used = reg_alloc_table[first_reg];
    int after = used.empty() ? INT_MIN : *std::max_element(used.begin(), used.end());

    auto instrs_to_place = get_act_mode2_prep_instrs(first_reg, ports);
    // A reused group still holds the last mask: zero what this one leaves 0.
    if (!used.empty()) {
      for (int i = num_regs_needed - 1; i >= 0; --i) {
        if (((ports >> (16 * i)) & 0xFFFF) == 0) {
          instrs_to_place.insert(instrs_to_place.begin(),
                                 {{"mode", 1},
                                  {"operand1", 0},
                                  {"operand2_sd", 0},
                                  {"operand2", 0},
                                  {"result", first_reg + i}});
        }
      }
    }
    std::vector<int> slots;
    for (int c = cycle - 1;
         c > after && c >= cycle - max_attempts &&
         slots.size() < instrs_to_place.size();
         --c) {
      if (cell_time_table.find(c) == cell_time_table.end()) {
        slots.push_back(c);
      }
    }
    if (slots.size() < instrs_to_place.size()) {
      continue;
    }

    for (int i = 0; i < num_regs_needed; i++) {
      reg_alloc_table[first_reg + i].push_back(cycle);
      for (int c : slots) {
        reg_alloc_table[first_reg + i].push_back(c);
      }
    }
    // The latest slot takes the last instruction, as before.
    for (size_t k = 0; k < slots.size(); ++k) {
      llvm::outs() << "Allocating registers r" << first_reg << " to r"
                   << (first_reg + num_regs_needed - 1)
                   << " for ACT mode 2 prep instructions for ports "
                   << std::bitset<64>(ports).to_string() << " at cycle "
                   << slots[k] << "\n";
      auto prep_op = compose_calc_mlir_op(
          op, rewriter, instrs_to_place[instrs_to_place.size() - 1 - k]);
      create_time_table_entry(cell_time_table, slots[k], prep_op);
    }
    return first_reg;
  }

  llvm::outs() << "Error: Cannot allocate registers for activation mode 2 at "
                  "cycle "
               << cycle
               << ". [r4->r7] and [r8->r11] are both still in use.\n";
  exit(EXIT_FAILURE);
}

std::vector<std::unordered_map<std::string, int>>
ScheduleEpochPassRewriter::get_act_mode2_prep_instrs(int first_reg_address,
                                                     uint64_t ports) const {
  std::vector<std::unordered_map<std::string, int>> act_mode2_prep_instrs;

  int num_regs_needed = 4;
  for (int loop_index = 1; loop_index <= 4; ++loop_index) {
    uint64_t regValueShouldBe = ports >> (16 * (loop_index - 1)) & 0xFFFF;
    int reg_i = first_reg_address + (loop_index - 1);

    llvm::outs() << "r" << reg_i << " should be: "
                 << std::bitset<16>(regValueShouldBe).to_string() << "\n";

    int msb_shift = ((16 * loop_index) - 8); // 8, 24, 40, 56
    int lsb_shift = (16 * (loop_index - 1)); // 0, 16, 32, 48

    llvm::outs() << "reg_i = " << reg_i << ", msb_shift = " << msb_shift
                 << ", lsb_shift = " << lsb_shift << "\n";

    llvm::outs() << "operand2: "
                 << static_cast<int>((ports >> msb_shift) & 0xFF) << ", "
                 << static_cast<int>((ports >> lsb_shift) & 0xFF) << "\n";

    uint64_t msb_value = static_cast<uint64_t>((ports >> msb_shift) & 0xFF);
    uint64_t lsb_value = static_cast<uint64_t>((ports >> lsb_shift) & 0xFF);

    int add_lsb_reg_address = reg_i;
    if (msb_value != 0) {
      // set reg_i 8 MSB
      std::unordered_map<std::string, int> load_msb = {
          {"mode", 23},                       // addh
          {"operand1", 0},                    // reg 0
          {"operand2_sd", 0},                 // use immediate
          {"operand2", (int)msb_value},       // 8 MSBs
          {"result", reg_i}                   // store in reg_i
      };
      act_mode2_prep_instrs.push_back(load_msb);
    } else {
      add_lsb_reg_address = 0; // r0 is always 0
    }

    if (lsb_value != 0) {
      std::unordered_map<std::string, int> load_lsb = {
          {"mode", 1},                          // add
          {"operand1", add_lsb_reg_address},    // reg_i if msb is not 0, else r0
          {"operand2_sd", 0},                   // use immediate
          {"operand2", (int)lsb_value},         // 8 LSBs
          {"result", reg_i}                     // store in reg_i
      };
      act_mode2_prep_instrs.push_back(load_lsb);
    }
  }

  return act_mode2_prep_instrs;
}

bool ScheduleEpochPassRewriter::op_is_act_mode2(::mlir::Operation *op) const {
  auto instr_type = op->getAttr("type");
  auto instr_type_str = ::mlir::dyn_cast_or_null<::mlir::StringAttr>(instr_type);
  if (!instr_type_str || instr_type_str.getValue() != "act") {
    return false;
  }

  auto instr_params = op->getAttr("param");
  auto instr_params_dict =
      ::mlir::dyn_cast_or_null<::mlir::DictionaryAttr>(instr_params);
  if (!instr_params_dict) {
    return false;
  }

  auto mode_attr = instr_params_dict.get("mode");
  auto mode_attr_int =
      ::mlir::dyn_cast_or_null<::mlir::IntegerAttr>(mode_attr);
  if (!mode_attr_int || mode_attr_int.getInt() != 2) {
    return false;
  }

  return true;
}

uint64_t ScheduleEpochPassRewriter::get_ports_from_act_mode2_instr(
    ::mlir::Operation *op) const {
  auto instr_params = op->getAttr("param");
  auto instr_params_dict =
      ::mlir::dyn_cast_or_null<::mlir::DictionaryAttr>(instr_params);
  if (!instr_params_dict) {
    llvm::outs() << "Error: Invalid parameters for act mode 2 instruction.\n";
    std::exit(EXIT_FAILURE);
  }

  auto ports_attr = instr_params_dict.get("ports");
  auto ports_attr_int =
      ::mlir::dyn_cast_or_null<::mlir::IntegerAttr>(ports_attr);
  if (!ports_attr_int) {
    llvm::outs() << "Error: Invalid ports attribute for act mode 2 "
                    "instruction.\n";
    std::exit(EXIT_FAILURE);
  }

  return ports_attr_int.getInt();
}

} // namespace vesyla::pasm::schedule_epoch_detail
