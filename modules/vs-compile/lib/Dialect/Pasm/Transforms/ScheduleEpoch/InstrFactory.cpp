#include "ScheduleEpochPassDetail.hpp"
#include "llvm/Support/raw_ostream.h"
#include <bitset>

namespace vesyla::pasm::schedule_epoch_detail {

std::optional<std::unordered_map<std::string, uint64_t>>
ScheduleEpochPassRewriter::create_act_0_instr(std::vector<int> indices) const {
  int min_index = *std::min_element(indices.begin(), indices.end());
  int max_index = *std::max_element(indices.begin(), indices.end());
  int min_slot = min_index / 4;
  int max_slot = max_index / 4;

  if (max_slot - min_slot >= 4) {
    llvm::outs() << "ACT mode 0 failed: slot indices are too far apart.\n";
    return std::nullopt;
  }

  int ports = 0;
  for (auto index : indices) {
    ports |= (1 << (index - min_slot * 4));
  }
  return std::make_optional<std::unordered_map<std::string, uint64_t>>(
      {{"mode", 0}, {"param", min_slot}, {"ports", ports}});
}

std::optional<std::unordered_map<std::string, uint64_t>>
ScheduleEpochPassRewriter::create_act_1_instr(std::vector<int> indices) const {
  int ports = 0;
  std::vector<int> param_vec(16, 0);

  // compose the param_vec using the port indices
  for (auto index : indices) {
    int slot = index / 4;
    int port = index % 4;
    ports |= (1 << slot);           // apply 1 to the slot index in ports
    param_vec[slot] |= (1 << port); // apply 1 to the port index in param
  }

  // it's valid only if param_vec value is either 0 or equal value
  std::optional<int> commonPortMask;
  for (auto param_value : param_vec) {
    if (param_value == 0) {
      continue; // skip the empty slots
    }
    if (!commonPortMask.has_value()) {
      commonPortMask = param_value;
    } else if (commonPortMask.value() != param_value) {
      llvm::outs() << "ACT mode 1 failed: mixed port index patterns.\n";
      return std::nullopt; // invalid combination of port indices
    }
  }
  if (!commonPortMask.has_value()) {
    // if we have a common port mask, we can use it
    llvm::outs() << "Warning: ACT instruction generated with no ports to "
                    "activate (mode 1).\n";
  }
  int param = commonPortMask.value_or(0);

  return std::make_optional<std::unordered_map<std::string, uint64_t>>(
      {{"mode", 1}, {"param", param}, {"ports", ports}});
}

std::optional<std::unordered_map<std::string, uint64_t>>
ScheduleEpochPassRewriter::create_act_2_instr(std::vector<int> indices) const {
  uint64_t port_vec = 0;
  for (auto index : indices) {
    port_vec |= (1ULL << index); // set the bit for the port index
  }
  // print binary representation of port_vec
  llvm::outs() << "Port vector: " << std::bitset<64>(port_vec).to_string()
               << "\n";

  // return param = 0 as it will be replaced when allocating registers
  return std::make_optional<std::unordered_map<std::string, uint64_t>>(
      {{"mode", 2}, {"param", 0}, {"ports", port_vec}});
}

std::unordered_map<std::string, uint64_t>
ScheduleEpochPassRewriter::create_act_instr(std::vector<int> indices) const {
  if (indices.size() == 0) {
    llvm::outs() << "Error: No port indices provided.\n";
    std::exit(EXIT_FAILURE);
  }

  // try ACT mode 0
  auto result = create_act_0_instr(indices);
  if (result.has_value()) {
    return result.value();
  }

  // try ACT mode 1
  result = create_act_1_instr(indices);
  if (result.has_value()) {
    return result.value();
  }

  // try ACT mode 2
  result = create_act_2_instr(indices);
  if (result.has_value()) {
    return result.value();
  } else {
    llvm::outs() << "Error: Cannot find a valid ACT instruction for the given "
                    "port indices: ";
    for (auto index : indices) {
      llvm::outs() << index << " ";
    }
    llvm::outs() << "\n";
    exit(EXIT_FAILURE); // invalid combination of port indices for mode 1
  }
}

std::unordered_map<std::string, int>
ScheduleEpochPassRewriter::create_wait_instr(int cycle) const {
  llvm::outs() << "Create WAIT for cycle: " << cycle << "\n";
  return {{"mode", 0}, {"cycle", cycle}};
}

::vesyla::pasm::InstrOp ScheduleEpochPassRewriter::compose_act_mlir_op(
    ::vesyla::pasm::EpochOp op, ::mlir::PatternRewriter &rewriter,
    std::unordered_map<std::string, uint64_t> &act_instr_param_map) const {
  return rewriter.create<::vesyla::pasm::InstrOp>(
      op->getLoc(),
      rewriter.getStringAttr(::vesyla::util::Common::gen_random_string(8)),
      rewriter.getStringAttr("act"),
      rewriter.getDictionaryAttr({
          rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                            act_instr_param_map["mode"])),
          rewriter.getNamedAttr("param", rewriter.getI32IntegerAttr(
                                             act_instr_param_map["param"])),
          rewriter.getNamedAttr("ports", rewriter.getI64IntegerAttr(
                                             act_instr_param_map["ports"])),
      }));
}

::vesyla::pasm::InstrOp ScheduleEpochPassRewriter::compose_calc_mlir_op(
    ::vesyla::pasm::EpochOp op, ::mlir::PatternRewriter &rewriter,
    std::unordered_map<std::string, int> &calc_instr_param_map) const {
  return rewriter.create<::vesyla::pasm::InstrOp>(
      op->getLoc(),
      rewriter.getStringAttr(::vesyla::util::Common::gen_random_string(8)),
      rewriter.getStringAttr("calc"),
      rewriter.getDictionaryAttr({
          rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                            calc_instr_param_map["mode"])),
          rewriter.getNamedAttr(
              "operand1",
              rewriter.getI32IntegerAttr(calc_instr_param_map["operand1"])),
          rewriter.getNamedAttr(
              "operand2_sd",
              rewriter.getI32IntegerAttr(calc_instr_param_map["operand2_sd"])),
          rewriter.getNamedAttr(
              "operand2",
              rewriter.getI32IntegerAttr(calc_instr_param_map["operand2"])),
          rewriter.getNamedAttr(
              "result",
              rewriter.getI32IntegerAttr(calc_instr_param_map["result"])),
      }));
}

std::vector<int> ScheduleEpochPassRewriter::get_absolute_port_indices(
    std::vector<::mlir::Operation *> &rop_ops) const {
  std::vector<int> slot_port_index_list;
  for (auto op : rop_ops) {
    auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(op);
    int slot = rop_op.getSlot();
    // The port is no longer on the rop; read it from the port-bearing event
    // instruction in the body (these rops are selected because they contain an
    // evt, which carries the port the ACT signal must address).
    int port = 0;
    if (!rop_op.getBody().empty()) {
      for (::mlir::Operation &body_op : rop_op.getBody().front()) {
        if (auto instr = llvm::dyn_cast<::vesyla::pasm::InstrOp>(&body_op)) {
          if (auto port_attr = llvm::dyn_cast_or_null<::mlir::IntegerAttr>(
                  instr.getParam().get("port"))) {
            port = port_attr.getInt();
            break;
          }
        }
      }
    }
    slot_port_index_list.push_back(slot * 4 + port);
  }
  return slot_port_index_list;
}

} // namespace vesyla::pasm::schedule_epoch_detail
