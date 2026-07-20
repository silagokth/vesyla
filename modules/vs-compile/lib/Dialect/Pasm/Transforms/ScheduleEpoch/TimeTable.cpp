#include "ScheduleEpochPassDetail.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace vesyla::pasm::schedule_epoch_detail {

::mlir::Block *ScheduleEpochPassRewriter::getOrCreateEntryBlock(
    ::mlir::Region &region, ::mlir::PatternRewriter &rewriter) const {
  ::mlir::Block *block;
  if (region.empty()) {
    block = rewriter.createBlock(&region);
  } else {
    block = &region.front();
  }
  return block;
}

::mlir::Block *ScheduleEpochPassRewriter::getEpochBodyEntryBlock(
    ::vesyla::pasm::EpochOp epoch_op,
    ::mlir::PatternRewriter &rewriter) const {
  // Check if the EpochOp is valid
  if (!epoch_op) {
    llvm::outs() << "Error: Cannot find the EpochOp in the operation.\n";
    std::exit(EXIT_FAILURE);
  }

  ::mlir::Region &epochBodyRegion = epoch_op.getBody();
  ::mlir::Block *block = getOrCreateEntryBlock(epochBodyRegion, rewriter);
  return block;
}

std::map<int, ::mlir::Operation *> &
ScheduleEpochPassRewriter::getOrCreateCellTimeTable(
    std::map<std::string, std::map<int, ::mlir::Operation *>> &time_table,
    const std::string &label) const {
  if (time_table.find(label) == time_table.end())
    time_table[label] = std::map<int, ::mlir::Operation *>();
  return time_table[label];
}

void ScheduleEpochPassRewriter::create_time_table_entry(
    std::map<int, ::mlir::Operation *> &cell_time_table, int t,
    ::vesyla::pasm::InstrOp &instr_op) const {

  if (cell_time_table.find(t) == cell_time_table.end()) {
    cell_time_table[t] = instr_op.getOperation();
  } else {
    llvm::outs() << "Error: time table already has the entry: " << t << "("
                 << cell_time_table[t]->getName() << ")"
                 << "\n";
    std::exit(EXIT_FAILURE);
  }
}

std::map<std::string, std::vector<::mlir::Operation *>>
ScheduleEpochPassRewriter::get_rop_ops_for_cycle(
    int currentCycle,
    std::unordered_map<::mlir::Operation *, int> time_table_rop) const {
  std::map<std::string, std::vector<::mlir::Operation *>> rop_ops_at_t;
  for (auto it = time_table_rop.begin(); it != time_table_rop.end(); ++it) {
    // only keep the rop ops for the current cycle
    if (it->second != currentCycle)
      continue;

    auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(it->first);
    int row = rop_op.getRow();
    int col = rop_op.getCol();
    std::string label = std::to_string(row) + "_" + std::to_string(col);

    // if the cell label is not found, create an entry
    if (rop_ops_at_t.find(label) == rop_ops_at_t.end()) {
      rop_ops_at_t[label] = std::vector<::mlir::Operation *>();
    }
    rop_ops_at_t[label].push_back(it->first);
  }

  // Sort each cell's rop list deterministically by source position (the
  // order in which rops appear in the parent block). The input
  // time_table_rop is an unordered_map keyed by Operation*, so its
  // iteration order varies between runs. Without this sort, rops that
  // share a cycle within a cell get serialised in random order, which
  // changes the dispatch order in instr.bin and exposes a flaky-looking
  // test failure on testcases like mul_512_1_1. Using source order
  // matches what the user wrote in the pasm and keeps the dispatch
  // sequence stable.
  for (auto &kv : rop_ops_at_t) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](::mlir::Operation *a, ::mlir::Operation *b) {
                return a->isBeforeInBlock(b);
              });
  }

  return rop_ops_at_t;
}

void ScheduleEpochPassRewriter::replace_time_in_instr_param(
    ::vesyla::pasm::EpochOp &op,
    std::unordered_map<std::string, int> &schedule_table,
    ::mlir::PatternRewriter &rewriter) const {

  auto epoch_op = op;
  ::mlir::Region &epoch_region = epoch_op.getBody();
  if (epoch_region.empty()) {
    return;
  }
  ::mlir::Block *epoch_block = &epoch_region.front();
  for (::mlir::Operation &child_op : *epoch_block) {
    if (auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(&child_op)) {
      ::mlir::Region &rop_region = rop_op.getBody();
      if (rop_region.empty()) {
        continue;
      }
      ::mlir::Block *rop_entry_block = &rop_region.front();
      for (::mlir::Operation &instr_op : *rop_entry_block) {
        if (auto instr = llvm::dyn_cast<::vesyla::pasm::InstrOp>(&instr_op)) {
          ::mlir::DictionaryAttr current_instr_params = instr.getParam();
          llvm::SmallVector<::mlir::NamedAttribute> updated_attrs;
          bool params_changed = false;

          for (const ::mlir::NamedAttribute &named_attr_entry :
               current_instr_params) {
            auto attr_name = named_attr_entry.getName();
            auto attr_value = named_attr_entry.getValue();

            // The `variant` selector is a structural string, not a timing
            // reference. It must never be substituted from the schedule table,
            // even if its value coincides with a scheduled symbol name (e.g. a
            // "swb" variant colliding with a rop named "swb").
            if (attr_name == "variant") {
              updated_attrs.push_back(named_attr_entry);
            } else if (auto str_attr =
                           llvm::dyn_cast<::mlir::StringAttr>(attr_value)) {
              std::string str_value = str_attr.getValue().str();
              auto it = schedule_table.find(str_value);
              if (it != schedule_table.end()) {
                int int_value = it->second;
                ::mlir::Attribute new_attr_value =
                    rewriter.getIntegerAttr(rewriter.getI32Type(), int_value);
                updated_attrs.push_back(
                    rewriter.getNamedAttr(attr_name, new_attr_value));
                params_changed = true;
              } else {
                updated_attrs.push_back(named_attr_entry);
              }
            } else {
              updated_attrs.push_back(named_attr_entry);
            }
          }

          if (params_changed) {
            ::mlir::DictionaryAttr new_instr_params =
                rewriter.getDictionaryAttr(updated_attrs);
            instr->setAttr("param", new_instr_params);
          }
        }
      }

    } else if (auto cop_op = llvm::dyn_cast<::vesyla::pasm::CopOp>(&child_op)) {
      ::mlir::Region &cop_region = cop_op.getBody();
      if (cop_region.empty()) {
        continue;
      }
      ::mlir::Block *cop_entry_block = &cop_region.front();
      for (::mlir::Operation &instr_op : *cop_entry_block) {
        if (auto instr = llvm::dyn_cast<::vesyla::pasm::InstrOp>(&instr_op)) {
          ::mlir::DictionaryAttr current_instr_params = instr.getParam();
          llvm::SmallVector<::mlir::NamedAttribute> updated_attrs;
          bool params_changed = false;

          for (const ::mlir::NamedAttribute &named_attr_entry :
               current_instr_params) {
            auto attr_name = named_attr_entry.getName();
            auto attr_value = named_attr_entry.getValue();

            // The `variant` selector is a structural string, not a timing
            // reference. It must never be substituted from the schedule table,
            // even if its value coincides with a scheduled symbol name (e.g. a
            // "swb" variant colliding with a rop named "swb").
            if (attr_name == "variant") {
              updated_attrs.push_back(named_attr_entry);
            } else if (auto str_attr =
                           llvm::dyn_cast<::mlir::StringAttr>(attr_value)) {
              std::string str_value = str_attr.getValue().str();
              auto it = schedule_table.find(str_value);
              if (it != schedule_table.end()) {
                int int_value = it->second;
                ::mlir::Attribute new_attr_value =
                    rewriter.getIntegerAttr(rewriter.getI32Type(), int_value);
                updated_attrs.push_back(
                    rewriter.getNamedAttr(attr_name, new_attr_value));
                params_changed = true;
              } else {
                updated_attrs.push_back(named_attr_entry);
              }
            } else {
              updated_attrs.push_back(named_attr_entry);
            }
          }

          if (params_changed) {
            ::mlir::DictionaryAttr new_instr_params =
                rewriter.getDictionaryAttr(updated_attrs);
            instr->setAttr("param", new_instr_params);
          }
        }
      }
    }
  }
}

void ScheduleEpochPassRewriter::print_time_table(
    std::map<std::string, std::map<int, ::mlir::Operation *>> &time_table)
    const {
  // print the time table
  for (auto it = time_table.begin(); it != time_table.end(); ++it) {
    auto cell_label = it->first;
    auto cell_time_table = it->second;
    llvm::outs() << "Time table: " << cell_label << "\n";
    for (auto cell_it = cell_time_table.begin();
         cell_it != cell_time_table.end(); ++cell_it) {
      auto cycle = cell_it->first;
      auto op = llvm::dyn_cast<::vesyla::pasm::InstrOp>(cell_it->second);
      llvm::outs() << "  " << cycle << ": ";
      op.print(llvm::outs());
      llvm::outs() << "\n";
    }
  }
}

} // namespace vesyla::pasm::schedule_epoch_detail
