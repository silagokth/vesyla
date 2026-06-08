#include "ScheduleEpochPassDetail.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace vesyla::pasm::schedule_epoch_detail {

void ScheduleEpochPassRewriter::insert_rop_instructions(
    std::vector<::mlir::Operation *> &rop_ops, int t,
    ::mlir::PatternRewriter &rewriter,
    std::map<int, ::mlir::Operation *> &cell_time_table,
    bool allow_unsafe) const {
  // for each ROP
  for (auto op : rop_ops) {
    auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(op);

    ::mlir::Region &ropBodyRegion = rop_op.getBody();
    ::mlir::Block *ropEntryBlock;
    if (ropBodyRegion.empty()) {
      ropEntryBlock = rewriter.createBlock(&ropBodyRegion);
    } else {
      ropEntryBlock = &ropBodyRegion.front();
    }

    // gather instructions in the ROP body
    std::vector<::mlir::Operation *> rop_child_ops;
    for (::mlir::Operation &rop_child_op : *ropEntryBlock) {
      if (llvm::dyn_cast<::vesyla::pasm::InstrOp>(rop_child_op))
        rop_child_ops.push_back(&rop_child_op);
    }
    int num_instructions = rop_child_ops.size();
    if (num_instructions == 0)
      continue; // skip if there is no instruction in the ROP

    // search for an empty time slot that is the size of the number of
    // instructions in the ROP
    int curr_t = t - 1;
    bool found_slot = false;
    while (!found_slot) {
      found_slot = true;
      for (int offset = 0; offset < num_instructions; offset++) {
        int slot_t = curr_t - offset;
        if (cell_time_table.find(slot_t) != cell_time_table.end()) {
          found_slot = false;
          curr_t = slot_t - 1;
          break;
        }
      }
    }

    // if not found, reverse the order of the child operations
    if (!found_slot && allow_unsafe) {
      llvm::outs() << "Warning: Cannot find a safe time slot for ROP at time "
                   << t
                   << ". Inserting instructions in reverse order, which may"
                      "cause hazards.\n";
      std::reverse(rop_child_ops.begin(), rop_child_ops.end());
      for (auto rop_child_op : rop_child_ops) {
        if (auto instr_op =
                llvm::dyn_cast<::vesyla::pasm::InstrOp>(rop_child_op)) {
          while (cell_time_table.find(curr_t) != cell_time_table.end()) {
            curr_t--;
          }
          cell_time_table[curr_t] = rop_child_op;
          curr_t--;
        }
      }
      continue;
    }

    // place instructions in consecutive time slots
    for (int offset = 0; offset < num_instructions; offset++) {
      int slot_t = curr_t - (num_instructions - 1) + offset;
      cell_time_table[slot_t] = rop_child_ops[offset];
    }
  }
}

void ScheduleEpochPassRewriter::insert_cop_instructions(
    ::mlir::Block *copEntryBlock,
    const std::unordered_map<std::string, int> &schedule_table,
    std::map<int, ::mlir::Operation *> &cell_time_table) const {
  int instr_count = 0;
  for (::mlir::Operation &cop_child_op : *copEntryBlock) {
    if (auto instr_op =
            llvm::dyn_cast<::vesyla::pasm::InstrOp>(&cop_child_op)) {
      std::string instr_anchor =
          instr_op.getId().str() + "_e" + std::to_string(instr_count);

      if (schedule_table.find(instr_anchor) == schedule_table.end()) {
        llvm::outs() << "Error: Cannot find the instruction anchor: "
                     << instr_anchor << "\n";
        std::exit(EXIT_FAILURE);
      }

      int t = schedule_table.at(instr_anchor);

      if (cell_time_table.find(t) != cell_time_table.end()) {
        llvm::outs() << "Error: time table already has the entry: " << t << "("
                     << cell_time_table[t]->getName() << ")"
                     << "\n";
        std::exit(EXIT_FAILURE);
      }

      cell_time_table[t] = &cop_child_op;
      instr_count++;
    } else if (auto yield_op = llvm::dyn_cast<::vesyla::pasm::YieldOp>(
                   &cop_child_op)) {
      // DO NOTHING
    } else {
      llvm::outs() << "Illegal operation type in CopOp: "
                   << cop_child_op.getName() << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void ScheduleEpochPassRewriter::synchronize(
    ::vesyla::pasm::EpochOp &op,
    std::unordered_map<std::string, int> &schedule_table,
    ::mlir::PatternRewriter &rewriter, bool allow_unsafe) const {

  // Get the block to insert the new operations
  ::mlir::Block *block = getEpochBodyEntryBlock(op, rewriter);

  rewriter.setInsertionPointToEnd(block);

  // initialize the time_table and ordered_time_table, add the label for
  // every cell in the fabric
  std::map<std::string, std::map<int, ::mlir::Operation *>> time_table;
  std::map<std::string, std::vector<::mlir::Operation *>> ordered_time_table;
  for (int r = 0; r < _row; r++) {
    for (int c = 0; c < _col; c++) {
      std::string label = std::to_string(r) + "_" + std::to_string(c);
      time_table[label] = std::map<int, ::mlir::Operation *>();
      ordered_time_table[label] = std::vector<::mlir::Operation *>();
    }
  }

  // initialize the time_tables of rop and cop operations
  std::unordered_map<::mlir::Operation *, int> time_table_rop;
  std::unordered_map<::mlir::Operation *, int> time_table_cop;
  for (::mlir::Operation &child_op : *block) {
    if (auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(&child_op)) {
      time_table_rop[&child_op] = schedule_table[rop_op.getId().str()];
    } else if (auto cop_op =
                   llvm::dyn_cast<::vesyla::pasm::CopOp>(&child_op)) {
      time_table_cop[&child_op] = schedule_table[cop_op.getId().str()];
    } else if (auto raw_op =
                   llvm::dyn_cast<::vesyla::pasm::RawOp>(&child_op)) {
      // DO NOTHING
    } else if (auto instr_op =
                   llvm::dyn_cast<::vesyla::pasm::InstrOp>(&child_op)) {
      // DO NOTHING
    } else if (auto cstr_op =
                   llvm::dyn_cast<::vesyla::pasm::CstrOp>(&child_op)) {
      // DO NOTHING
    } else if (auto yield_op =
                   llvm::dyn_cast<::vesyla::pasm::YieldOp>(&child_op)) {
      // DO NOTHING
    } else {
      llvm::outs() << "Illegal operation type in EpochOp for synchronization: "
                   << child_op.getName() << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  // place all COPs
  for (auto it = time_table_cop.begin(); it != time_table_cop.end(); ++it) {
    auto cop_op = llvm::dyn_cast<::vesyla::pasm::CopOp>(it->first);
    std::string label = std::to_string(cop_op.getRow()) + "_" +
                        std::to_string(cop_op.getCol());
    auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);

    ::mlir::Region &copBodyRegion = cop_op.getBody();
    ::mlir::Block *copEntryBlock =
        getOrCreateEntryBlock(copBodyRegion, rewriter);
    insert_cop_instructions(copEntryBlock, schedule_table, cell_time_table);
  }

  // place the ACT instruction of all ROPs
  int total_latency = schedule_table["total_latency"];
  std::unordered_map<std::string, bool> cell_contains_act_mode2;
  for (int t = 0; t < total_latency; t++) {
    std::map<std::string, std::vector<::mlir::Operation *>> rop_ops_at_t =
        get_rop_ops_for_cycle(t, time_table_rop);

    if (rop_ops_at_t.empty())
      continue;

    llvm::outs() << "ROPs at t=" << t << ":\n";

    for (auto it = rop_ops_at_t.begin(); it != rop_ops_at_t.end(); ++it) {
      std::string label = it->first;
      cell_contains_act_mode2[label] =
          cell_contains_act_mode2.find(label) != cell_contains_act_mode2.end()
              ? cell_contains_act_mode2[label]
              : false;
      auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);
      std::vector<::mlir::Operation *> rop_ops = it->second;
      std::vector<int> slot_port_index_list = get_absolute_port_indices(rop_ops);

      llvm::outs() << "  - cell " << label << ": [";
      for (size_t rop_i = 0; rop_i < rop_ops.size(); rop_i++) {
        auto rop_op = llvm::dyn_cast<::vesyla::pasm::RopOp>(rop_ops[rop_i]);
        auto rop_port = slot_port_index_list[rop_i];
        if (rop_i == rop_ops.size() - 1)
          llvm::outs() << rop_op.getId() << " (" << rop_port << ")]\n";
        else
          llvm::outs() << rop_op.getId() << " (" << rop_port << "), ";
      }

      auto act_instr_param_map = create_act_instr(slot_port_index_list);

      if (act_instr_param_map["mode"] == 2)
        cell_contains_act_mode2[label] = true;

      auto act_instr = compose_act_mlir_op(op, rewriter, act_instr_param_map);
      create_time_table_entry(cell_time_table, t, act_instr);
    }
  }

  for (auto &cell_entry : time_table) {
    const std::string &cell_label = cell_entry.first;
    auto &cell_time_table = cell_entry.second;

    if (!cell_contains_act_mode2[cell_label])
      continue;

    std::vector<std::pair<int, ::mlir::Operation *>> cycles(
        cell_time_table.begin(), cell_time_table.end());

    print_time_table(time_table);

    std::map<int, std::vector<int>> register_allocation_table =
        create_reg_alloc_table();
    for (const auto &cycle_entry : cycles) {
      int cycle = cycle_entry.first;
      ::mlir::Operation *act_op = cycle_entry.second;
      if (op_is_act_mode2(act_op)) {
        uint64_t ports = get_ports_from_act_mode2_instr(act_op);
        int first_reg_address = allocateAndScheduleActMode2PrepInstructions(
            register_allocation_table, cell_time_table, rewriter, op, cycle,
            ports);
        changeActMode2Param(act_op, first_reg_address, rewriter);
      }
    }
  }

  llvm::outs() << "Time table after placing ACT instructions:\n";
  print_time_table(time_table);

  for (int t = 0; t < total_latency; t++) {
    std::map<std::string, std::vector<::mlir::Operation *>> rop_ops_at_t =
        get_rop_ops_for_cycle(t, time_table_rop);
    for (auto it = rop_ops_at_t.begin(); it != rop_ops_at_t.end(); ++it) {
      std::string label = it->first;
      auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);
      std::vector<::mlir::Operation *> rop_ops = it->second;
      insert_rop_instructions(rop_ops, t, rewriter, cell_time_table,
                              allow_unsafe);
    }
  }

  llvm::outs() << "Time table after placing ROP instructions:\n";
  print_time_table(time_table);

  // find out the time shift amount, so that the first operation is at
  // time 0
  auto test = time_table;
  int min_shift_time = 0;
  for (auto it = time_table.begin(); it != time_table.end(); ++it) {
    int min_time = std::numeric_limits<int>::max();
    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      if (it2->first < min_time) {
        min_time = it2->first;
      }
    }
    if (min_time < min_shift_time) {
      min_shift_time = min_time;
    }
  }
  min_shift_time = -min_shift_time;
  llvm::outs() << "Min shift time: " << min_shift_time << "\n";
  total_latency = total_latency + min_shift_time;
  llvm::outs() << "Total latency after shifting: " << total_latency << "\n";

  // shift the time table
  for (auto it = time_table.begin(); it != time_table.end(); ++it) {
    std::map<int, ::mlir::Operation *> new_time_table;
    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      new_time_table[it2->first + min_shift_time] = it2->second;
    }
    time_table[it->first] = std::move(new_time_table);
  }

  llvm::outs() << "Time table after shifting:\n";
  print_time_table(time_table);

  // order the operations by time
  for (auto it = time_table.begin(); it != time_table.end(); ++it) {
    std::vector<std::pair<int, ::mlir::Operation *>> time_op_vec;
    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      time_op_vec.push_back(*it2);
    }
    std::sort(time_op_vec.begin(), time_op_vec.end(),
              [](const std::pair<int, ::mlir::Operation *> &a,
                 const std::pair<int, ::mlir::Operation *> &b) {
                return a.first < b.first;
              });

    // wait(N) consumes N+1 cycles (1 issue + N stall). Mid-gap fills
    // curr_t - prev_t - 1 cycles, tail pads to total_latency.
    int prev_t = -1;
    std::vector<std::pair<int, ::mlir::Operation *>> new_time_op_vec;
    if (time_op_vec.size() > 0) {
      // start from the smallest time, insert the WAIT instructions if
      // there is a gap between consecutive operations
      for (size_t i = 0; i < time_op_vec.size(); i++) {
        int curr_t = time_op_vec[i].first;
        if (curr_t - prev_t > 1) {
          rewriter.setInsertionPointToEnd(block);
          auto wait_instr_param_map = create_wait_instr(curr_t - prev_t - 2);
          ::mlir::StringAttr id = rewriter.getStringAttr(
              ::vesyla::util::Common::gen_random_string(8));
          ::mlir::StringAttr type = rewriter.getStringAttr("wait");
          ::mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
              {rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                                 wait_instr_param_map["mode"])),
               rewriter.getNamedAttr(
                   "cycle", rewriter.getI32IntegerAttr(
                                wait_instr_param_map["cycle"]))});
          auto wait_instr = rewriter.create<::vesyla::pasm::InstrOp>(
              time_op_vec[i].second->getLoc(), id, type, param);
          new_time_op_vec.push_back(
              std::make_pair(prev_t + 1, wait_instr.getOperation()));
          new_time_op_vec.push_back(time_op_vec[i]);
        } else {
          new_time_op_vec.push_back(time_op_vec[i]);
        }
        prev_t = curr_t;
      }
    }

    if (prev_t != total_latency - 1) {
      rewriter.setInsertionPointToEnd(block);
      auto wait_instr_param_map =
          create_wait_instr(total_latency - 2 - prev_t);
      ::mlir::StringAttr id = rewriter.getStringAttr(
          ::vesyla::util::Common::gen_random_string(8));
      ::mlir::StringAttr type = rewriter.getStringAttr("wait");
      ::mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
          {rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                             wait_instr_param_map["mode"])),
           rewriter.getNamedAttr(
               "cycle", rewriter.getI32IntegerAttr(
                            wait_instr_param_map["cycle"]))});
      rewriter.setInsertionPointToEnd(block);
      auto wait_instr = rewriter.create<::vesyla::pasm::InstrOp>(
          op->getLoc(), id, type, param);
      new_time_op_vec.push_back(
          std::make_pair(prev_t + 1, wait_instr.getOperation()));
    }

    if (ordered_time_table.find(it->first) == ordered_time_table.end()) {
      ordered_time_table[it->first] = std::vector<::mlir::Operation *>();
    }

    for (auto it2 = new_time_op_vec.begin(); it2 != new_time_op_vec.end();
         ++it2) {
      ordered_time_table[it->first].push_back(it2->second);
    }

    llvm::outs() << "Ordered time table: " << it->first << "\n";
    for (auto it2 = ordered_time_table[it->first].begin();
         it2 != ordered_time_table[it->first].end(); ++it2) {
      llvm::dyn_cast<::vesyla::pasm::InstrOp>(*it2).print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs().flush();
    }
  }

  // add a yield instruction at the end of the block to separate the
  // operations
  rewriter.setInsertionPointToEnd(block);
  auto yield_instr =
      rewriter.create<::vesyla::pasm::YieldOp>(op->getLoc());

  // insert a RawOp for each cell at the end of the block
  for (auto it = ordered_time_table.begin(); it != ordered_time_table.end();
       ++it) {
    std::string label = it->first;
    int row = std::stoi(label.substr(0, label.find("_")));
    int col = std::stoi(label.substr(label.find("_") + 1));
    std::string raw_op_id = ::vesyla::util::Common::gen_random_string(8);
    ::mlir::StringAttr id = rewriter.getStringAttr(raw_op_id);
    ::mlir::IntegerAttr row_attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), row);
    ::mlir::IntegerAttr col_attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), col);
    rewriter.setInsertionPointToEnd(block);
    auto raw_op = rewriter.create<::vesyla::pasm::RawOp>(op->getLoc(), id,
                                                         row_attr, col_attr);
    ::mlir::Region &raw_op_body = raw_op.getBody();
    ::mlir::Block *raw_op_entry_block;
    if (raw_op_body.empty()) {
      raw_op_entry_block = rewriter.createBlock(&raw_op_body);
    } else {
      raw_op_entry_block = &raw_op_body.front();
    }

    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      rewriter.setInsertionPointToEnd(raw_op_entry_block);
      rewriter.clone(**it2);
    }
    rewriter.setInsertionPointToEnd(raw_op_entry_block);
    rewriter.create<::vesyla::pasm::YieldOp>(raw_op->getLoc());

    // Verifier: per-cell stream cycles must equal total_latency.
    int stream_cycles = 0;
    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      auto instr_op = llvm::dyn_cast<::vesyla::pasm::InstrOp>(*it2);
      if (!instr_op) {
        continue;
      }
      if (instr_op.getType() == "wait") {
        ::mlir::DictionaryAttr params = instr_op.getParam();
        int wait_n = 0;
        if (params.contains("cycle")) {
          if (auto int_attr =
                  llvm::dyn_cast<::mlir::IntegerAttr>(params.get("cycle"))) {
            wait_n = int_attr.getInt();
          }
        }
        stream_cycles += wait_n + 1;
      } else {
        stream_cycles += 1;
      }
    }
    if (stream_cycles != total_latency) {
      llvm::outs() << "Error: ScheduleEpochPass cell " << it->first
                   << " stream cycles (" << stream_cycles
                   << ") != total_latency (" << total_latency
                   << "). Inter-cell sync broken — "
                   << "check op duration_expr (multi-cycle ops like "
                      "rep/act must reflect HW cycles in MZN) or wait "
                      "param math.\n";
      std::exit(EXIT_FAILURE);
    }
  }

  // start from the end, remove everything after the first yield
  // instruction in block
  bool found_yield = false;
  std::vector<::mlir::Operation *> remove_ops;
  for (auto it = block->getOperations().rbegin();
       it != block->getOperations().rend(); ++it) {
    if (!found_yield) {
      if (llvm::isa<::vesyla::pasm::YieldOp>(*it)) {
        found_yield = true;
        remove_ops.push_back(&*it);
        continue;
      }
    } else {
      remove_ops.push_back(&*it);
    }
  }

  for (auto *op_to_remove : remove_ops) {
    rewriter.eraseOp(op_to_remove);
  }

  rewriter.setInsertionPointToEnd(block);
  rewriter.create<::vesyla::pasm::YieldOp>(op->getLoc());

  llvm::outs() << "Block contents after synchronization:\n";
  for (::mlir::Operation &child_op : *block) {
    llvm::outs() << "Operation type: " << child_op.getName() << "\n";
  }
}

} // namespace vesyla::pasm::schedule_epoch_detail
