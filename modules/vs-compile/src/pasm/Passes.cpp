#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "pasm/Passes.hpp"
#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include "util/Common.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_REPLACELOOPOP
#define GEN_PASS_DEF_REPLACEEPOCHOP
#include "pasm/Passes.hpp.inc"

std::string gen_random_string(size_t length) {
  if (length == 0) {
    return "";
  }

  // generate a random string starting with a alphabetic character
  const std::string letters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const std::string numbers = "0123456789";
  const std::string characters = letters + numbers;
  std::string result;
  result.reserve(length);
  result += letters[rand() % letters.size()];
  for (size_t i = 1; i < length; ++i) {
    result += characters[rand() % characters.size()];
  }
  return result;
}

namespace {

//===----------------------------------------------------------------------===//
class ReplaceEpochOpRewriter : public OpRewritePattern<EpochOp> {
public:
  nlohmann::json component_map;
  std::string component_path;
  std::string tmp_path;
  int _row;
  int _col;

public:
  ReplaceEpochOpRewriter(MLIRContext *context, std::string component_map,
                         std::string component_path, std::string tmp_path,
                         int row_, int col_)
      : OpRewritePattern<EpochOp>(context),
        component_map(nlohmann::json::parse(component_map)),
        component_path(std::move(component_path)),
        tmp_path(std::move(tmp_path)), _row(row_), _col(col_) {}

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

  std::unordered_map<string, int>
  create_act_instr(std::vector<int> indices) const {

    llvm::outs() << "Create ACT for indices: ";
    for (auto index : indices) {
      llvm::outs() << index << " ";
    }
    llvm::outs() << "\n";

    if (indices.size() == 0) {
      llvm::outs() << "Error: No indices provided.\n";
      std::exit(-1);
    }

    int mode = 0;
    int param = 0;
    int ports = 0;

    int min_index = *std::min_element(indices.begin(), indices.end());
    int max_index = *std::max_element(indices.begin(), indices.end());

    int min_slot = min_index / 4;
    int max_slot = max_index / 4 + 1;
    if (max_slot - min_slot < 4) {
      mode = 0;
      param = min_slot;
      for (auto index : indices) {
        ports |= (1 << (index - min_slot * 4));
      }
    } else {
      mode = 1;
      std::vector<int> param_vec(16, 0);
      for (auto index : indices) {
        int slot = index / 4;
        int port = index % 4;
        ports |= (1 << slot);
        param_vec[index] |= (1 << port);
      }
      // it's valid only if param_vec value is either 0 or equal value
      int param = 0;
      for (auto param_value : param_vec) {
        if (param == 0) {
          param = param_value;
        } else if (param != param_value && param_value != 0) {
          llvm::outs() << "Cannot find a valid ACT instruction for the given "
                          "indices: ";
          for (auto index : indices) {
            llvm::outs() << index << " ";
          }
          llvm::outs() << "\n";
        }
      }
    }

    return {{"mode", mode}, {"param", param}, {"ports", ports}};
  }

  std::unordered_map<string, int> create_wait_instr(int cycles) const {
    llvm::outs() << "Create WAIT for cycles: " << cycles << "\n";
    return {{"mode", 0}, {"cycles", cycles}};
  }

  void synchronize(EpochOp *op,
                   std::unordered_map<std::string, int> &schedule_table,
                   PatternRewriter &rewriter) const {

    // Get the block to insert the new operations
    auto epoch_op = *op;
    if (!epoch_op) {
      llvm::outs() << "Error: Cannot find the EpochOp in the operation.\n";
      std::exit(-1);
    }
    mlir::Region &epochBodyRegion = epoch_op.getBody();
    mlir::Block *block;
    if (epochBodyRegion.empty()) {
      block = rewriter.createBlock(&epochBodyRegion);
    } else {
      block = &epochBodyRegion.front();
    }

    rewriter.setInsertionPointToEnd(block);

    std::unordered_map<string, std::unordered_map<int, mlir::Operation *>>
        time_table;
    std::unordered_map<string, std::vector<mlir::Operation *>>
        ordered_time_table;
    std::unordered_map<mlir::Operation *, int> time_table_rop;
    std::unordered_map<mlir::Operation *, int> time_table_cop;

    // initialize the time_table and ordered_time_table, add the label for every
    // cell in the fabric
    for (int r = 0; r < _row; r++) {
      for (int c = 0; c < _col; c++) {
        std::string label = std::to_string(r) + "_" + std::to_string(c);
        time_table[label] = std::unordered_map<int, mlir::Operation *>();
        ordered_time_table[label] = std::vector<mlir::Operation *>();
      }
    }

    for (mlir::Operation &child_op : *block) {
      if (auto rop_op = llvm::dyn_cast<RopOp>(&child_op)) {
        time_table_rop[&child_op] = schedule_table[rop_op.getId().str()];
      } else if (auto cop_op = llvm::dyn_cast<CopOp>(&child_op)) {
        time_table_cop[&child_op] = schedule_table[cop_op.getId().str()];
      } else if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
        // DO NOTHING
      } else if (auto instr_op = llvm::dyn_cast<InstrOp>(&child_op)) {
        // DO NOTHING
      } else if (auto constraint_op = llvm::dyn_cast<ConstraintOp>(&child_op)) {
        // DO NOTHING
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&child_op)) {
        // DO NOTHING
      } else {
        llvm::outs()
            << "Illegal operation type in EpochOp for synchronization: "
            << child_op.getName() << "\n";
        std::exit(-1);
      }
    }

    // place all COPs
    for (auto it = time_table_cop.begin(); it != time_table_cop.end(); ++it) {
      auto cop_op = llvm::dyn_cast<CopOp>(it->first);
      int row = cop_op.getRow();
      int col = cop_op.getCol();
      std::string label = std::to_string(row) + "_" + std::to_string(col);
      if (time_table.find(label) == time_table.end()) {
        time_table[label] = unordered_map<int, mlir::Operation *>();
      }

      // get internal instructions
      mlir::Region &copBodyRegion = cop_op.getBody();
      mlir::Block *copEntryBlock;
      if (copBodyRegion.empty()) {
        copEntryBlock = rewriter.createBlock(&copBodyRegion);
      } else {
        copEntryBlock = &copBodyRegion.front();
      }
      int instr_count = 0;
      for (mlir::Operation &cop_child_op : *copEntryBlock) {
        if (auto instr_op = llvm::dyn_cast<InstrOp>(&cop_child_op)) {
          if (schedule_table.find(instr_op.getId().str() + "_e" +
                                  std::to_string(instr_count)) ==
              schedule_table.end()) {
            llvm::outs() << "Error: Cannot find the instruction anchor: "
                         << instr_op.getId().str() + "_e" +
                                std::to_string(instr_count)
                         << "\n";
            std::exit(-1);
          }
          int t = schedule_table[instr_op.getId().str() + "_e" +
                                 std::to_string(instr_count)];
          if (time_table[label].find(t) == time_table[label].end()) {
            time_table[label][t] = &cop_child_op;
          } else {
            llvm::outs() << "Error: time table already has the entry: " << t
                         << "(" << time_table[label][t]->getName() << ")"
                         << "\n";
            std::exit(-1);
          }
          instr_count++;
        } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&cop_child_op)) {
          // DO NOTHING
        } else {
          llvm::outs() << "Illegal operation type in CopOp: "
                       << cop_child_op.getName() << "\n";
          std::exit(-1);
        }
      }
    }

    // place the ACT instruction of all ROPs
    int total_latency = schedule_table["total_latency"];

    for (int t = 0; t < total_latency; t++) {
      std::unordered_map<std::string, std::vector<mlir::Operation *>>
          rop_ops_at_t;
      for (auto it = time_table_rop.begin(); it != time_table_rop.end(); ++it) {
        if (it->second == t) {
          llvm::outs() << "ROP: " << it->first->getName() << " at time " << t
                       << "\n";
          auto rop_op = llvm::dyn_cast<RopOp>(it->first);
          llvm::outs() << "ROP: " << rop_op.getId() << " at time " << t << "\n";
          int row = rop_op.getRow();
          int col = rop_op.getCol();
          std::string label = std::to_string(row) + "_" + std::to_string(col);
          if (rop_ops_at_t.find(label) == rop_ops_at_t.end()) {
            rop_ops_at_t[label] = std::vector<mlir::Operation *>();
          }
          rop_ops_at_t[label].push_back(it->first);
        }
      }
      for (auto it = rop_ops_at_t.begin(); it != rop_ops_at_t.end(); ++it) {
        std::string label = it->first;
        auto rop_ops = it->second;
        std::vector<int> slot_port_index_list;
        for (auto op : rop_ops) {
          auto rop_op = llvm::dyn_cast<RopOp>(op);
          int slot = rop_op.getSlot();
          int port = rop_op.getPort();
          slot_port_index_list.push_back(slot * 4 + port);
        }

        llvm::outs() << "ROP2: " << "\n";

        // create an ACT instruction
        auto act_instr_param_map = create_act_instr(slot_port_index_list);
        auto act_instr = rewriter.create<vesyla::pasm::InstrOp>(
            op->getLoc(), rewriter.getStringAttr(get_random_string(8)),
            rewriter.getStringAttr("act"),
            rewriter.getDictionaryAttr({
                rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                                  act_instr_param_map["mode"])),
                rewriter.getNamedAttr(
                    "param",
                    rewriter.getI32IntegerAttr(act_instr_param_map["param"])),
                rewriter.getNamedAttr(
                    "ports",
                    rewriter.getI32IntegerAttr(act_instr_param_map["ports"])),
            }));
        if (time_table.find(label) == time_table.end()) {
          time_table[label] = unordered_map<int, mlir::Operation *>();
        }
        if (time_table[label].find(t) == time_table[label].end()) {
          time_table[label][t] = act_instr.getOperation();
        } else {
          llvm::outs() << "Error: time table already has the entry: " << t
                       << "(" << time_table[label][t]->getName() << ")"
                       << "\n";
          std::exit(-1);
        }

        llvm::outs() << "ACT: " << act_instr.getId() << " at time " << t
                     << "\n";

        // start from t-1, insert the instructions in the rop_ops
        for (auto op : rop_ops) {
          auto rop_op = llvm::dyn_cast<RopOp>(op);
          int curr_t = t - 1;
          mlir::Region &ropBodyRegion = rop_op.getBody();
          mlir::Block *ropEntryBlock;
          if (ropBodyRegion.empty()) {
            ropEntryBlock = rewriter.createBlock(&ropBodyRegion);
          } else {
            ropEntryBlock = &ropBodyRegion.front();
          }
          vector<mlir::Operation *> rop_child_ops;
          for (mlir::Operation &rop_child_op : *ropEntryBlock) {
            rop_child_ops.push_back(&rop_child_op);
          }
          // reverse the order of the child operations
          std::reverse(rop_child_ops.begin(), rop_child_ops.end());
          for (auto rop_child_op : rop_child_ops) {
            if (auto instr_op = llvm::dyn_cast<InstrOp>(rop_child_op)) {

              // find out all StringAttr parameters in the DictionaryAttr, check
              // if it matches any entry in the schedule_table, if it does, then
              // replace the field with integer value in schedule_table.

              mlir::DictionaryAttr current_instr_params = instr_op.getParam();
              llvm::SmallVector<mlir::NamedAttribute> updated_attrs;
              bool params_changed = false;

              for (const mlir::NamedAttribute &named_attr_entry :
                   current_instr_params) {
                auto attr_name = named_attr_entry.getName();
                auto attr_value = named_attr_entry.getValue();

                if (auto str_attr =
                        llvm::dyn_cast<mlir::StringAttr>(attr_value)) {
                  std::string str_value = str_attr.getValue().str();
                  // Check if the string value exists as a key in the
                  // schedule_table
                  auto it = schedule_table.find(str_value);
                  if (it != schedule_table.end()) {
                    int int_value = it->second;
                    mlir::Attribute new_attr_value = rewriter.getIntegerAttr(
                        rewriter.getI32Type(), int_value);
                    // Add the modified attribute to our new list
                    updated_attrs.push_back(
                        rewriter.getNamedAttr(attr_name, new_attr_value));
                    params_changed = true;
                  } else {
                    // If no match in schedule_table, keep the original
                    // attribute
                    updated_attrs.push_back(named_attr_entry);
                  }
                } else {
                  // If not a StringAttr, keep the original attribute
                  updated_attrs.push_back(named_attr_entry);
                }
              }

              // If any attributes were changed, create a new DictionaryAttr and
              // update the operation
              if (params_changed) {
                mlir::DictionaryAttr new_instr_params =
                    rewriter.getDictionaryAttr(updated_attrs);
                // "param" is the name of the DictionaryAttr attribute in your
                // InstrOp definition
                instr_op->setAttr("param", new_instr_params);
              }

              while (time_table[label].find(curr_t) !=
                     time_table[label].end()) {
                curr_t--;
              }
              time_table[label][curr_t] = rop_child_op;
              curr_t--;
            }
          }
        }
      }
    }
    // print the time table
    for (auto it = time_table.begin(); it != time_table.end(); ++it) {
      llvm::outs() << "Time table: " << it->first << "\n";
      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        llvm::outs() << "  " << it2->first << ": " << it2->second->getName()
                     << "\n";
      }
    }

    // find out the time shift amount, so that the first operation is at
    // time 0
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

    // shift the time table
    for (auto it = time_table.begin(); it != time_table.end(); ++it) {
      std::unordered_map<int, mlir::Operation *> new_time_table;
      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        new_time_table[it2->first + min_shift_time] = it2->second;
      }
      time_table[it->first] = std::move(new_time_table);
    }

    // order the operations by time
    for (auto it = time_table.begin(); it != time_table.end(); ++it) {
      std::vector<std::pair<int, mlir::Operation *>> time_op_vec;
      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        time_op_vec.push_back(*it2);
      }
      std::sort(time_op_vec.begin(), time_op_vec.end(),
                [](const std::pair<int, mlir::Operation *> &a,
                   const std::pair<int, mlir::Operation *> &b) {
                  return a.first < b.first;
                });

      int prev_t = 0;
      std::vector<std::pair<int, mlir::Operation *>> new_time_op_vec;
      if (time_op_vec.size() > 0) {
        // start from the smallest time, insert the WAIT instructions if there
        // is a gap between consecutive operations
        for (int i = 0; i < time_op_vec.size(); i++) {
          int curr_t = time_op_vec[i].first;
          if (curr_t - prev_t > 1) {
            // create a WAIT instruction
            auto wait_instr_param_map = create_wait_instr(curr_t - prev_t - 1);
            mlir::StringAttr id = rewriter.getStringAttr(get_random_string(8));
            mlir::StringAttr type = rewriter.getStringAttr("wait");
            mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
                {rewriter.getNamedAttr(
                     "mode",
                     rewriter.getI32IntegerAttr(wait_instr_param_map["mode"])),
                 rewriter.getNamedAttr("cycles",
                                       rewriter.getI32IntegerAttr(
                                           wait_instr_param_map["cycles"]))});
            auto wait_instr = rewriter.create<InstrOp>(
                time_op_vec[i].second->getLoc(), id, type, param);
            new_time_op_vec.push_back(
                std::make_pair(prev_t + 1, wait_instr.getOperation()));
          } else {
            new_time_op_vec.push_back(time_op_vec[i]);
          }
          prev_t = curr_t;
        }
      }

      // insert a wait instruction at the end if the last operation is not the
      // total_latency-1
      if (prev_t != total_latency - 1) {
        auto wait_instr_param_map =
            create_wait_instr(total_latency - 1 - prev_t);
        mlir::StringAttr id = rewriter.getStringAttr(get_random_string(8));
        mlir::StringAttr type = rewriter.getStringAttr("wait");
        mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
            {rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                               wait_instr_param_map["mode"])),
             rewriter.getNamedAttr(
                 "cycles",
                 rewriter.getI32IntegerAttr(wait_instr_param_map["cycles"]))});
        rewriter.setInsertionPointToEnd(block);
        auto wait_instr = rewriter.create<vesyla::pasm::InstrOp>(
            op->getLoc(), id, type, param);
        new_time_op_vec.push_back(
            std::make_pair(prev_t + 1, wait_instr.getOperation()));
      }

      if (ordered_time_table.find(it->first) == ordered_time_table.end()) {
        ordered_time_table[it->first] = std::vector<mlir::Operation *>();
      }

      for (auto it2 = new_time_op_vec.begin(); it2 != new_time_op_vec.end();
           ++it2) {
        ordered_time_table[it->first].push_back(it2->second);
      }

      // print the ordered time table
      llvm::outs() << "Ordered time table: " << it->first << "\n";
      for (auto it2 = ordered_time_table[it->first].begin();
           it2 != ordered_time_table[it->first].end(); ++it2) {
        llvm::outs() << llvm::dyn_cast<vesyla::pasm::InstrOp>(*it2).getType()
                     << "\n";
      }
    }

    // add a yield instruction at the end of the block to separate the
    // operations
    auto yield_instr = rewriter.create<vesyla::pasm::YieldOp>(op->getLoc());

    // insert a RawOp for each cell at the end of the block
    for (auto it = ordered_time_table.begin(); it != ordered_time_table.end();
         ++it) {
      std::string label = it->first;
      int row = std::stoi(label.substr(0, label.find("_")));
      int col = std::stoi(label.substr(label.find("_") + 1));
      std::string raw_op_id = get_random_string(8);
      mlir::StringAttr id = rewriter.getStringAttr(raw_op_id);
      mlir::IntegerAttr row_attr =
          rewriter.getIntegerAttr(rewriter.getI32Type(), row);
      mlir::IntegerAttr col_attr =
          rewriter.getIntegerAttr(rewriter.getI32Type(), col);
      rewriter.setInsertionPointToEnd(block);
      auto raw_op = rewriter.create<vesyla::pasm::RawOp>(op->getLoc(), id,
                                                         row_attr, col_attr);
      // insert the instructions in the time table into the block
      mlir::Region &raw_op_body = raw_op.getBody();
      mlir::Block *raw_op_entry_block;
      if (raw_op_body.empty()) {
        raw_op_entry_block = rewriter.createBlock(&raw_op_body);
      } else {
        raw_op_entry_block = &raw_op_body.front();
      }
      rewriter.setInsertionPointToEnd(raw_op_entry_block);
      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        rewriter.clone(**it2);
      }
      // add a terminator to raw_op_block
      rewriter.setInsertionPointToEnd(raw_op_entry_block);
      rewriter.create<YieldOp>(raw_op->getLoc());
    }

    // start from the end, remove everything after the first yield
    // instruction in block
    bool found_yield = false;
    std::vector<mlir::Operation *> remove_ops;
    for (auto it = block->getOperations().rbegin();
         it != block->getOperations().rend(); ++it) {
      if (!found_yield) {
        if (llvm::isa<vesyla::pasm::YieldOp>(*it)) {
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
    // insert the yield instruction at the end of the block
    rewriter.create<vesyla::pasm::YieldOp>(op->getLoc());

    // print the whole block
    llvm::outs() << "Block contents after synchronization:\n";
    for (mlir::Operation &child_op : *block) {
      llvm::outs() << "Operation type: " << child_op.getName() << "\n";
    }
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

        // create a temporary file in the tmp directory with random name
        // and call the timing model builder to generate the timing model
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

    std::unordered_map<std::string, int> schedule_table;
    for (auto it = result.begin(); it != result.end(); ++it) {
      std::string key = it->first;
      std::string value = it->second;
      // if value is not starting with "[", then it is a number
      if (value[0] != '[') {
        schedule_table[key] = std::stoi(value);
      }
    }

    synchronize(&op, schedule_table, rewriter);

    return success();
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
    int row = this->row;
    int col = this->col;

    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceEpochOpRewriter>(&getContext(), component_map,
                                         component_path, tmp_path, row, col);
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
        // Note: This simple loop assumes a single block in the LoopOp
        // body and doesn't handle block arguments or complex control flow
        // transfer. If LoopOp body can have multiple blocks or branches,
        // a more sophisticated merging/cloning logic (like
        // inlineRegionBefore) is needed.
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

//===----------------------------------------------------------------------===//
// class MergeRawOpRewriter : public OpRewritePattern<EpochOp> {
// public:
//   using OpRewritePattern<EpochOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(EpochOp op,
//                                 PatternRewriter &rewriter) const final {
//     llvm::outs() << "MergeRawOpRewriter triggered on EpochOp.\n";

//     // get parent block
//     mlir::Block *moduleBlock = op.getOperation()->getBlock();
//     if (!moduleBlock) {
//       llvm::outs() << "Error: EpochOp is not inside a block.\n";
//       exit(-1);
//     }

//     // Iterate through all operations in the module's block
//     Operation *target_epoch_op = nullptr;
//     std::vector<Operation *> epoch_op_list;
//     llvm::outs() << "size of moduleBlock: "
//                  << moduleBlock->getOperations().size() << "\n";
//     for (mlir::Operation &child_op : moduleBlock->getOperations()) {
//       // check if it's a EpochOp
//       if (auto epoch_op = llvm::dyn_cast<EpochOp>(&child_op)) {
//         llvm::outs() << "Found EpochOp: " << epoch_op.getId() << "\n";
//         if (target_epoch_op == nullptr) {
//           target_epoch_op = &child_op;
//         } else {
//           epoch_op_list.push_back(&child_op);
//         }
//         mlir::Region &child_op_region = epoch_op.getBody();
//         if (child_op_region.empty()) {
//           llvm::outs() << "Error: EpochOp has no body.\n";
//           exit(-1);
//         }
//         mlir::Block *child_op_block = &child_op_region.front();
//         // Iterate through all operations in the child operation's block
//         for (mlir::Operation &child_op_child : *child_op_block) {
//           if (llvm::dyn_cast<RawOp>(&child_op_child) ||
//               llvm::dyn_cast<YieldOp>(&child_op_child)) {
//             // DO NOTHING
//           } else {
//             llvm::outs() << "Error: Illegal operation type in EpochOp: "
//                          << child_op_child.getName() << "\n";
//             exit(-1);
//           }
//         }
//       } else {
//         llvm::outs() << "Error: Illegal operation type in EpochOp: "
//                      << child_op.getName() << "\n";
//         return mlir::failure();
//       }
//     }

//     llvm::outs() << "Found " << epoch_op_list.size()
//                  << " EpochOps in the module.\n";
//     for (auto &child_op : epoch_op_list) {
//       llvm::outs() << "EpochOp: " <<
//       llvm::dyn_cast<EpochOp>(child_op).getId()
//                    << "\n";
//     }

//     if (epoch_op_list.size() == 0) {
//       return mlir::failure();
//     }

//     // create a map for insertion location
//     std::unordered_map<std::string, mlir::Operation *>
//         target_insertion_raw_op_map;
//     // get all the raw ops in the target epoch op
//     for (auto &child_op : *target_epoch_op->getRegion(0).begin()) {
//       if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
//         int row = raw_op.getRow();
//         int col = raw_op.getCol();
//         std::string label = std::to_string(row) + "_" + std::to_string(col);
//         if (target_insertion_raw_op_map.find(label) ==
//             target_insertion_raw_op_map.end()) {
//           target_insertion_raw_op_map[label] = &child_op;
//         } else {
//           llvm::outs() << "Error: RawOp already exists in the target epoch
//           op: "
//                        << label << "\n";
//           std::exit(-1);
//         }
//       }
//     }

//     // Iterate through all operations in the module's block
//     for (auto &child_op : epoch_op_list) {
//       if (auto epoch_op = llvm::dyn_cast<EpochOp>(child_op)) {
//         mlir::Region &child_op_region = epoch_op.getBody();
//         if (child_op_region.empty()) {
//           continue;
//         }
//         mlir::Block *child_op_block = &child_op_region.front();
//         // Iterate through all operations in the child operation's block
//         for (mlir::Operation &child_op_child : *child_op_block) {
//           if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op_child)) {
//             int row = raw_op.getRow();
//             int col = raw_op.getCol();
//             std::string label = std::to_string(row) + "_" +
//             std::to_string(col); if (target_insertion_raw_op_map.find(label)
//             !=
//                 target_insertion_raw_op_map.end()) {
//               // set the insertion point to the body of the target raw op,
//               just
//               // before the terminator
//               Operation *target_raw_op = target_insertion_raw_op_map[label];
//               mlir::Region &target_raw_op_region =
//                   llvm::dyn_cast<RawOp>(target_raw_op).getBody();
//               mlir::Block *target_raw_op_block;
//               if (target_raw_op_region.empty()) {
//                 target_raw_op_block =
//                     rewriter.createBlock(&target_raw_op_region);
//               } else {
//                 target_raw_op_block = &target_raw_op_region.front();
//               }

//               // clone the child op to the target raw op
//               mlir::Region &raw_op_body = raw_op.getBody();
//               mlir::Block *raw_op_entry_block;
//               if (raw_op_body.empty()) {
//                 raw_op_entry_block = rewriter.createBlock(&raw_op_body);
//               } else {
//                 raw_op_entry_block = &raw_op_body.front();
//               }
//               for (mlir::Operation &raw_op_child : *raw_op_entry_block) {
//                 if (auto instr_op = llvm::dyn_cast<InstrOp>(&raw_op_child)) {
//                   rewriter.setInsertionPoint(
//                       target_raw_op_block->getTerminator());
//                   rewriter.clone(raw_op_child);
//                 } else if (auto yield_op =
//                                llvm::dyn_cast<YieldOp>(&raw_op_child)) {
//                   // DO NOTHING
//                 } else {
//                   llvm::outs() << "Error: Illegal operation type in RawOp: "
//                                << raw_op_child.getName() << "\n";
//                   exit(-1);
//                 }
//               }
//             } else {
//               // clone this raw op to the target epoch op
//               mlir::Region &target_epoch_op_region =
//                   llvm::dyn_cast<EpochOp>(target_epoch_op).getBody();
//               mlir::Block *target_epoch_op_block =
//                   &target_epoch_op_region.front();
//               rewriter.setInsertionPoint(
//                   target_epoch_op_block->getTerminator());
//               auto new_raw_op = rewriter.clone(child_op_child);
//               // add the raw op to the target insertion map
//               target_insertion_raw_op_map[label] = new_raw_op;
//             }
//           }
//         }
//       }
//     }

//     // remove all the epoch ops
//     for (auto &child_op : epoch_op_list) {
//       if (auto epoch_op = llvm::dyn_cast<EpochOp>(child_op)) {
//         rewriter.eraseOp(epoch_op);
//       }
//     }

//     // print the whole block
//     llvm::outs() << "Block contents after merging:\n";
//     for (mlir::Operation &child_op : *moduleBlock) {
//       llvm::outs() << "Operation type: " << child_op.getName() << "\n";
//       if (auto epoch_op = llvm::dyn_cast<EpochOp>(&child_op)) {
//         llvm::outs() << "EpochOp ID: " << epoch_op.getId() << "\n";
//         mlir::Region &epochBodyRegion = epoch_op.getBody();
//         if (epochBodyRegion.empty()) {
//           llvm::outs() << "Error: EpochOp has no body.\n";
//           exit(-1);
//         }
//         mlir::Block *epochEntryBlock = &epochBodyRegion.front();
//         // Iterate through all operations in the child operation's block
//         for (mlir::Operation &epoch_child_op : *epochEntryBlock) {
//           if (auto raw_op = llvm::dyn_cast<RawOp>(&epoch_child_op)) {
//             llvm::outs() << "RawOp ID: " << raw_op.getId() << "\n";
//           } else if (auto yield_op = llvm::dyn_cast<vesyla::pasm::YieldOp>(
//                          &epoch_child_op)) {
//             // DO NOTHING
//           } else {
//             llvm::outs() << "Error: Illegal operation type in EpochOp: "
//                          << epoch_child_op.getName() << "\n";
//             exit(-1);
//           }
//         }
//       }
//     }

//     return mlir::success();
//   }
// };

// class MergeRawOp : public impl::MergeRawOpBase<MergeRawOp> {
// public:
//   using impl::MergeRawOpBase<MergeRawOp>::MergeRawOpBase;

//   void runOnOperation() {
//     llvm::outs() << "Running MergeRawOp pass...\n";

//     // Get the current module
//     mlir::ModuleOp module = getOperation();

//     // Create a pattern set
//     RewritePatternSet patterns(&getContext());
//     patterns.add<MergeRawOpRewriter>(&getContext());
//     FrozenRewritePatternSet patternSet(std::move(patterns));
//     // Apply the patterns to the module
//     if (failed(applyPatternsAndFoldGreedily(module, patternSet))) {
//       signalPassFailure();
//     }
//   }
// };

} // namespace
} // namespace vesyla::pasm