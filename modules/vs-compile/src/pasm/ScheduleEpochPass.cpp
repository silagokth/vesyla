#include "ScheduleEpochPass.hpp"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "pasm/Ops.hpp"
#include "llvm/Support/raw_ostream.h"
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#define SCALAR_REGISTER_COUNT 16

namespace vesyla::pasm {
#define GEN_PASS_DEF_SCHEDULEEPOCHPASS
#include "pasm/Passes.hpp.inc"

namespace {

//===----------------------------------------------------------------------===//
class ScheduleEpochPassRewriter : public OpRewritePattern<EpochOp> {
public:
  nlohmann::json component_map;
  std::string component_path;
  std::string tmp_path;
  int _row;
  int _col;

public:
  ScheduleEpochPassRewriter(MLIRContext *context, nlohmann::json component_map,
                            std::string component_path, std::string tmp_path,
                            int row_, int col_)
      : OpRewritePattern<EpochOp>(context), component_map(component_map),
        component_path(std::move(component_path)),
        tmp_path(std::move(tmp_path)), _row(row_), _col(col_) {}

private:
  mlir::Block *getOrCreateEntryBlock(mlir::Region &region,
                                     PatternRewriter &rewriter) const {
    mlir::Block *block;
    if (region.empty()) {
      block = rewriter.createBlock(&region);
    } else {
      block = &region.front();
    }
    return block;
  }

  mlir::Block *getEpochBodyEntryBlock(EpochOp epoch_op,
                                      PatternRewriter &rewriter) const {
    // Check if the EpochOp is valid
    if (!epoch_op) {
      llvm::outs() << "Error: Cannot find the EpochOp in the operation.\n";
      std::exit(EXIT_FAILURE);
    }

    mlir::Region &epochBodyRegion = epoch_op.getBody();
    mlir::Block *block = getOrCreateEntryBlock(epochBodyRegion, rewriter);
    return block;
  }

  std::optional<std::unordered_map<string, uint64_t>>
  create_act_0_instr(std::vector<int> indices) const {
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
    return std::make_optional<std::unordered_map<string, uint64_t>>(
        {{"mode", 0}, {"param", min_slot}, {"ports", ports}});
  }

  std::optional<std::unordered_map<string, uint64_t>>
  create_act_1_instr(std::vector<int> indices) const {
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

    return std::make_optional<std::unordered_map<string, uint64_t>>(
        {{"mode", 1}, {"param", param}, {"ports", ports}});
  }

  std::optional<std::unordered_map<string, uint64_t>>
  create_act_2_instr(std::vector<int> indices) const {
    uint64_t port_vec = 0;
    for (auto index : indices) {
      port_vec |= (1ULL << index); // set the bit for the port index
    }
    // print binary representation of port_vec
    llvm::outs() << "Port vector: " << std::bitset<64>(port_vec).to_string()
                 << "\n";

    // return param = 0 as it will be replaced when allocating registers
    return std::make_optional<std::unordered_map<string, uint64_t>>(
        {{"mode", 2}, {"param", 0}, {"ports", port_vec}});
  }

  std::unordered_map<string, uint64_t>
  create_act_instr(std::vector<int> indices) const {
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
      llvm::outs()
          << "Error: Cannot find a valid ACT instruction for the given "
             "port indices: ";
      for (auto index : indices) {
        llvm::outs() << index << " ";
      }
      llvm::outs() << "\n";
      exit(EXIT_FAILURE); // invalid combination of port indices for mode 1
    }
  }

  std::unordered_map<string, int> create_wait_instr(int cycle) const {
    llvm::outs() << "Create WAIT for cycle: " << cycle << "\n";
    return {{"mode", 0}, {"cycle", cycle}};
  }

  nlohmann::json op2json(mlir::Operation *op) const {
    nlohmann::json op_json;
    if (auto rop_op = llvm::dyn_cast<RopOp>(op)) {
      op_json["kind"] = "rop";
      op_json["id"] = rop_op.getId().str();
      op_json["row"] = rop_op.getRow();
      op_json["col"] = rop_op.getCol();
      op_json["slot"] = rop_op.getSlot();
      op_json["port"] = rop_op.getPort();

      // get its internal block
      mlir::Region &ropBodyRegion = rop_op.getBody();
      mlir::Block *ropEntryBlock;
      if (!ropBodyRegion.empty()) {
        ropEntryBlock = &ropBodyRegion.front();

        op_json["body"] = nlohmann::json::array();
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
                std::exit(EXIT_FAILURE);
              }
            }

            nlohmann::json instr_json;
            instr_json["id"] = id;
            instr_json["kind"] = type;
            instr_json["params"] = nlohmann::json::array();
            for (auto &param : param_map) {
              nlohmann::json param_json;
              param_json["name"] = param.first;
              param_json["value"] = param.second;
              instr_json["params"].push_back(param_json);
            }
            op_json["body"].push_back(instr_json);
          } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&rop_child_op)) {
            // DO NOTHING
          } else {
            llvm::outs() << "Illegal operation type in RopOp: "
                         << rop_child_op.getName() << "\n";
            std::exit(EXIT_FAILURE);
          }
        }
      }
    } else if (auto cop_op = llvm::dyn_cast<CopOp>(op)) {
      op_json["kind"] = "cop";
      op_json["id"] = cop_op.getId().str();
      op_json["row"] = cop_op.getRow();
      op_json["col"] = cop_op.getCol();

      // get its internal block
      mlir::Region &copBodyRegion = cop_op.getBody();
      mlir::Block *copEntryBlock;
      if (!copBodyRegion.empty()) {

        copEntryBlock = &copBodyRegion.front();

        op_json["body"] = nlohmann::json::array();
        for (mlir::Operation &cop_child_op : *copEntryBlock) {
          if (auto instr_op = llvm::dyn_cast<InstrOp>(&cop_child_op)) {
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
                std::exit(EXIT_FAILURE);
              }
            }

            nlohmann::json instr_json;
            instr_json["kind"] = type;
            instr_json["params"] = nlohmann::json::array();
            for (auto &param : param_map) {
              nlohmann::json param_json;
              param_json["name"] = param.first;
              param_json["value"] = param.second;
              instr_json["params"].push_back(param_json);
            }
            op_json["body"].push_back(instr_json);
          } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&cop_child_op)) {
            // DO NOTHING
          } else {
            llvm::outs() << "Illegal operation type in CopOp: "
                         << cop_child_op.getName() << "\n";
            std::exit(EXIT_FAILURE);
          }
        }
      }
    } else {
      llvm::outs() << "Unsupported operation type for JSON conversion: "
                   << op->getName() << "\n";
      std::exit(EXIT_FAILURE);
    }

    return op_json;
  }

  void json2op(nlohmann::json op_json, PatternRewriter &rewriter) const {
    if (op_json["kind"].get<std::string>() == "rop") {
      auto rop_op = rewriter.create<RopOp>(
          rewriter.getUnknownLoc(),
          rewriter.getStringAttr(op_json["id"].get<std::string>()),
          rewriter.getI32IntegerAttr(op_json["row"].get<int>()),
          rewriter.getI32IntegerAttr(op_json["col"].get<int>()),
          rewriter.getI32IntegerAttr(op_json["slot"].get<int>()),
          rewriter.getI32IntegerAttr(op_json["port"].get<int>()));

      // get its internal block
      mlir::Region &ropBodyRegion = rop_op.getBody();
      mlir::Block *ropEntryBlock;
      if (ropBodyRegion.empty()) {
        ropEntryBlock = rewriter.createBlock(&ropBodyRegion);
      } else {
        ropEntryBlock = &ropBodyRegion.front();
      }
      rewriter.setInsertionPointToEnd(ropEntryBlock);

      for (auto &instr : op_json["body"]) {
        std::unordered_map<std::string, std::string> param_map;
        for (auto &param : instr["params"]) {
          param_map[param["name"].get<std::string>()] =
              param["value"].get<std::string>();
        }

        mlir::StringAttr id =
            rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8));
        mlir::StringAttr type =
            rewriter.getStringAttr(instr["kind"].get<std::string>());
        llvm::SmallVector<mlir::NamedAttribute> attrs;

        for (const auto &param : param_map) {
          // check if it's a number or a string
          if (std::isdigit(param.second[0])) {
            attrs.push_back(rewriter.getNamedAttr(
                param.first,
                rewriter.getI32IntegerAttr(std::stoi(param.second))));
          } else {
            attrs.push_back(rewriter.getNamedAttr(
                param.first, rewriter.getStringAttr(param.second)));
          }
        }
        mlir::DictionaryAttr param = rewriter.getDictionaryAttr(attrs);
        rewriter.create<InstrOp>(
            rop_op.getLoc(),
            rewriter.getStringAttr(instr["id"].get<std::string>()),
            rewriter.getStringAttr(instr["kind"].get<std::string>()), param);
      }
      // insert a yield operation at the end of the RopOp
      rewriter.create<YieldOp>(rop_op.getLoc());

    } else if (op_json["kind"] == "cop") {
      auto cop_op = rewriter.create<CopOp>(
          rewriter.getUnknownLoc(),
          rewriter.getStringAttr(op_json["id"].get<std::string>()),
          rewriter.getI32IntegerAttr(op_json["row"].get<int>()),
          rewriter.getI32IntegerAttr(op_json["col"].get<int>()));
      // get its internal block
      mlir::Region &copBodyRegion = cop_op.getBody();
      mlir::Block *copEntryBlock;
      if (copBodyRegion.empty()) {
        copEntryBlock = rewriter.createBlock(&copBodyRegion);
      } else {
        copEntryBlock = &copBodyRegion.front();
      }
      rewriter.setInsertionPointToEnd(copEntryBlock);
      for (auto &instr : op_json["body"]) {
        std::unordered_map<std::string, std::string> param_map;
        for (auto &param : instr["params"]) {
          param_map[param["name"].get<std::string>()] =
              param["value"].get<std::string>();
        }

        mlir::StringAttr id =
            rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8));
        mlir::StringAttr type =
            rewriter.getStringAttr(instr["kind"].get<std::string>());
        llvm::SmallVector<mlir::NamedAttribute> attrs;

        for (const auto &param : param_map) {
          // check if it's a number or a string
          if (std::isdigit(param.second[0])) {
            attrs.push_back(rewriter.getNamedAttr(
                param.first,
                rewriter.getI32IntegerAttr(std::stoi(param.second))));
          } else {
            attrs.push_back(rewriter.getNamedAttr(
                param.first, rewriter.getStringAttr(param.second)));
          }
        }
        mlir::DictionaryAttr param = rewriter.getDictionaryAttr(attrs);
        rewriter.create<InstrOp>(
            cop_op.getLoc(),
            rewriter.getStringAttr(instr["id"].get<std::string>()),
            rewriter.getStringAttr(instr["kind"].get<std::string>()), param);
      }
      // insert a yield operation at the end of the CopOp
      rewriter.create<YieldOp>(cop_op.getLoc());
    } else {
      llvm::outs() << "Unsupported operation kind: "
                   << op_json["kind"].get<std::string>() << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  void replace_time_in_instr_param(
      EpochOp &op, std::unordered_map<std::string, int> &schedule_table,
      PatternRewriter &rewriter) const {

    auto epoch_op = op;
    mlir::Region &epoch_region = epoch_op.getBody();
    if (epoch_region.empty()) {
      return;
    }
    mlir::Block *epoch_block = &epoch_region.front();
    for (mlir::Operation &child_op : *epoch_block) {
      if (auto rop_op = llvm::dyn_cast<RopOp>(&child_op)) {
        mlir::Region &rop_region = rop_op.getBody();
        if (rop_region.empty()) {
          continue;
        }
        mlir::Block *rop_entry_block = &rop_region.front();
        for (mlir::Operation &instr_op : *rop_entry_block) {
          if (auto instr = llvm::dyn_cast<InstrOp>(&instr_op)) {
            mlir::DictionaryAttr current_instr_params = instr.getParam();
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
                  mlir::Attribute new_attr_value =
                      rewriter.getIntegerAttr(rewriter.getI32Type(), int_value);
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

            // If any attributes were changed, create a new DictionaryAttr
            // and update the operation
            if (params_changed) {
              mlir::DictionaryAttr new_instr_params =
                  rewriter.getDictionaryAttr(updated_attrs);
              instr->setAttr("param", new_instr_params);
            }
          }
        }

      } else if (auto cop_op = llvm::dyn_cast<CopOp>(&child_op)) {
        mlir::Region &cop_region = cop_op.getBody();
        if (cop_region.empty()) {
          continue;
        }
        mlir::Block *cop_entry_block = &cop_region.front();
        for (mlir::Operation &instr_op : *cop_entry_block) {
          if (auto instr = llvm::dyn_cast<InstrOp>(&instr_op)) {
            mlir::DictionaryAttr current_instr_params = instr.getParam();
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
                  mlir::Attribute new_attr_value =
                      rewriter.getIntegerAttr(rewriter.getI32Type(), int_value);
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

            // If any attributes were changed, create a new DictionaryAttr
            // and update the operation
            if (params_changed) {
              mlir::DictionaryAttr new_instr_params =
                  rewriter.getDictionaryAttr(updated_attrs);
              instr->setAttr("param", new_instr_params);
            }
          }
        }
      }
    }
  }

  void reshape_instr(EpochOp &op, PatternRewriter &rewriter) const {
    // Get the block to insert the new operations
    mlir::Block *block = getEpochBodyEntryBlock(op, rewriter);

    std::vector<mlir::Operation *> ops_to_erase;
    std::vector<mlir::Operation *> ops_to_reshape;

    for (mlir::Operation &child_op : *block) {
      ops_to_reshape.push_back(&child_op);
    }

    for (mlir::Operation *child_op : ops_to_reshape) {
      if (auto rop_op = llvm::dyn_cast<RopOp>(child_op)) {
        // convert the RopOp to JSON
        nlohmann::json rop_json = op2json(child_op);
        // create command
        // create a temporary file in the tmp directory with random name
        // and call the timing model builder to generate the timing model
        // expression.
        std::string random_str = vesyla::util::Common::gen_random_string(10);
        LOG_DEBUG << "Random string for temporary file: " << random_str;
        std::string input_filename = tmp_path + "/" + random_str + ".json";
        std::string output_filename = tmp_path + "/" + random_str + "_out.json";
        LOG_DEBUG << "Input filename: " << input_filename;
        LOG_DEBUG << "Output filename: " << output_filename;
        std::string label = std::to_string(rop_json["row"].get<int>()) + "_" +
                            std::to_string(rop_json["col"].get<int>()) + "_" +
                            std::to_string(rop_json["slot"].get<int>()) + "_" +
                            std::to_string(rop_json["port"].get<int>());
        LOG_DEBUG << "Label: " << label;
        if (component_map.find(label) == component_map.end()) {
          llvm::outs() << "Error: Cannot find the component : " << label
                       << "\n";
          std::exit(EXIT_FAILURE);
        }
        std::string command = component_path + "/resources/" +
                              component_map[label].get<std::string>() +
                              "/compile_util reshape_instr " + input_filename +
                              " " + output_filename;

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

        // read the output file
        std::ifstream output_file(output_filename);
        if (!output_file.is_open()) {
          llvm::outs() << "Error: Failed to open output file.\n";
          std::exit(EXIT_FAILURE);
        }
        nlohmann::json output_json = nlohmann::json::parse(output_file);
        output_file.close();
        // delete the temporary files
        std::filesystem::remove(input_filename);
        std::filesystem::remove(output_filename);
        if (output_json["kind"].get<std::string>() != "rop") {
          llvm::outs() << "Error: Output JSON is not a RopOp.\n";
          std::exit(EXIT_FAILURE);
        }
        // convert the output JSON to operation
        rewriter.setInsertionPointToEnd(block);
        json2op(output_json, rewriter);
        // remove the original RopOp
        ops_to_erase.push_back(child_op);
      } else if (auto cop_op = llvm::dyn_cast<CopOp>(child_op)) {
        // convert the CopOp to JSON
        nlohmann::json cop_json = op2json(child_op);
        // create command
        // create a temporary file in the tmp directory with random name
        // and call the timing model builder to generate the timing model
        // expression.
        std::string random_str = vesyla::util::Common::gen_random_string(10);
        LOG_DEBUG << "Random string for temporary file: " << random_str;
        std::string input_filename = tmp_path + "/" + random_str + ".json";
        std::string output_filename = tmp_path + "/" + random_str + "_out.json";
        LOG_DEBUG << "Input filename: " << input_filename;
        LOG_DEBUG << "Output filename: " << output_filename;
        std::string label = std::to_string(cop_json["row"].get<int>()) + "_" +
                            std::to_string(cop_json["col"].get<int>());
        LOG_DEBUG << "Label: " << label;
        if (component_map.find(label) == component_map.end()) {
          llvm::outs() << "Error: Cannot find the component : " << label
                       << "\n";
          std::exit(EXIT_FAILURE);
        }
        std::string command = component_path + "/resources/" +
                              component_map[label].get<std::string>() +
                              "/compile_util reshape_instr " + input_filename +
                              " " + output_filename;

        llvm::outs() << "Executing command: " << command << "\n";

        std::ofstream file(input_filename);
        if (!file.is_open()) {
          llvm::outs() << "Error: Failed to create temporary file.\n";
          std::exit(EXIT_FAILURE);
        }
        file << cop_json.dump(4);
        file.close();
        int result = system(command.c_str());
        if (result != 0) {
          llvm::outs() << "Error: Command failed with error code: " << result
                       << "\n";
          std::exit(EXIT_FAILURE);
        }

        // read the output file
        std::ifstream output_file(output_filename);
        if (!output_file.is_open()) {
          llvm::outs() << "Error: Failed to open output file.\n";
          std::exit(EXIT_FAILURE);
        }
        nlohmann::json output_json = nlohmann::json::parse(output_file);
        output_file.close();
        // delete the temporary files
        std::filesystem::remove(input_filename);
        std::filesystem::remove(output_filename);
        if (output_json["kind"].get<std::string>() != "cop") {
          llvm::outs() << "Error: Output JSON is not a CopOp.\n";
          std::exit(EXIT_FAILURE);
        }
        // convert the output JSON to operation
        rewriter.setInsertionPointToEnd(block);
        json2op(output_json, rewriter);
        // remove the original CopOp
        ops_to_erase.push_back(child_op);
      } else if (auto raw_op = llvm::dyn_cast<RawOp>(child_op)) {
        // DO NOTHING
      } else if (auto cstr_op = llvm::dyn_cast<CstrOp>(child_op)) {
        // DO NOTHING
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(child_op)) {
        // DO NOTHING
      } else {
        llvm::outs() << "Illegal operation type in EpochOp: "
                     << child_op->getName() << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // erase the original operations
    for (auto op : ops_to_erase) {
      rewriter.eraseOp(op);
    } // end of erasing operations
  }

  void insert_rop_instructions(
      vector<mlir::Operation *> &rop_ops, int t, PatternRewriter &rewriter,
      std::map<int, mlir::Operation *> &cell_time_table) const {
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
          while (cell_time_table.find(curr_t) != cell_time_table.end()) {
            curr_t--;
          }
          cell_time_table[curr_t] = rop_child_op;
          curr_t--;
        }
      }
    }
  }

  void print_time_table(
      std::unordered_map<string, std::map<int, mlir::Operation *>> &time_table)
      const {
    // print the time table
    for (auto it = time_table.begin(); it != time_table.end(); ++it) {
      auto cell_label = it->first;
      auto cell_time_table = it->second;
      llvm::outs() << "Time table: " << cell_label << "\n";
      for (auto cell_it = cell_time_table.begin();
           cell_it != cell_time_table.end(); ++cell_it) {
        auto cycle = cell_it->first;
        auto op = llvm::dyn_cast<vesyla::pasm::InstrOp>(cell_it->second);
        llvm::outs() << "  " << cycle << ": ";
        op.print(llvm::outs());
        llvm::outs() << "\n";
      }
    }
  }

  vesyla::pasm::InstrOp compose_act_mlir_op(
      EpochOp op, PatternRewriter &rewriter,
      std::unordered_map<string, uint64_t> &act_instr_param_map) const {
    return rewriter.create<vesyla::pasm::InstrOp>(
        op->getLoc(),
        rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8)),
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

  vesyla::pasm::InstrOp compose_calc_mlir_op(
      EpochOp op, PatternRewriter &rewriter,
      std::unordered_map<string, int> &calc_instr_param_map) const {
    return rewriter.create<vesyla::pasm::InstrOp>(
        op->getLoc(),
        rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8)),
        rewriter.getStringAttr("calc"),
        rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                              calc_instr_param_map["mode"])),
            rewriter.getNamedAttr(
                "operand1",
                rewriter.getI32IntegerAttr(calc_instr_param_map["operand1"])),
            rewriter.getNamedAttr("operand2_sd",
                                  rewriter.getI32IntegerAttr(
                                      calc_instr_param_map["operand2_sd"])),
            rewriter.getNamedAttr(
                "operand2",
                rewriter.getI32IntegerAttr(calc_instr_param_map["operand2"])),
            rewriter.getNamedAttr(
                "result",
                rewriter.getI32IntegerAttr(calc_instr_param_map["result"])),
        }));
  }

  std::map<int, mlir::Operation *> &getOrCreateCellTimeTable(
      std::unordered_map<string, std::map<int, mlir::Operation *>> &time_table,
      const std::string &label) const {
    if (time_table.find(label) == time_table.end())
      time_table[label] = std::map<int, mlir::Operation *>();
    return time_table[label];
  }

  void
  create_time_table_entry(std::map<int, mlir::Operation *> &cell_time_table,
                          int t, vesyla::pasm::InstrOp &instr_op) const {

    if (cell_time_table.find(t) == cell_time_table.end()) {
      cell_time_table[t] = instr_op.getOperation();
    } else {
      llvm::outs() << "Error: time table already has the entry: " << t << "("
                   << cell_time_table[t]->getName() << ")"
                   << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<int>
  get_absolute_port_indices(vector<mlir::Operation *> &rop_ops) const {
    std::vector<int> slot_port_index_list;
    for (auto op : rop_ops) {
      auto rop_op = llvm::dyn_cast<RopOp>(op);
      int slot = rop_op.getSlot();
      int port = rop_op.getPort();
      slot_port_index_list.push_back(slot * 4 + port);
    }
    return slot_port_index_list;
  }

  std::unordered_map<std::string, std::vector<mlir::Operation *>>
  get_rop_ops_for_cycle(
      int currentCycle,
      std::unordered_map<mlir::Operation *, int> time_table_rop) const {
    std::unordered_map<std::string, std::vector<mlir::Operation *>>
        rop_ops_at_t;
    for (auto it = time_table_rop.begin(); it != time_table_rop.end(); ++it) {
      // only keep the rop ops for the current cycle
      if (it->second != currentCycle)
        continue;

      auto rop_op = llvm::dyn_cast<RopOp>(it->first);
      int row = rop_op.getRow();
      int col = rop_op.getCol();
      std::string label = std::to_string(row) + "_" + std::to_string(col);

      // if the cell label is not found, create an entry
      if (rop_ops_at_t.find(label) == rop_ops_at_t.end()) {
        rop_ops_at_t[label] = std::vector<mlir::Operation *>();
      }
      rop_ops_at_t[label].push_back(it->first);
    }

    return rop_ops_at_t;
  }

  std::map<int, std::vector<int>> create_reg_alloc_table() const {
    std::map<int, std::vector<int>> reg_alloc_table;
    for (int reg_i = 0; reg_i < SCALAR_REGISTER_COUNT; reg_i++) {
      reg_alloc_table[reg_i] = std::vector<int>();
    }

    return reg_alloc_table;
  }

  void changeActMode2Param(mlir::Operation *act_op, int first_reg_address,
                           PatternRewriter &rewriter) const {
    mlir::DictionaryAttr new_instr_params = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(2)),
        rewriter.getNamedAttr("param",
                              rewriter.getI32IntegerAttr(first_reg_address)),
        rewriter.getNamedAttr("ports", rewriter.getI64IntegerAttr(0)),
    });
    act_op->setAttr("param", new_instr_params);
  }

  int allocateAndScheduleActMode2PrepInstructions(
      std::map<int, std::vector<int>> &reg_alloc_table,
      std::map<int, mlir::Operation *> &cell_time_table,
      PatternRewriter &rewriter, EpochOp op, int cycle, uint64_t ports) const {

    int potential_first_reg = 4;
    int num_regs_needed = 4; // TODO: make this configurable
    int first_reg;

    // check if the first reg is available
    if (reg_alloc_table[potential_first_reg].empty()) {
      first_reg = potential_first_reg;
    } else if (reg_alloc_table[potential_first_reg + num_regs_needed].empty()) {
      first_reg = potential_first_reg + num_regs_needed;
    } else {
      llvm::outs()
          << "Error: Cannot allocate registers for activation mode 2 at cycle "
          << cycle
          << ". [r4->r7] and [r8->r11] are not available in controller.\n";
      exit(EXIT_FAILURE);
    }

    // add act cycle to the register allocation table
    for (int i = 0; i < num_regs_needed; i++) {
      reg_alloc_table[first_reg + i].push_back(cycle);
    }

    // get the list of prep instructions for activation mode 2
    auto instrs_to_place = get_act_mode2_prep_instrs(first_reg, ports);
    int num_instrs = instrs_to_place.size();
    int cycles_needed = num_instrs;

    // find the first empty slot
    int max_attempts = 1000; // or some reasonable limit
    int attempts = 0;
    do {
      cycle--;
      attempts++;
      if (attempts > max_attempts) {
        llvm::outs() << "Error: Too many attempts to allocate registers for "
                        "ACT mode 2 prep instructions at cycle " << cycle << ".\n";
        exit(EXIT_FAILURE);
      }
      if (cell_time_table.find(cycle) != cell_time_table.end()) {
        continue; // skip if there are already operations scheduled at this
                  // cycle
      }

      // check if the regs are available
      bool registers_available = true;
      for (int i = 0; i < num_regs_needed; i++) {
        // check if the table contains the cycle for the register
        if (!reg_alloc_table[first_reg + i].empty() &&
            std::find(reg_alloc_table[first_reg + i].begin(),
                      reg_alloc_table[first_reg + i].end(),
                      cycle) != reg_alloc_table[first_reg + i].end()) {
          registers_available = false;
          break;
        }
      }
      if (!registers_available) {
        continue; // skip if registers are not available
      }
      // else allocate them for the current cycle
      for (int i = 0; i < num_regs_needed; i++) {
        reg_alloc_table[first_reg + i].push_back(cycle);
      }
      llvm::outs() << "Allocating registers r" << first_reg << " to r"
                   << (first_reg + num_regs_needed - 1)
                   << " for ACT mode 2 prep "
                   << "instructions for ports "
                   << std::bitset<64>(ports).to_string() << " at cycle "
                   << cycle << "\n";

      // place a prep instruction in the time table
      auto prep_op = compose_calc_mlir_op(op, rewriter,
                                          instrs_to_place[cycles_needed - 1]);
      create_time_table_entry(cell_time_table, cycle, prep_op);
      cycles_needed--;

    } while (cycles_needed != 0);

    return first_reg; // return the first register address
  }

  std::vector<std::unordered_map<string, int>>
  get_act_mode2_prep_instrs(int first_reg_address, uint64_t ports) const {
    std::vector<std::unordered_map<string, int>> act_mode2_prep_instrs;

    int num_regs_needed = 4;
    for (int loop_index = 1; loop_index <= 4; ++loop_index) {
      uint64_t regValueShouldBe = ports >> (16 * (loop_index - 1)) & 0xFFFF;
      int reg_i = first_reg_address + (loop_index - 1);

      llvm::outs() << "r" << reg_i << " should be: "
                   << std::bitset<16>(regValueShouldBe).to_string() << "\n";

      // Calculate shift for MSB and LSB
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
            {"mode", 23},            // addh
            {"operand1", 0},         // reg 0
            {"operand2_sd", 0},      // use immediate
            {"operand2", msb_value}, // 8 MSBs
            {"result", reg_i}        // store in reg_i
        };
        act_mode2_prep_instrs.push_back(load_msb);
      } else {
        add_lsb_reg_address = 0; // r0 is always 0
      }

      if (lsb_value != 0) {
        // set reg_i 8 LSB
        std::unordered_map<std::string, int> load_lsb = {
            {"mode", 1},                       // add
            {"operand1", add_lsb_reg_address}, // reg_i if msb is not 0, else r0
            {"operand2_sd", 0},                // use immediate
            {"operand2", lsb_value},           // 8 LSBs
            {"result", reg_i}                  // store in reg_i
        };
        act_mode2_prep_instrs.push_back(load_lsb);
      }
    }

    return act_mode2_prep_instrs;
  }

  bool op_is_act_mode2(mlir::Operation *op) const {
    auto instr_type = op->getAttr("type");
    auto instr_type_str = mlir::dyn_cast_or_null<mlir::StringAttr>(instr_type);
    if (!instr_type_str || instr_type_str.getValue() != "act") {
      return false; // not an act operation
    }

    auto instr_params = op->getAttr("param");
    auto instr_params_dict =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(instr_params);
    if (!instr_params_dict) {
      return false; // invalid parameters
    }

    auto mode_attr = instr_params_dict.get("mode");
    auto mode_attr_int = mlir::dyn_cast_or_null<mlir::IntegerAttr>(mode_attr);
    if (!mode_attr_int || mode_attr_int.getInt() != 2) {
      return false; // not mode 2
    }

    return true;
  }

  uint64_t get_ports_from_act_mode2_instr(mlir::Operation *op) const {
    auto instr_params = op->getAttr("param");
    auto instr_params_dict =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(instr_params);
    if (!instr_params_dict) {
      llvm::outs() << "Error: Invalid parameters for act mode 2 instruction.\n";
      std::exit(EXIT_FAILURE);
    }

    auto ports_attr = instr_params_dict.get("ports");
    auto ports_attr_int = mlir::dyn_cast_or_null<mlir::IntegerAttr>(ports_attr);
    if (!ports_attr_int) {
      llvm::outs() << "Error: Invalid ports attribute for act mode 2 "
                      "instruction.\n";
      std::exit(EXIT_FAILURE);
    }

    return ports_attr_int.getInt();
  }

  void insert_cop_instructions(
      mlir::Block *copEntryBlock,
      const std::unordered_map<std::string, int> &schedule_table,
      std::map<int, mlir::Operation *> &cell_time_table) const {
    int instr_count = 0;
    for (mlir::Operation &cop_child_op : *copEntryBlock) {
      if (auto instr_op = llvm::dyn_cast<InstrOp>(&cop_child_op)) {
        std::string instr_anchor =
            instr_op.getId().str() + "_e" + std::to_string(instr_count);

        if (schedule_table.find(instr_anchor) == schedule_table.end()) {
          llvm::outs() << "Error: Cannot find the instruction anchor: "
                       << instr_anchor << "\n";
          std::exit(EXIT_FAILURE);
        }

        int t = schedule_table.at(instr_anchor);

        if (cell_time_table.find(t) != cell_time_table.end()) {
          llvm::outs() << "Error: time table already has the entry: " << t
                       << "(" << cell_time_table[t]->getName() << ")"
                       << "\n";
          std::exit(EXIT_FAILURE);
        }

        cell_time_table[t] = &cop_child_op;
        instr_count++;
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&cop_child_op)) {
        // DO NOTHING
      } else {
        llvm::outs() << "Illegal operation type in CopOp: "
                     << cop_child_op.getName() << "\n";
        std::exit(EXIT_FAILURE);
      }
    }
  }

  void synchronize(EpochOp &op,
                   std::unordered_map<std::string, int> &schedule_table,
                   PatternRewriter &rewriter) const {

    // Get the block to insert the new operations
    mlir::Block *block = getEpochBodyEntryBlock(op, rewriter);

    rewriter.setInsertionPointToEnd(block);

    // initialize the time_table and ordered_time_table, add the label for
    // every cell in the fabric
    std::unordered_map<string, std::map<int, mlir::Operation *>> time_table;
    std::unordered_map<string, std::vector<mlir::Operation *>>
        ordered_time_table;
    for (int r = 0; r < _row; r++) {
      for (int c = 0; c < _col; c++) {
        std::string label = std::to_string(r) + "_" + std::to_string(c);
        time_table[label] = std::map<int, mlir::Operation *>();
        ordered_time_table[label] = std::vector<mlir::Operation *>();
      }
    }

    // initialize the time_tables of rop and cop operations
    std::unordered_map<mlir::Operation *, int> time_table_rop;
    std::unordered_map<mlir::Operation *, int> time_table_cop;
    for (mlir::Operation &child_op : *block) {
      if (auto rop_op = llvm::dyn_cast<RopOp>(&child_op)) {
        time_table_rop[&child_op] = schedule_table[rop_op.getId().str()];
      } else if (auto cop_op = llvm::dyn_cast<CopOp>(&child_op)) {
        time_table_cop[&child_op] = schedule_table[cop_op.getId().str()];
      } else if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
        // DO NOTHING
      } else if (auto instr_op = llvm::dyn_cast<InstrOp>(&child_op)) {
        // DO NOTHING
      } else if (auto cstr_op = llvm::dyn_cast<CstrOp>(&child_op)) {
        // DO NOTHING
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&child_op)) {
        // DO NOTHING
      } else {
        llvm::outs()
            << "Illegal operation type in EpochOp for synchronization: "
            << child_op.getName() << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // place all COPs
    for (auto it = time_table_cop.begin(); it != time_table_cop.end(); ++it) {
      auto cop_op = llvm::dyn_cast<CopOp>(it->first);
      std::string label = std::to_string(cop_op.getRow()) + "_" +
                          std::to_string(cop_op.getCol());
      auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);

      mlir::Region &copBodyRegion = cop_op.getBody();
      mlir::Block *copEntryBlock =
          getOrCreateEntryBlock(copBodyRegion, rewriter);
      insert_cop_instructions(copEntryBlock, schedule_table, cell_time_table);
    }

    // place the ACT instruction of all ROPs
    int total_latency = schedule_table["total_latency"];
    std::unordered_map<string, bool> cell_contains_act_mode2;
    for (int t = 0; t < total_latency; t++) {
      std::unordered_map<std::string, std::vector<mlir::Operation *>>
          rop_ops_at_t = get_rop_ops_for_cycle(t, time_table_rop);

      // if no rop ops are schedule at t, continue
      if (rop_ops_at_t.empty())
        continue;

      // Debug print
      llvm::outs() << "ROPs at t=" << t << ":\n";

      // for all rop operations at time t
      for (auto it = rop_ops_at_t.begin(); it != rop_ops_at_t.end(); ++it) {
        std::string label = it->first;
        cell_contains_act_mode2[label] =
            cell_contains_act_mode2.find(label) != cell_contains_act_mode2.end()
                ? cell_contains_act_mode2[label]
                : false;
        auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);
        std::vector<mlir::Operation *> rop_ops = it->second;
        std::vector<int> slot_port_index_list =
            get_absolute_port_indices(rop_ops);

        // Debug print
        llvm::outs() << "  - cell " << label << ": [";
        for (int rop_i = 0; rop_i < rop_ops.size(); rop_i++) {
          auto rop_op = llvm::dyn_cast<RopOp>(rop_ops[rop_i]);
          auto rop_port = slot_port_index_list[rop_i];
          if (rop_i == rop_ops.size() - 1)
            llvm::outs() << rop_op.getId() << " (" << rop_port << ")]\n";
          else
            llvm::outs() << rop_op.getId() << " (" << rop_port << "), ";
        }

        // place act instructions in time table
        auto act_instr_param_map = create_act_instr(slot_port_index_list);

        // if the act instruction mode is 2, compose the act vectors
        // and place them in the time table
        if (act_instr_param_map["mode"] == 2)
          cell_contains_act_mode2[label] = true;
        //          int vector_address =
        //              compose_act_vectors(register_allocation_table, t,
        //              cell_time_table,
        //                                  act_instr_param_map["ports"]);
        //          act_instr_param_map["param"] = vector_address;
        //          print_time_table(time_table);
        //        }

        auto act_instr = compose_act_mlir_op(op, rewriter, act_instr_param_map);
        create_time_table_entry(cell_time_table, t, act_instr);
      }
    }

    for (auto &cell_entry : time_table) {
      // go throught the time table in reverse order and allocate registers
      const std::string &cell_label = cell_entry.first;
      auto &cell_time_table = cell_entry.second;

      // if the cell does not contain act mode 2, continue
      if (!cell_contains_act_mode2[cell_label])
        continue;

      std::vector<std::pair<int, mlir::Operation *>> cycles(
          cell_time_table.begin(), cell_time_table.end());

      print_time_table(time_table);

      std::map<int, std::vector<int>> register_allocation_table =
          create_reg_alloc_table();
      for (const auto &cycle_entry : cycles) {
        int cycle = cycle_entry.first;
        mlir::Operation *act_op = cycle_entry.second;
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
      std::unordered_map<std::string, std::vector<mlir::Operation *>>
          rop_ops_at_t = get_rop_ops_for_cycle(t, time_table_rop);
      for (auto it = rop_ops_at_t.begin(); it != rop_ops_at_t.end(); ++it) {
        std::string label = it->first;
        auto &cell_time_table = getOrCreateCellTimeTable(time_table, label);
        std::vector<mlir::Operation *> rop_ops = it->second;
        insert_rop_instructions(rop_ops, t, rewriter, cell_time_table);
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

    // shift the time table
    for (auto it = time_table.begin(); it != time_table.end(); ++it) {
      std::map<int, mlir::Operation *> new_time_table;
      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        new_time_table[it2->first + min_shift_time] = it2->second;
      }
      time_table[it->first] = std::move(new_time_table);
    }

    llvm::outs() << "Time table after shifting:\n";
    print_time_table(time_table);

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
        // start from the smallest time, insert the WAIT instructions if
        // there is a gap between consecutive operations
        for (int i = 0; i < time_op_vec.size(); i++) {
          int curr_t = time_op_vec[i].first;
          if (curr_t - prev_t > 1) {
            // create a WAIT instruction
            rewriter.setInsertionPointToEnd(block);
            auto wait_instr_param_map = create_wait_instr(curr_t - prev_t - 1);
            mlir::StringAttr id = rewriter.getStringAttr(
                vesyla::util::Common::gen_random_string(8));
            mlir::StringAttr type = rewriter.getStringAttr("wait");
            mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
                {rewriter.getNamedAttr(
                     "mode",
                     rewriter.getI32IntegerAttr(wait_instr_param_map["mode"])),
                 rewriter.getNamedAttr("cycle",
                                       rewriter.getI32IntegerAttr(
                                           wait_instr_param_map["cycle"]))});
            auto wait_instr = rewriter.create<InstrOp>(
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

      // insert a wait instruction at the end if the last operation is not
      // the total_latency-1
      if (prev_t != total_latency - 1) {
        rewriter.setInsertionPointToEnd(block);
        auto wait_instr_param_map =
            create_wait_instr(total_latency - 1 - prev_t);
        mlir::StringAttr id =
            rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8));
        mlir::StringAttr type = rewriter.getStringAttr("wait");
        mlir::DictionaryAttr param = rewriter.getDictionaryAttr(
            {rewriter.getNamedAttr("mode", rewriter.getI32IntegerAttr(
                                               wait_instr_param_map["mode"])),
             rewriter.getNamedAttr(
                 "cycle",
                 rewriter.getI32IntegerAttr(wait_instr_param_map["cycle"]))});
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
        llvm::dyn_cast<vesyla::pasm::InstrOp>(*it2).print(llvm::outs());
        llvm::outs() << "\n";
        llvm::outs().flush();
      }
    }

    // add a yield instruction at the end of the block to separate the
    // operations
    rewriter.setInsertionPointToEnd(block);
    auto yield_instr = rewriter.create<vesyla::pasm::YieldOp>(op->getLoc());

    // insert a RawOp for each cell at the end of the block
    for (auto it = ordered_time_table.begin(); it != ordered_time_table.end();
         ++it) {
      std::string label = it->first;
      int row = std::stoi(label.substr(0, label.find("_")));
      int col = std::stoi(label.substr(label.find("_") + 1));
      std::string raw_op_id = vesyla::util::Common::gen_random_string(8);
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

      for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
        rewriter.setInsertionPointToEnd(raw_op_entry_block);
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
    tm::TimingModel model;
    tm::Solver solver(tmp_path);
    // Get the EpochOp's ID
    std::string originalIdStr = op.getId().str();
    mlir::Block *entryBlock = getEpochBodyEntryBlock(op, rewriter);
    std::set<std::string> operation_type_set;
    for (mlir::Operation &child_op : *entryBlock) {
      // cast to the correct type
      if (auto rop_op = llvm::dyn_cast<RopOp>(&child_op)) {

        nlohmann::json rop_json = op2json(rop_op.getOperation());

        // create a temporary file in the tmp directory with random name
        // and call the timing model builder to generate the timing model
        // expression.
        std::string random_str = vesyla::util::Common::gen_random_string(10);
        LOG_DEBUG << "Random string for temporary file: " << random_str;
        std::string input_filename = tmp_path + "/" + random_str + ".json";
        std::string output_filename = tmp_path + "/" + random_str + "_out.txt";
        LOG_DEBUG << "Input filename: " << input_filename;
        LOG_DEBUG << "Output filename: " << output_filename;
        std::string label = std::to_string(rop_json["row"].get<int>()) + "_" +
                            std::to_string(rop_json["col"].get<int>()) + "_" +
                            std::to_string(rop_json["slot"].get<int>()) + "_" +
                            std::to_string(rop_json["port"].get<int>());
        LOG_DEBUG << "Label: " << label;
        if (component_map.find(label) == component_map.end()) {
          llvm::outs() << "Error: Cannot find the component : " << label
                       << "\n";
          std::exit(EXIT_FAILURE);
        }
        std::string command = component_path + "/resources/" +
                              component_map[label].get<std::string>() +
                              "/compile_util get_timing_model " +
                              input_filename + " " + output_filename;

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

        // read the output file
        std::ifstream output_file(output_filename);
        if (!output_file.is_open()) {
          llvm::outs() << "Error: Failed to open output file.\n";
          std::exit(EXIT_FAILURE);
        }
        std::string output_str((std::istreambuf_iterator<char>(output_file)),
                               std::istreambuf_iterator<char>());
        output_file.close();

        tm::Operation operation =
            tm::Operation(rop_json["id"].get<std::string>(), output_str);
        operation.col = rop_json["col"].get<int>();
        operation.row = rop_json["row"].get<int>();
        operation.slot = rop_json["slot"].get<int>();
        operation.port = rop_json["port"].get<int>();
        model.add_operation(operation);

        // delete the temporary files
        remove(input_filename.c_str());
        remove(output_filename.c_str());

        op_exprs.push_back(OpExprTuple{
            rop_json["id"].get<std::string>(),
            rop_json["kind"].get<std::string>(), rop_json["row"].get<int>(),
            rop_json["col"].get<int>(), rop_json["slot"].get<int>(),
            rop_json["port"].get<int>(), output_str});

      } else if (auto cop_op = llvm::dyn_cast<CopOp>(&child_op)) {
        llvm::outs() << "CopOp ID: " << cop_op.getId() << "\n";
      } else if (auto raw_op = llvm::dyn_cast<RawOp>(&child_op)) {
        if ((operation_type_set.find("pasm.rop") != operation_type_set.end()) ||
            operation_type_set.find("pasm.cop") != operation_type_set.end()) {
          llvm::outs() << "Error: RawOp cannot be used with RopOp or CopOp.\n";
          std::exit(EXIT_FAILURE);
        }
        return failure();
      } else if (auto cstr_op = llvm::dyn_cast<CstrOp>(&child_op)) {
        std::string type = cstr_op.getType().str();
        std::string expr = cstr_op.getExpr().str();
        model.add_constraint(tm::Constraint(type, expr));
      } else if (auto yield_op = llvm::dyn_cast<YieldOp>(&child_op)) {
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
    std::unordered_map<std::string, std::vector<std::string>>
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
        if (all_control_op_anchors.find(label) ==
            all_control_op_anchors.end()) {
          all_control_op_anchors[label] = std::vector<std::string>();
        }
        std::vector<std::string> anchors =
            model.get_operation(op_expr.id).get_all_anchors();
        all_control_op_anchors[label].insert(
            all_control_op_anchors[label].end(), anchors.begin(),
            anchors.end());
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
        // add a constraint that all ROPs in the same cell cannot be executed
        // at the same time
        for (size_t i = 0; i < ops.size(); i++) {
          int slot = -1;
          int port = -1;

          // Find the current operation in the op_exprs
          for (auto &op : op_exprs) {
            if (op.id == ops[i]) {
              slot = op.slot;
              port = op.port;
              break;
            }
          }

          if (all_control_op_anchors.find(label) !=
              all_control_op_anchors.end()) {
            // add a constraint that the ROPs cannot be executed at the same
            // time as the control operations
            for (auto &anchor : all_control_op_anchors[label]) {
              model.add_constraint(
                  tm::Constraint("linear", ops[i] + " != " + anchor));
            }
          }
        }
      }
    }

    // empty log buffer
    llvm::outs().flush();

    // solve the timing model
    std::unordered_map<std::string, std::string> result = solver.solve(model);
    if (result.empty()) {
      llvm::outs() << "Error: No solution found.\n";
      std::exit(EXIT_FAILURE);
    }

    std::unordered_map<std::string, int> schedule_table;
    for (auto it = result.begin(); it != result.end(); ++it) {
      std::string key = it->first;
      std::string value = it->second;

      // check if the key is use_act_mode_0
      if (key == "use_act_mode_0") {
        // if it is, then set the total_latency to 1
        llvm::outs() << "Using ACT mode 0: " << value << "\n";
        continue;
      }
      // check if the key is use_act_mode_1
      if (key == "use_act_mode_1") {
        // if it is, then set the total_latency to 2
        llvm::outs() << "Using ACT mode 1: " << value << "\n";
        continue;
      }

      // if value is not starting with "[", then it is a number
      if (value[0] != '[') {
        schedule_table[key] = std::stoi(value);
      }
    }

    replace_time_in_instr_param(op, schedule_table, rewriter);
    reshape_instr(op, rewriter);
    synchronize(op, schedule_table, rewriter);

    return success();
  }
};
class ScheduleEpochPass
    : public impl::ScheduleEpochPassBase<ScheduleEpochPass> {
public:
  using impl::ScheduleEpochPassBase<ScheduleEpochPass>::ScheduleEpochPassBase;
  void runOnOperation() override {
    vesyla::pasm::Config cfg;
    // convert json string to json object
    nlohmann::json component_map_json = cfg.get_component_map_json();
    std::string component_path = this->component_path;
    std::string tmp_path = this->tmp_path;
    nlohmann::json arch_json = cfg.get_arch_json();
    int row = arch_json["parameters"]["ROWS"].get<int>();
    int col = arch_json["parameters"]["COLS"].get<int>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ScheduleEpochPassRewriter>(&getContext(), component_map_json,
                                            component_path, tmp_path, row, col);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vesyla::pasm
