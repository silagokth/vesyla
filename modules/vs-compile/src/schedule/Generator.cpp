#include "Generator.hpp"
#include <fstream>
#include <string>

namespace vesyla {
namespace schedule {

std::string int2bin(int value, int bits) {
  // convert an integer to binary string, padding 0 or 1 depending on the sign
  std::string binary;
  if (value < 0) {
    value = (1 << bits) + value; // convert to unsigned representation
  }
  for (int i = bits - 1; i >= 0; --i) {
    binary += (value & (1 << i)) ? '1' : '0';
  }
  return binary;
}

void Generator::generate(mlir::ModuleOp module, const std::string &output_dir,
                         const std::string &filename) {
  gen_asm(module, output_dir, filename + ".asm");
  gen_bin(module, output_dir, filename + ".bin");
}

void Generator::gen_asm(mlir::ModuleOp module, const std::string &output_dir,
                        const std::string &filename) {
  // open the file for writing
  std::ofstream output_file(output_dir + "/" + filename);
  if (!output_file.is_open()) {
    LOG_FATAL << "Error: Failed to open output file.";
    std::exit(-1);
  }

  mlir::Region &module_region = module.getBodyRegion();
  if (!module_region.empty()) {
    mlir::Block &module_block = module_region.front();
    // Iterate through all operations in the module's block
    for (mlir::Operation &child_op : module_block) {
      if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(&child_op)) {
        mlir::Region &epoch_region = epoch_op.getBody();
        if (!epoch_region.empty()) {
          mlir::Block &epoch_block = epoch_region.front();
          // Iterate through all operations in the epoch's block
          for (mlir::Operation &child_child_op : epoch_block) {
            if (auto raw_op =
                    llvm::dyn_cast<vesyla::pasm::RawOp>(&child_child_op)) {
              int row = raw_op.getRow();
              int col = raw_op.getCol();
              output_file << "cell (row=" << row << ", col=" << col << ")\n";
              mlir::Region &raw_region = raw_op.getBody();
              if (!raw_region.empty()) {
                mlir::Block &raw_block = raw_region.front();
                // Iterate through all operations in the raw's block
                for (mlir::Operation &child_child_child_op : raw_block) {
                  if (auto instr_op = llvm::dyn_cast<vesyla::pasm::InstrOp>(
                          &child_child_child_op)) {
                    std::string type = instr_op.getType().str();
                    output_file << type << "(";

                    mlir::DictionaryAttr current_instr_params =
                        instr_op.getParam();

                    bool is_first = true;
                    for (const mlir::NamedAttribute &named_attr_entry :
                         current_instr_params) {
                      auto attr_name = named_attr_entry.getName();
                      auto attr_value = named_attr_entry.getValue();

                      if (auto int_attr =
                              llvm::dyn_cast<mlir::IntegerAttr>(attr_value)) {
                        if (!is_first) {
                          output_file << ", ";
                        }
                        is_first = false;
                        output_file << attr_name.str() << "="
                                    << int_attr.getInt();
                      } else {
                        llvm::outs()
                            << "Unsupported parameter type in InstrOp: "
                            << attr_value << "\n";
                        std::exit(-1);
                      }
                    }

                    output_file << ")\n";
                  } else if (auto yield_op =
                                 llvm::dyn_cast<vesyla::pasm::YieldOp>(
                                     &child_child_child_op)) {
                    // DO NOTHING
                  } else {
                    llvm::outs() << "Error: Illegal operation type in RawOp: "
                                 << child_child_child_op.getName() << "\n";
                    std::exit(-1);
                  }
                }
              }
            } else if (auto yield_op = llvm::dyn_cast<vesyla::pasm::YieldOp>(
                           &child_child_op)) {
              // DO NOTHING
            } else {
              llvm::outs() << "Error: Illegal operation type in EpochOp: "
                           << child_op.getName() << "\n";
              std::exit(-1);
            }
          }
        }
      } else if (auto yield_op =
                     llvm::dyn_cast<vesyla::pasm::YieldOp>(&child_op)) {
        // DO NOTHING
      } else {
        llvm::outs() << "Error: Illegal operation type in ModuleOp: "
                     << child_op.getName() << "\n";
        std::exit(-1);
      }
    }
  }

  // close the file
  output_file.close();
}
void Generator::gen_bin(mlir::ModuleOp module, const std::string &output_dir,
                        const std::string &filename) {
  vesyla::pasm::Config cfg;
  nlohmann::json component_map_json = cfg.get_component_map_json();
  nlohmann::json isa_json = cfg.get_isa_json();

  int instr_bitwidth = isa_json["format"]["instr_bitwidth"].get<int>();
  int instr_opcode_bitwidth =
      isa_json["format"]["instr_opcode_bitwidth"].get<int>();
  int instr_slot_bitwidth =
      isa_json["format"]["instr_slot_bitwidth"].get<int>();
  int instr_type_bitwidth =
      isa_json["format"]["instr_type_bitwidth"].get<int>();

  // open the file for writing
  std::ofstream output_file(output_dir + "/" + filename);
  if (!output_file.is_open()) {
    llvm::outs() << "Error: Failed to open output file.\n";
    std::exit(-1);
  }

  mlir::Region &module_region = module.getBodyRegion();
  if (!module_region.empty()) {
    mlir::Block &module_block = module_region.front();
    // Iterate through all operations in the module's block
    for (mlir::Operation &child_op : module_block) {
      if (auto epoch_op = llvm::dyn_cast<vesyla::pasm::EpochOp>(&child_op)) {
        mlir::Region &epoch_region = epoch_op.getBody();
        if (!epoch_region.empty()) {
          mlir::Block &epoch_block = epoch_region.front();
          // Iterate through all operations in the epoch's block
          for (mlir::Operation &child_child_op : epoch_block) {
            if (auto raw_op =
                    llvm::dyn_cast<vesyla::pasm::RawOp>(&child_child_op)) {
              int row = raw_op.getRow();
              int col = raw_op.getCol();
              output_file << "cell " << row << " " << col << "\n";
              mlir::Region &raw_region = raw_op.getBody();
              if (!raw_region.empty()) {
                mlir::Block &raw_block = raw_region.front();
                // Iterate through all operations in the raw's block
                for (mlir::Operation &child_child_child_op : raw_block) {
                  if (auto instr_op = llvm::dyn_cast<vesyla::pasm::InstrOp>(
                          &child_child_child_op)) {
                    // get instr name
                    std::string instr_type = instr_op.getType().str();
                    // check if it has the field "slot" and "port"
                    mlir::DictionaryAttr current_instr_params =
                        instr_op.getParam();
                    int slot = -1;
                    int port = -1;
                    for (const mlir::NamedAttribute &named_attr_entry :
                         current_instr_params) {
                      auto attr_name = named_attr_entry.getName();
                      auto attr_value = named_attr_entry.getValue();
                      if (attr_name.str() == "slot") {
                        if (auto int_attr =
                                llvm::dyn_cast<mlir::IntegerAttr>(attr_value)) {
                          slot = int_attr.getInt();
                        } else {
                          llvm::outs()
                              << "Unsupported parameter type in InstrOp: "
                              << attr_value << "\n";
                          std::exit(-1);
                        }
                      } else if (attr_name.str() == "port") {
                        if (auto int_attr =
                                llvm::dyn_cast<mlir::IntegerAttr>(attr_value)) {
                          port = int_attr.getInt();
                        } else {
                          llvm::outs()
                              << "Unsupported parameter type in InstrOp: "
                              << attr_value << "\n";
                          std::exit(-1);
                        }
                      }
                    }

                    std::string label =
                        std::to_string(row) + "_" + std::to_string(col);
                    if (slot != -1 && port != -1) {
                      label += "_" + std::to_string(slot) + "_" +
                               std::to_string(port);
                    }
                    if (component_map_json.find(label) ==
                        component_map_json.end()) {
                      llvm::outs() << "Error: Cannot find the component for "
                                      "label: "
                                   << label << "\n";
                      std::exit(-1);
                    }
                    std::string component_kind =
                        component_map_json[label].get<std::string>();

                    nlohmann::json instr_json;
                    for (auto component : isa_json["components"]) {
                      if (component["kind"] == component_kind) {
                        for (auto instr : component["instructions"]) {
                          if (instr["name"] == instr_type) {
                            instr_json = instr;
                            break;
                          }
                        }
                        break;
                      }
                    }
                    if (instr_json.empty()) {
                      llvm::outs() << "Error: Cannot find the instruction "
                                      "definition for: "
                                   << instr_type << "\n";
                      std::exit(-1);
                    }
                    std::string instr_bin = "";
                    instr_bin += int2bin(instr_json["instr_type"].get<int>(),
                                         instr_type_bitwidth);
                    instr_bin += int2bin(instr_json["opcode"].get<int>(),
                                         instr_opcode_bitwidth);
                    if (slot != -1) {
                      instr_bin += int2bin(slot, instr_slot_bitwidth);
                    }

                    for (const auto &segment : instr_json["segments"]) {
                      std::string segment_name =
                          segment["name"].get<std::string>();
                      int segment_bitwidth = segment["bitwidth"].get<int>();
                      if (current_instr_params.contains(segment_name)) {
                        auto attr_value =
                            current_instr_params.get(segment_name);
                        if (auto int_attr =
                                llvm::dyn_cast<mlir::IntegerAttr>(attr_value)) {
                          instr_bin +=
                              int2bin(int_attr.getInt(), segment_bitwidth);
                        } else {
                          llvm::outs()
                              << "Unsupported parameter type in InstrOp: "
                              << attr_value << "\n";
                          std::exit(-1);
                        }
                      } else {
                        llvm::outs()
                            << "Error: Missing parameter '" << segment_name
                            << "' in InstrOp for instruction: " << instr_type
                            << "\n";
                        std::exit(-1);
                      }
                    }

                    // pad the instruction binary to the required bitwidth
                    if (instr_bin.size() < instr_bitwidth) {
                      instr_bin +=
                          std::string(instr_bitwidth - instr_bin.size(), '0');
                    } else if (instr_bin.size() > instr_bitwidth) {
                      llvm::outs() << "Error: Instruction binary size "
                                   << instr_bin.size()
                                   << " exceeds the required bitwidth: "
                                   << instr_bitwidth << "\n";
                      std::exit(-1);
                    }

                    output_file << instr_bin << "\n";
                  } else if (auto yield_op =
                                 llvm::dyn_cast<vesyla::pasm::YieldOp>(
                                     &child_child_child_op)) {
                    // DO NOTHING
                  } else {
                    llvm::outs() << "Error: Illegal operation type in RawOp: "
                                 << child_child_child_op.getName() << "\n";
                    std::exit(-1);
                  }
                }
              }

            } else if (auto yield_op = llvm::dyn_cast<vesyla::pasm::YieldOp>(
                           &child_child_op)) {
              // DO NOTHING
            } else {
              llvm::outs() << "Error: Illegal operation type in EpochOp: "
                           << child_op.getName() << "\n";
              std::exit(-1);
            }
          }
        }
      } else if (auto yield_op =
                     llvm::dyn_cast<vesyla::pasm::YieldOp>(&child_op)) {
        // DO NOTHING
      } else {
        llvm::outs() << "Error: Illegal operation type in ModuleOp: "
                     << child_op.getName() << "\n";
        std::exit(-1);
      }
    }
  }

  // close the file
  output_file.close();
}

} // namespace schedule
} // namespace vesyla
