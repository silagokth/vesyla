#include "CodeGen.hpp"
#include <fstream>

namespace vesyla {
namespace pasm {
void CodeGen::generate(mlir::ModuleOp module, const std::string &output_dir,
                       const std::string &filename) {
  gen_asm(module, output_dir, filename + ".asm");
  gen_bin(module, output_dir, filename + ".bin");
}

void CodeGen::gen_bin(mlir::ModuleOp module, const std::string &output_dir,
                      const std::string &filename) {
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
void CodeGen::gen_asm(mlir::ModuleOp module, const std::string &output_dir,
                      const std::string &filename) {}
} // namespace pasm
} // namespace vesyla
