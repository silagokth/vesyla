#ifndef __VESYLA_PASM_GENERATOR_HPP__
#define __VESYLA_PASM_GENERATOR_HPP__

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmTypes.hpp"

#include "vesyla/Support/Config.hpp"

#include <fstream>
#include <functional>

namespace vesyla {
namespace schedule {
class Generator {
public:
  // Function to generate code from the given MLIR module
  void generate(mlir::ModuleOp module, const std::string &output_dir,
                const std::string &filename);

private:
  void gen_bin(mlir::ModuleOp module, const std::string &output_dir,
               const std::string &filename);
  void gen_asm(mlir::ModuleOp module, const std::string &output_dir,
               const std::string &filename);

  // Walk the module's top-level ops, invoking emit_epoch on each epoch. Loops
  // have already been lowered to epochs (with their control instructions
  // injected) by the ReplaceLoopOp pass, so only epochs reach codegen.
  void emit_program(
      mlir::ModuleOp module,
      const std::function<void(vesyla::pasm::EpochOp)> &emit_epoch);

  // Emit the textual asm for a single epoch's body.
  void emit_epoch_asm(vesyla::pasm::EpochOp epoch_op,
                      std::ofstream &output_file);
  // Emit the binary encoding for a single epoch's body.
  void emit_epoch_bin(vesyla::pasm::EpochOp epoch_op,
                      std::ofstream &output_file,
                      nlohmann::json &component_map_json,
                      nlohmann::json &isa_json, int instr_bitwidth,
                      int instr_opcode_bitwidth, int instr_slot_bitwidth,
                      int instr_type_bitwidth);
};
} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_PASM_GENERATOR_HPP__