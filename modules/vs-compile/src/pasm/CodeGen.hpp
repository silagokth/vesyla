#ifndef __VESYLA_PASM_CODEGEN_HPP__
#define __VESYLA_PASM_CODEGEN_HPP__

#include "Dialect.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
#include "Types.hpp"
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

namespace vesyla {
namespace pasm {
class CodeGen {
public:
  CodeGen() = default;
  ~CodeGen() = default;

  // Function to generate code from the given MLIR module
  void generate(mlir::ModuleOp module, const std::string &output_dir,
                const std::string &filename);

private:
  void gen_bin(mlir::ModuleOp module, const std::string &output_dir,
               const std::string &filename);
  void gen_asm(mlir::ModuleOp module, const std::string &output_dir,
               const std::string &filename);
};
} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_CODEGEN_HPP__