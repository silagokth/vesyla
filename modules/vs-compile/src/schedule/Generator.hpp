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
#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include "pasm/Passes.hpp"
#include "pasm/Types.hpp"

#include "pasm/Config.hpp"

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
};
} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_PASM_GENERATOR_HPP__