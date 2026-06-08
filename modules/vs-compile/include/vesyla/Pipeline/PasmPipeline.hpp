#ifndef __VESYLA_SCHEDULE_SCHEDULER_HPP__
#define __VESYLA_SCHEDULE_SCHEDULER_HPP__

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

#include <iostream>

#include "vesyla/Codegen/BinaryGenerator.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Support/Common.hpp"

#include "vesyla/Parser/PasmTextParser.hpp"
#include "vesyla/Support/Config.hpp"

namespace vesyla {
namespace schedule {
class Scheduler {

public:
  void run(mlir::ModuleOp &module, std::string output_dir, bool allow_unsafe);
  void run(std::string pasm_file, std::string output_dir, bool allow_unsafe);

private:
  void save_mlir(mlir::ModuleOp &module, const std::string &filename);
};
} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_SCHEDULE_SCHEDULER_HPP__
