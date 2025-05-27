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

#include "pasm/CodeGen.hpp"
#include "pasm/Dialect.hpp"
#include "pasm/Passes.hpp"
#include "util/Common.hpp"

#include "pasmpar/Parser.hpp"

namespace vesyla {
namespace schedule {
class Scheduler {
private:
  std::string arch_file;
  std::string isa_file;

public:
  Scheduler(const std::string &arch_file_, const std::string &isa_file_)
      : arch_file(arch_file_), isa_file(isa_file_) {}
  void run(mlir::ModuleOp &module);
};
} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_SCHEDULE_SCHEDULER_HPP__