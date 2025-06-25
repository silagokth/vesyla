#ifndef __VESYLA_SCHEDULE_BISON_UTIL_HPP__
#define __VESYLA_SCHEDULE_BISON_UTIL_HPP__

#include "GlobalUtil.hpp"
#include "util/Common.hpp"
#include <iostream>
#include <string>

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

namespace vesyla {
namespace schedule {

extern mlir::ModuleOp *module;
extern mlir::Operation *temp_epoch_op;

void print_error(const char *message);

void print_grammar(const std::string &grammar_,
                   const bool printLineNum_ = false);

} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_SCHEDULE_BISON_UTIL_HPP__
