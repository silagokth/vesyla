#ifndef __VESYLA_SCHEDULE_PARSER_HPP__
#define __VESYLA_SCHEDULE_PARSER_HPP__

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
#include <cctype>

#include "vesyla/Parser/GlobalUtil.hpp"
#include "PasmParser.hpp"
#include "vesyla/Support/Common.hpp"

namespace vesyla {
namespace schedule {

class Parser {
public:
  void parse(std::string &input, mlir::ModuleOp *module_);
};

} // namespace schedule
} // namespace vesyla

#endif // __VESYLA_SCHEDULE_PARSER_HPP__
