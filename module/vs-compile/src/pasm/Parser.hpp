#ifndef __VESYLA_PASM_PARSER_HPP__
#define __VESYLA_PASM_PARSER_HPP__

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
#include <cctype>

#include <regex>

namespace vesyla {
namespace pasm {

class Parser {
public:
  mlir::ModuleOp *parse(std::string &input);

private:
  void parse_instr(std::string &line);
  void parse_epoch(std::string &line);
  void parse_rop(std::string &line);
  void parse_cop(std::string &line);
  void parse_raw(std::string &line);
  void parse_for(std::string &line);
  void parse_if(std::string &line);
};

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_PARSER_HPP__
