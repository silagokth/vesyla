#ifndef __VESYLA_SCHEDULE_BISON_UTIL_HPP__
#define __VESYLA_SCHEDULE_BISON_UTIL_HPP__

#include "vesyla/Parser/GlobalUtil.hpp"
#include "vesyla/Support/Common.hpp"
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
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmTypes.hpp"
#include "vesyla/Support/Anchor.hpp"

namespace vesyla {
namespace schedule {

extern mlir::ModuleOp *module;
extern mlir::Operation *temp_epoch_op;

void print_error(const char *message);

void print_grammar(const std::string &grammar_,
                   const bool printLineNum_ = false);

} // namespace schedule
} // namespace vesyla

struct rop_ref_t {
  ::vesyla::AnchorRange range;
  int offset;
};

struct index_list_t {
  std::vector<int32_t> indices;
};

mlir::Operation *build_cstr(rop_ref_t *lhs, rop_ref_t *rhs,
                            const std::string &cmp);

std::vector<mlir::Operation *> parse_and_build_cstr(const std::string &expr);

#endif // __VESYLA_SCHEDULE_BISON_UTIL_HPP__
