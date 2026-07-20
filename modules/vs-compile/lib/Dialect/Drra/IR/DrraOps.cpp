#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"

using namespace mlir;
using namespace vesyla::drra;

#define GET_OP_CLASSES
#include "vesyla/Dialect/Drra/IR/DrraOps.cpp.inc"

//===----------------------------------------------------------------------===//

void DrraDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "vesyla/Dialect/Drra/IR/DrraOps.cpp.inc"
      >();
}
