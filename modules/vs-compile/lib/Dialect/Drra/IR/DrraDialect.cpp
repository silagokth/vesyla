#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"

using namespace mlir;
using namespace vesyla::drra;

#include "vesyla/Dialect/Drra/IR/DrraDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Drra dialect.
//===----------------------------------------------------------------------===//

void DrraDialect::initialize() { registerOps(); }
