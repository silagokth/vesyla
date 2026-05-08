#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmTypes.hpp"

using namespace mlir;
using namespace vesyla::pasm;

#include "vesyla/Dialect/Pasm/IR/PasmDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void PasmDialect::initialize() {
  registerOps();
  registerTypes();
}
