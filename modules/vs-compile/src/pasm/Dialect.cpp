#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include "pasm/Types.hpp"

using namespace mlir;
using namespace vesyla::pasm;

#include "pasm/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void PasmDialect::initialize() {
  registerOps();
  registerTypes();
}
