#ifndef __VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace assign_ids {
#define GEN_PASS_DECL
#include "transformation/assign_ids/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/assign_ids/Passes.hpp.inc"

} // namespace assign_ids
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_ASSIGN_IDS_PASSES_HPP__
