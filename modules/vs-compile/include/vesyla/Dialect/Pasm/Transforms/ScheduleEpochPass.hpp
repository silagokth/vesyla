#ifndef __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__
#define __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__

#include "vesyla/Support/Config.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"

#include "vesyla/Analysis/TimingModel/Solver.hpp"
#include "vesyla/Analysis/TimingModel/TimingModel.hpp"
#include "vesyla/Support/Common.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#endif // __VESYLA_PASM_SCHEDULE_EPOCH_PASS_HPP__