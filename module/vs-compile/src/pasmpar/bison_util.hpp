// Copyright (C) 2019 Yu Yang
//
// This file is part of Vesyla.
//
// Vesyla is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Vesyla is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Vesyla.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __VESYLA_PASMPAR_BISON_UTIL_HPP__
#define __VESYLA_PASMPAR_BISON_UTIL_HPP__

#include "global_util.hpp"
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
namespace pasmpar {

extern mlir::ModuleOp *module;
extern mlir::Operation *temp_epoch_op;

void print_error(const char *message);

void print_grammar(const std::string &grammar_,
                   const bool printLineNum_ = false);

} // namespace pasmpar
} // namespace vesyla

#endif // __VESYLA_PASMPAR_BISON_UTIL_HPP__
