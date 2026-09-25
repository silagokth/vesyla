#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_REWRITER_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_REWRITER_HPP__

#include "Matcher.hpp"
#include "PatternLibrary.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace vesyla {
namespace conversion {
namespace select_instructions {

// Turns one match into the instructions its pattern says it becomes.
//
// The replacement is not built op by op in C++ -- it is cloned out of the
// resource's @replace function, with argument i standing for whatever match
// argument i bound to in the program. So what a pattern turns into is stated in
// the component library, next to the shape it matches, and adding a resource
// never touches this file.
//
// Three things the library cannot state, because they belong to the individual
// match rather than to the pattern, are attached here: `id`, copied from the
// matched root; `map`, lifted from the matched access; and `storage`, the
// memref the access lands in, for a resource that holds its own.
// Called immediately after the match that produced `match`, so the bindings name
// values that are live right now. An earlier replacement's rop is an ordinary
// IR user, so when its own root is later replaced, replaceAllUsesWith rewires
// it like anything else -- no separate bookkeeping is needed to keep operands
// current.
mlir::LogicalResult applyReplacement(const Pattern &pattern,
                                     mlir::Operation *root,
                                     const MatchResult &match,
                                     llvm::raw_ostream &diag);

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_REWRITER_HPP__
