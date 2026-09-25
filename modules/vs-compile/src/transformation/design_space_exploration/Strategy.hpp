#ifndef __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__
#define __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__

#include "Architecture.hpp"
#include "ConflictGraph.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace vesyla {
namespace transformation {
namespace dse {

// Binding: which instance of the allocated resources each operation runs on.
//
// `graph` holds the operations of one scope -- one pasm.epoch, or the module
// when the program has no epochs -- and says which of them cannot be put
// together, so binding is a colouring of it: the instances of a kind are the
// colours, and adjacent nodes must be given different ones.
//
// A binder writes its decision as the `resource` attribute of every drra.rop it
// claims, either a single pasm::ResourceAttr or an array laid out [results...,
// operands...], which is what GenerateIcdepPass and DrraToPasmPass read.
class Binder {
public:
  virtual ~Binder() = default;
  virtual mlir::LogicalResult bind(const ConflictGraph &graph,
                                   const Architecture &arch) = 0;
};

// Greedy colouring of the conflict graph, taking the nodes in order of how
// contended they are. It is not optimal -- colouring is not, and a greedy pass
// can need a colour more than the graph forces -- but it never puts two
// conflicting operations on one instance, and when it runs out of instances it
// says which kind ran short rather than overcommitting one.
std::unique_ptr<Binder> create_greedy_coloring_binder();

} // namespace dse
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__
