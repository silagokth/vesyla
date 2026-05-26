#ifndef __VESYLA_PASM_INTERCONNECT_BINDING_HPP__
#define __VESYLA_PASM_INTERCONNECT_BINDING_HPP__

#include "Ops.hpp"
#include "RoutingDepGraph.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>

namespace vesyla::pasm {

struct InterconnectConfig {
  ResourceAttr src;
  mlir::ArrayAttr dst;
  std::optional<int> sr;

  bool operator==(const InterconnectConfig &o) const {
    return src == o.src && dst == o.dst && sr == o.sr;
  }
};

struct InterconnectConfigOption {
  std::vector<InterconnectConfig> configs;
  std::vector<Anchor> first_anchors;
  std::vector<Anchor> last_anchors;
};

struct InterconnectBinding {
  std::array<std::vector<InterconnectConfigOption>, 4> slots;
  std::vector<int> sequence;
};

// 3x3 direction code: row-major in [-1..1] x [-1..1], NW=0..SE=8.
// Sign-clamps any non-zero delta to the nearest neighbor direction.
int direction_code(ResourceAttr from, ResourceAttr to);

bool has_conflict(const Node &candidate, const std::vector<InterconnectConfig> &current,
                  llvm::StringRef kind);

InterconnectBinding bind_interconnect(RoutingDepGraph graph,
                                      llvm::StringRef kind);

void dump_binding(const InterconnectBinding &b, llvm::raw_ostream &os);

RopOp emit_swb_instructions(const InterconnectBinding &binding,
                            CellOp cell, mlir::OpBuilder &builder);

RopOp emit_route_instructions(const InterconnectBinding &binding,
                              CellOp cell, mlir::OpBuilder &builder);

void emit_sequence_instructions(const InterconnectBinding &binding,
                                RopOp rop, mlir::OpBuilder &builder);

void emit_interconnect_constraints(const InterconnectBinding &binding,
                                   RopOp rop, mlir::OpBuilder &builder);

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_INTERCONNECT_BINDING_HPP__
