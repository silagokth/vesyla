#ifndef __VESYLA_PASM_INTERCONNECT_BINDING_HPP__
#define __VESYLA_PASM_INTERCONNECT_BINDING_HPP__

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/IR/RoutingDepGraph.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>

namespace vesyla::pasm {

// The fields a config turns into on the wire: sr (0 send, 1 recv, -1 for an
// swb config, which has none) and the source and target the conf instruction
// carries.
struct ConfigKey {
  int sr;
  int source;
  int target;

  bool operator==(const ConfigKey &o) const {
    return sr == o.sr && source == o.source && target == o.target;
  }
};

struct InterconnectConfig {
  ResourceAttr src;
  mlir::ArrayAttr dst;
  std::optional<int> sr;

  // Two configs are the same configuration when the fabric is told the same
  // thing, whatever endpoints they were derived from -- see config_key.
  bool operator==(const InterconnectConfig &o) const;
};

// What `cfg` is emitted as. A send names the direction it leaves by rather
// than the slot it reaches, so two transfers leaving the same slot for the
// same neighbour are one route and not two; comparing the endpoints instead
// splits them, and every config that is split forces a configuration option
// that the fabric does not need.
//
// Emission reads the same key, so the identity used to bind and the
// instruction that comes out of it cannot drift apart.
ConfigKey config_key(const InterconnectConfig &cfg);

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
