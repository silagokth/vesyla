#ifndef __VESYLA_TRANSFORMATION_DSE_CONFLICT_GRAPH_HPP__
#define __VESYLA_TRANSFORMATION_DSE_CONFLICT_GRAPH_HPP__

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace vesyla {
namespace transformation {
namespace dse {

// Why two nodes cannot share a resource instance. A bitmask: an edge may have
// been drawn for more than one reason, and keeping all of them is what lets the
// dumped graph answer "why do these two conflict" rather than only "they do".
enum class ConflictReason : unsigned {
  none = 0,
  // One node's result is a direct operand of the other, so the producer still
  // holds its value when the consumer runs.
  data_flow = 1u << 0,
  // Their occupancies overlap: at some point in the program both lay claim to
  // an instance.
  live_overlap = 1u << 1,
  // They hold different storage, and one instance holds one storage.
  distinct_storage = 1u << 2,
};

inline ConflictReason operator|(ConflictReason a, ConflictReason b) {
  return static_cast<ConflictReason>(static_cast<unsigned>(a) |
                                     static_cast<unsigned>(b));
}

inline bool has_reason(ConflictReason set, ConflictReason bit) {
  return (static_cast<unsigned>(set) & static_cast<unsigned>(bit)) != 0;
}

// Undirected conflict graph over the drra.rop operations of a program.
//
// An edge says two nodes cannot be put on the same resource instance, and
// binding is then a colouring of this graph: one colour per instance, adjacent
// nodes coloured differently. This is the input binding needs, and it is what
// says which operations may share.
//
// A NODE IS NOT AN OPERATION. It is what has to be bound as a unit. Every
// access to one register file must land on the same register file, so all the
// operations that share a `storage` become one node -- a conflict graph can say
// two things must differ but never that two things must agree, so what must
// agree is merged before the graph is built rather than constrained inside it.
// Operations that hold no storage, and a resource that computes rather than
// stores, are each a node of their own.
//
// THE MODEL. Two nodes conflict when they are of the same kind -- two of
// different kinds cannot land on one instance whatever the program does -- and
// then either of the following.
//
//   DIFFERENT STORAGE. One instance holds one storage, so two nodes that hold
//   different ones never share, whatever else is true of them. This is the
//   conservative reading: two arrays could in principle sit in one register
//   file if their lifetimes were disjoint or their addresses could not collide,
//   but neither is established here. Lifetimes are not reconstructible from the
//   rops alone -- selection drops the memref, which is why `storage` is lifted
//   in the first place -- and address disjointness is a capacity question
//   against RF_DEPTH that this graph has no way to express.
//
//   OVERLAPPING OCCUPANCY AND OVERLAPPING PARTS. Both must hold.
//
//     Occupancy is where a node holds its instance: each of its operations,
//     always -- one with no results still holds its instance while it runs --
//     plus everywhere a result of one is live, since a produced value sits in
//     the resource that produced it until its last reader has taken it. Two
//     nodes whose stretches do not meet are never both running, so one instance
//     can serve both. When a value is live across an operation that has regions
//     -- an affine.for, say -- the whole body counts as occupied, not just the
//     loop header, which is what makes a value carried across a loop conflict
//     with the operations inside it.
//
//     Parts are what of the resource a node holds. A resource is not
//     indivisible: a dpu has config registers, narrow inputs and outputs, and
//     an AGU per event port; an rf has four access paths, each with its own AGU
//     and data wires. Two operations holding one instance at one moment still
//     do not contend unless they want the same part of it. An operation says
//     which parts it uses in its `uses` attribute -- see RopOp in DrraOps.td --
//     and a node's parts are the union over its operations. The strings are
//     opaque here: two parts are the same part when they are spelled the same,
//     so the resource author owns the vocabulary and this pass needs to know
//     nothing about any particular resource. An operation with no `uses` is
//     taken to hold the whole resource, which is what a resource that has not
//     declared its parts yet falls back to.
//
// WHAT IT DOES NOT KNOW. The occupancies come from data flow, which orders
// operations only as far as their dependencies do. Two independent nodes that
// this graph leaves unjoined may still end up overlapping once exact timing is
// solved further down; if a later pass re-times the program, the pairs it
// brings together have to be added here. So the graph is a lower bound on the
// conflicts a binding must respect, exact for the sequential reading of the
// program and no stronger.
class ConflictGraph {
public:
  // Build the graph for the whole program, and only ever for the whole program
  // -- which is why this takes the module rather than any operation to scope
  // itself to.
  //
  // A register file holds what was put into it until something else is, so
  // which instance a storage is bound to is a question about the program and
  // not about any one epoch. Building an epoch at a time would let the `@rf1`
  // written in one epoch and the `@rf1` read in the next be two nodes, free to
  // be bound to two different register files -- and the read would find an
  // empty one. Two storages must likewise stay on different instances across
  // the whole program, not merely within an epoch.
  //
  // Nothing is lost by widening the scope, because the conflicts that are about
  // time bound themselves: an SSA value never crosses an epoch, so two
  // operations in different epochs never lay claim to an instance at the same
  // point and never draw an occupancy edge. Two epochs that use one resource in
  // turn are still free to share it.
  static ConflictGraph build(mlir::ModuleOp module);

  unsigned size() const { return nodes_.size(); }
  bool empty() const { return nodes_.empty(); }

  // The operations bound as this node, in the order they are written. Always at
  // least one.
  llvm::ArrayRef<mlir::Operation *> ops(unsigned node) const {
    return nodes_[node].ops;
  }

  // The `kind` attribute instruction selection wrote, or null when it wrote
  // none. A node with no kind is joined to nothing: there is no telling what it
  // would contend with. The binder reports it.
  mlir::StringAttr kind(unsigned node) const { return nodes_[node].kind; }

  // The storage this node holds, or null when it holds none.
  mlir::FlatSymbolRefAttr storage(unsigned node) const {
    return nodes_[node].storage;
  }

  // The parts of its resource this node holds -- the union over its operations,
  // in the order first seen. Empty means none of them declared any, which is
  // read as holding all of them.
  llvm::ArrayRef<mlir::StringAttr> parts(unsigned node) const {
    return nodes_[node].parts;
  }

  // The nodes that cannot share an instance with `node`.
  const llvm::SmallDenseSet<unsigned> &neighbors(unsigned node) const {
    return adjacency_[node];
  }

  bool conflicts(unsigned a, unsigned b) const {
    return adjacency_[a].contains(b);
  }

  // Why `a` and `b` conflict; `none` when they do not.
  ConflictReason reason(unsigned a, unsigned b) const;

  // The parts `a` and `b` contend for -- what makes them conflict rather than
  // merely coincide. Empty when they do not conflict, when one of them declared
  // no parts and so was read as holding the whole resource, or when what keeps
  // them apart is their storage rather than any part.
  llvm::ArrayRef<mlir::StringAttr> shared_parts(unsigned a, unsigned b) const;

  // The node `op` was bound as, or nothing when `op` is not a drra.rop of this
  // program.
  std::optional<unsigned> node_of(mlir::Operation *op) const;

  // The graph as Graphviz DOT, one cluster per resource kind, edges labelled
  // with what the two contend for and the reason they were drawn.
  void write_dot(llvm::raw_ostream &os, llvm::StringRef graph_name) const;

private:
  using EdgeKey = std::pair<unsigned, unsigned>;

  struct Node {
    llvm::SmallVector<mlir::Operation *> ops;
    mlir::StringAttr kind;
    mlir::FlatSymbolRefAttr storage;
    llvm::SmallVector<mlir::StringAttr> parts;
  };

  struct Edge {
    ConflictReason why = ConflictReason::none;
    // The parts both endpoints hold. Empty when one of them declared none, or
    // when the edge is there because the storage differs.
    llvm::SmallVector<mlir::StringAttr> shared;
  };

  // Join `a` and `b`, unless they are the same node, cannot contend at all, or
  // want no part of the resource in common.
  void add_edge(unsigned a, unsigned b, ConflictReason why);

  static EdgeKey edge_key(unsigned a, unsigned b) {
    return a < b ? EdgeKey(a, b) : EdgeKey(b, a);
  }

  // Label for the dumped graph: what the node is, and what of its resource it
  // holds.
  std::string node_label(unsigned node) const;

  llvm::SmallVector<Node> nodes_;
  llvm::DenseMap<mlir::Operation *, unsigned> node_index_;
  llvm::SmallVector<llvm::SmallDenseSet<unsigned>> adjacency_;
  llvm::DenseMap<EdgeKey, Edge> edges_;
};

} // namespace dse
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DSE_CONFLICT_GRAPH_HPP__
