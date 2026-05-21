/**
 * @file InterconnectPass.hpp
 * @brief MLIR pass that builds a routing-dependency graph per epoch.
 *
 * For each EpochOp the pass walks the contained `pasm.icdep` ops to seed
 * the graph with a pair of nodes per "bulk" icdep (first-use and last-use
 * anchors), plus start/end sentinels. It then walks `pasm.cstr` ops to
 * discover dependencies between those nodes by DFS-ing constraint chains
 * starting from each non-sentinel node.
 *
 * Matching rules along a chain:
 *  - A frame matches a graph node when instr/event/dim agree and the
 *    candidate's anchor is at or above the frame's lower bound. The edge
 *    is bilateral only if every hop so far had min_delay == 0 AND the
 *    candidate sits exactly at the frame's lower bound.
 *  - Constraints with delay `[0,0]` are also traversed in reverse
 *    (dst -> src). A per-traversal visited set breaks the cycles that
 *    this reverse traversal would otherwise create.
 *  - Event-less constraints carry no indices; the destination is
 *    propagated as-is. Constraints whose src is a single element skip the
 *    delta-offset computation. Otherwise delta is applied to as many dst
 *    dimensions as there are delta entries.
 *
 * Nodes left without any incoming/outgoing edge after the DFS are wired
 * to the start/end sentinels. The resulting graph is emitted as
 * `routes_<epoch_id>.dot` (plus a best-effort PNG via graphviz).
 */
#ifndef __VESYLA_PASM_INTERCONNECT_PASS_HPP__
#define __VESYLA_PASM_INTERCONNECT_PASS_HPP__

#include "Passes.hpp"

#endif // __VESYLA_PASM_INTERCONNECT_PASS_HPP__
