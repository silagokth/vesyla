// Copyright (c) 2019 herenvarno
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef __BGLEX_RANDOM_TOPOLOGICAL_SORT_HPP__
#define __BGLEX_RANDOM_TOPOLOGICAL_SORT_HPP__

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/property_map/property_map.hpp>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

using namespace std;

namespace bglex {

/**
 * Find all topolotical sorts in a graph
 */
template <typename G>
vector<typename G::vertex_descriptor>
random_topological_sort(G g_, bool &is_unique_) {

  using namespace boost;
  typedef typename G::vertex_descriptor VD;
  typedef typename G::edge_descriptor ED;
  typedef typename G::vertex_iterator VI;
  typedef typename G::edge_iterator EI;
  typedef typename G::in_edge_iterator IEI;
  typedef typename G::out_edge_iterator OEI;

  typedef boost::adjacency_list<vecS, vecS, bidirectionalS> BG;

  BG bg;
  std::unordered_map<typename G::vertex_descriptor, BG::vertex_descriptor> g2bg;
  std::unordered_map<BG::vertex_descriptor, typename G::vertex_descriptor> bg2g;
  VI vvi, vvi_end;
  for (tie(vvi, vvi_end) = vertices(g_); vvi != vvi_end; vvi++) {
    BG::vertex_descriptor vd = add_vertex(bg);
    g2bg[*vvi] = vd;
    bg2g[vd] = *vvi;
  }
  EI eei, eei_end;
  for (tie(eei, eei_end) = edges(g_); eei != eei_end; eei++) {
    add_edge(g2bg[source(*eei, g_)], g2bg[target(*eei, g_)], bg);
  }

  std::unordered_map<BG::vertex_descriptor, bool> visited;
  vector<BG::vertex_descriptor> all_vertices;
  BG::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(bg); vi != vi_end; vi++) {
    visited[*vi] = false;
    all_vertices.push_back(*vi);
  }
  vector<BG::vertex_descriptor> res;
  is_unique_ = true;
  std::random_shuffle(all_vertices.begin(), all_vertices.end());

  while (res.size() < num_vertices(bg)) {
    int counter = 0;
    if (is_unique_ == true) {
      for (auto &vd : all_vertices) {
        if (in_degree(vd, bg) == 0 && !visited[vd]) {
          counter++;
          if (counter >= 2) {
            is_unique_ = false;
            break;
          }
        }
      }
    }
    for (auto &vd : all_vertices) {
      if (in_degree(vd, bg) == 0 && !visited[vd]) {
        res.push_back(vd);
        visited[vd] = true;
        clear_out_edges(vd, bg);
        break;
      }
    }
  }

  return res;
}

} // namespace bglex

#endif // __BGLEX_RANDOM_TOPOLOGICAL_SORT_HPP__
