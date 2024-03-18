// Copyright (c) 2019 herenvarno
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef __ALL_TOPOLOGICAL_SORTS_HPP__
#define __ALL_TOPOLOGICAL_SORTS_HPP__

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
vector<vector<typename G::vertex_descriptor>> all_topological_sorts(G g_) {

  using namespace boost;
  typedef typename G::vertex_descriptor VD;
  typedef typename G::edge_descriptor ED;
  typedef typename G::vertex_iterator VI;
  typedef typename G::edge_iterator EI;
  typedef typename G::in_edge_iterator IEI;
  typedef typename G::out_edge_iterator OEI;

  typedef boost::adjacency_list<vecS, vecS, bidirectionalS> BG;

  // Define a aux function _ats_util()
  std::function<void(vector<vector<BG::vertex_descriptor>> & ats, BG g,
                     vector<BG::vertex_descriptor> & res,
                     std::unordered_map<VD, bool> visited)>
      _ats_util;
  _ats_util = [&](vector<vector<BG::vertex_descriptor>> &ats, BG g,
                  vector<BG::vertex_descriptor> &res,
                  std::unordered_map<VD, bool> visited) {
    if (ats.size() >= 5) {
      return;
    }
    bool flag = false;

    BG::vertex_iterator vi, vi_end;
    for (tie(vi, vi_end) = vertices(g); vi != vi_end; vi++) {
      if (in_degree(*vi, g) == 0 && !visited[*vi]) {
        res.push_back(*vi);
        visited[*vi] = true;

        BG g0;
        copy_graph(g, g0);
        clear_out_edges(*vi, g0);
        _ats_util(ats, g0, res, visited);

        visited[*vi] = false;
        res.erase(res.end() - 1);
        flag = true;
      }
    }

    if (!flag) {
      ats.push_back(res);
    }
  };

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
  // copy_graph(g_, bg);

  vector<vector<VD>> ret;
  vector<vector<BG::vertex_descriptor>> ret1;
  std::unordered_map<BG::vertex_descriptor, bool> visited;
  BG::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(bg); vi != vi_end; vi++) {
    visited[*vi] = false;
  }
  vector<BG::vertex_descriptor> res;

  _ats_util(ret1, bg, res, visited);

  for (auto &res : ret1) {
    vector<VD> vec;
    for (auto &v : res) {
      vec.push_back(bg2g[v]);
    }
    ret.push_back(vec);
  }

  return ret;
}

} // namespace bglex

#endif
