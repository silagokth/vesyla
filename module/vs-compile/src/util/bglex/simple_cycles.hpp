// Copyright (c) 2019 herenvarno
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef __SIMPLE_CYCLES_HPP__
#define __SIMPLE_CYCLES_HPP__

#include <boost/graph/subgraph.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

using namespace std;

namespace bglex
{

/**
 * Find the simple cycles in a graph
 */
template <typename G>
vector<vector<typename G::vertex_descriptor>> simple_cycles(G g_)
{

  using namespace boost;
  //typedef typename G::vertex_descriptor Vertex;
  //typedef typename G::edge_descriptor Edge;
  typedef adjacency_list<vecS, vecS, directedS, property<vertex_color_t, int>, property<edge_index_t, int>> SimpleGraph;
  typedef subgraph<SimpleGraph> SubGraph;
  typedef SimpleGraph::vertex_descriptor Vertex;
  typedef SimpleGraph::vertex_iterator VertexIter;
  typedef SimpleGraph::edge_descriptor Edge;
  typedef SimpleGraph::edge_iterator EdgeIter;
  typedef SimpleGraph::out_edge_iterator OutEdgeIter;

  vector<vector<typename G::vertex_descriptor>> cycles;

  //Define a aux function _unblock()
  auto _unblock = [](Vertex thisnode, set<Vertex> &blocked,
                     std::unordered_map<Vertex, set<Vertex>> &B) {
    std::stack<Vertex> s;
    s.push(thisnode);
    while (!s.empty())
    {
      Vertex node = s.top();
      s.pop();
      if (blocked.find(node) != blocked.end())
      {
        blocked.erase(node);
        for (auto &n : B[node])
        {
          s.push(n);
        }
        B[node].clear();
      }
    }
  };

  // Find strong connected components with at least two vertices in it.

  std::unordered_map<typename G::vertex_descriptor, Vertex> g2subg;
  std::unordered_map<Vertex, typename G::vertex_descriptor> subg2g;

  SubGraph subg(num_vertices(g_));
  typename G::vertex_iterator vi, vi_end;
  int i = 0;
  for (tie(vi, vi_end) = vertices(g_); vi != vi_end; vi++, i++)
  {
    g2subg[*vi] = i;
    subg2g[i] = *vi;
  }
  typename G::edge_iterator ei, ei_end;
  for (tie(ei, ei_end) = edges(g_); ei != ei_end; ei++)
  {
    typename G::vertex_descriptor src, dest;
    src = source(*ei, g_);
    dest = target(*ei, g_);
    add_edge(g2subg[src], g2subg[dest], subg);
  }

  vector<int> component(num_vertices(subg));
  int num = strong_components(subg, make_iterator_property_map(component.begin(), get(boost::vertex_index, subg)));

  stack<vector<Vertex>> sccs;
  for (int i = 0; i < num; i++)
  {
    vector<Vertex> temp;
    for (int j = 0; j < component.size(); j++)
    {
      if (component[j] == i)
      {
        temp.push_back(j);
      }
    }
    if (temp.size() > 1)
    {
      sccs.push(temp);
    }
  }

  // Record self-loops and remove them
  vector<Vertex> self_loops;
  VertexIter svi, svi_end;
  for (tie(svi, svi_end) = vertices(subg); svi != svi_end; svi++)
  {
    Vertex vd = *svi;
    Edge ed;
    bool exist = false;
    tie(ed, exist) = edge(vd, vd, subg);
    if (exist)
    {
      self_loops.push_back(vd);
    }
  }
  for (auto vd : self_loops)
  {
    remove_edge(vd, vd, subg);
  }

  // main loop
  while (!sccs.empty())
  {
    auto scc = sccs.top();
    sccs.pop();
    auto sccG = subg.create_subgraph();
    for (auto s : scc)
    {
      add_vertex(s, sccG);
    }

    auto startnode = scc[0];
    vector<Vertex> path = {startnode};
    set<Vertex> blocked;
    set<Vertex> closed;
    blocked.insert(startnode);
    std::unordered_map<Vertex, set<Vertex>> B;
    std::vector<pair<Vertex, vector<Vertex>>> q;

    vector<Vertex> nbrs0;
    OutEdgeIter oei, oei_end;
    for (tie(oei, oei_end) = out_edges(sccG.global_to_local(startnode), sccG); oei != oei_end; oei++)
    {
      Vertex nextnode = sccG.local_to_global(target(*oei, sccG));
      nbrs0.push_back(nextnode);
    }
    q.push_back({startnode, nbrs0});
    while (!q.empty())
    {
      Vertex thisnode = q.back().first;
      vector<Vertex> &nbrs = q.back().second;
      if (!nbrs.empty())
      {
        Vertex nextnode = nbrs.back();
        nbrs.pop_back();
        if (nextnode == startnode)
        {
          cycles.push_back(path);
          for (auto p : path)
          {
            closed.insert(p);
          }
        }
        else if (blocked.find(nextnode) == blocked.end())
        {
          path.push_back(nextnode);
          vector<Vertex> new_nbrs;
          for (tie(oei, oei_end) = out_edges(sccG.global_to_local(nextnode), sccG); oei != oei_end; oei++)
          {
            Vertex xx = sccG.local_to_global(target(*oei, sccG));
            new_nbrs.push_back(xx);
          }
          q.push_back({nextnode, new_nbrs});
          closed.erase(nextnode);
          blocked.insert(nextnode);
          continue;
        }
      }

      if (nbrs.size() == 0)
      {
        if (closed.find(thisnode) != closed.end())
        {
          _unblock(thisnode, blocked, B);
        }
        else
        {
          for (tie(oei, oei_end) = out_edges(sccG.global_to_local(thisnode), sccG); oei != oei_end;
               oei++)
          {
            Vertex nextnode = sccG.local_to_global(target(*oei, sccG));
            if (B[nextnode].find(thisnode) == B[nextnode].end())
            {
              B[nextnode].insert(thisnode);
            }
          }
        }
      }
      q.pop_back();
      path.pop_back();
    }

    auto H = subg.create_subgraph();
    for (auto s : scc)
    {
      add_vertex(s, sccG);
    }
    vector<int> c(num_vertices(H));
    int n = strong_components(H, make_iterator_property_map(c.begin(), get(boost::vertex_index, H)));

    for (int i = 0; i < n; i++)
    {
      vector<Vertex> temp;
      for (int j = 0; j < c.size(); j++)
      {
        if (c[j] == i)
        {
          temp.push_back(j);
        }
      }
      if (temp.size() > 1)
      {
        sccs.push(temp);
      }
    }
  }

  // Final translate
  for (int i = 0; i < cycles.size(); i++)
  {
    for (int j = 0; j < cycles[i].size(); j++)
    {
      cycles[i][j] = subg2g[cycles[i][j]];
    }
  }
  return cycles;
}

} // namespace bglex

#endif
