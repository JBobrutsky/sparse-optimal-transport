#ifndef BIPARTITE_SPARSE_DIGRAPH_H
#define BIPARTITE_SPARSE_DIGRAPH_H

#include <cstdint>
#include <algorithm>

namespace lemon {

  // CSR-backed bipartite digraph view conforming to the subset of the LEMON
  // Digraph concept used by NetworkSimplexSimple::init() and its pivot loop.
  //
  // Nodes 0..n1-1 are sources; nodes n1..n1+n2-1 are sinks.
  // Arc id a in [0, k) corresponds to CSR entry a: target = col_idx[a] + n1,
  // source = the unique i such that row_ptr[i] <= a < row_ptr[i+1].
  class BipartiteSparseDigraph {
  public:
    typedef int     Node;
    typedef int64_t Arc;

    BipartiteSparseDigraph(int n1, int n2,
                           const int* row_ptr, const int* col_idx, int64_t k)
      : _n1(n1), _n2(n2), _node_num(n1 + n2),
        _arc_num(k), _row_ptr(row_ptr), _col_idx(col_idx) {}

    int     nodeNum()  const { return _node_num; }
    int64_t arcNum()   const { return _arc_num; }
    int     maxNodeId() const { return _node_num - 1; }
    int64_t maxArcId()  const { return _arc_num - 1; }

    Node operator()(int ix) const { return Node(ix); }
    static int index(const Node& n) { return n; }

    Node source(Arc a) const {
        // Binary search: largest i with row_ptr[i] <= a.
        int lo = 0, hi = _n1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (_row_ptr[mid] <= a) lo = mid; else hi = mid - 1;
        }
        return Node(lo);
    }
    Node target(Arc a) const { return Node(_col_idx[a] + _n1); }

    static int    id(Node n) { return n; }
    static int64_t id(Arc a) { return a; }
    static Node nodeFromId(int i)    { return Node(i); }
    static Arc  arcFromId(int64_t i) { return Arc(i); }

    void first(Node& n) const { n = _node_num - 1; }
    static void next(Node& n) { --n; }

    void first(Arc& a) const { a = _arc_num - 1; }
    static void next(Arc& a) { --a; }

  private:
    int _n1, _n2, _node_num;
    int64_t _arc_num;
    const int* _row_ptr;
    const int* _col_idx;
  };

} // namespace lemon

#endif // BIPARTITE_SPARSE_DIGRAPH_H
