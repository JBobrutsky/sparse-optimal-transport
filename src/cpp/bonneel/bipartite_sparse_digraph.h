#ifndef BIPARTITE_SPARSE_DIGRAPH_H
#define BIPARTITE_SPARSE_DIGRAPH_H

#include <cstdint>
#include <algorithm>
#include <vector>

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

    // arc_row_end[a] = row_ptr[source(a) + 1]: upper bound of a's row in CSR.
    // Used by nextOut to detect end-of-row without binary search.
    //
    // csc_col_ptr[j+1] - csc_col_ptr[j]: number of arcs with col_idx == j.
    // csc_arc_id[p]: arc id at CSC position p (sorted by column).
    // Used by firstIn/nextIn to iterate arcs incoming to a sink node.

    BipartiteSparseDigraph(int n1, int n2,
                           const int* row_ptr, const int* col_idx, int64_t k)
      : _n1(n1), _n2(n2), _node_num(n1 + n2),
        _arc_num(k), _row_ptr(row_ptr), _col_idx(col_idx)
    {
      // Build arc_row_end: for each arc a, store row_ptr[source(a)+1].
      _arc_row_end.resize(k, 0);
      for (int i = 0; i < n1; ++i) {
        int64_t end = static_cast<int64_t>(row_ptr[i + 1]);
        for (int64_t a = row_ptr[i]; a < end; ++a)
          _arc_row_end[a] = end;
      }

      // Build CSC for firstIn/nextIn.
      // csc_col_ptr has size n2+1; csc_arc_id has size k.
      _csc_col_ptr.assign(n2 + 1, 0);
      for (int64_t a = 0; a < k; ++a)
        ++_csc_col_ptr[col_idx[a] + 1];
      for (int j = 0; j < n2; ++j)
        _csc_col_ptr[j + 1] += _csc_col_ptr[j];
      _csc_arc_id.resize(k);
      std::vector<int64_t> tmp(_csc_col_ptr.begin(), _csc_col_ptr.end());
      for (int64_t a = 0; a < k; ++a)
        _csc_arc_id[tmp[col_idx[a]]++] = a;

      // csc_arc_pos[a]: position in csc_arc_id[] of arc a (for nextIn).
      _csc_arc_pos.resize(k);
      for (int64_t p = 0; p < k; ++p)
        _csc_arc_pos[_csc_arc_id[p]] = p;
    }

    int     nodeNum()    const { return _node_num; }
    int64_t arcNum()     const { return _arc_num; }
    int     maxNodeId()  const { return _node_num - 1; }
    int64_t maxArcId()   const { return _arc_num - 1; }

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

    static int     id(Node n) { return n; }
    static int64_t id(Arc a)  { return a; }
    static Node nodeFromId(int i)     { return Node(i); }
    static Arc  arcFromId(int64_t i)  { return Arc(i); }

    // --- Node / Arc iteration (full graph, used by init()) ---
    void first(Node& n) const { n = _node_num - 1; }
    static void next(Node& n) { --n; }

    void first(Arc& a) const { a = _arc_num - 1; }
    static void next(Arc& a) { --a; }

    // --- Out-arc iteration (source node u, arcs row_ptr[u]..row_ptr[u+1]-1) ---
    void firstOut(Arc& a, const Node& u) const {
        if (u >= _n1) { a = -1; return; }   // sink nodes have no out-arcs
        int64_t s = _row_ptr[u];
        int64_t e = _row_ptr[u + 1];
        a = (s < e) ? s : -1;
    }
    void nextOut(Arc& a) const {
        int64_t nxt = a + 1;
        a = (nxt < _arc_row_end[a]) ? nxt : -1;
    }

    // --- In-arc iteration (sink node v = col + n1) via CSC ---
    void firstIn(Arc& a, const Node& v) const {
        if (v < _n1) { a = -1; return; }   // source nodes have no in-arcs
        int j = v - _n1;
        int64_t s = _csc_col_ptr[j];
        int64_t e = _csc_col_ptr[j + 1];
        a = (s < e) ? _csc_arc_id[s] : -1;
    }
    void nextIn(Arc& a) const {
        // Find the CSC position of a, advance by 1, check same column.
        int64_t p    = _csc_arc_pos[a];
        int     col  = _col_idx[a];
        int64_t nxt  = p + 1;
        a = (nxt < _csc_col_ptr[col + 1]) ? _csc_arc_id[nxt] : -1;
    }

  private:
    int _n1, _n2, _node_num;
    int64_t _arc_num;
    const int*    _row_ptr;
    const int*    _col_idx;

    std::vector<int64_t> _arc_row_end;   // [k]: CSR row end for each arc
    std::vector<int64_t> _csc_col_ptr;   // [n2+1]: CSC column pointers
    std::vector<int64_t> _csc_arc_id;    // [k]: arc ids in column order
    std::vector<int64_t> _csc_arc_pos;   // [k]: CSC position of each arc id
  };

} // namespace lemon

#endif // BIPARTITE_SPARSE_DIGRAPH_H
