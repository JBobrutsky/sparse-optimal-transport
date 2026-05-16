// src/cpp/lemon_solver.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "lemon/core.h"
#include "lemon/tolerance.h"
#include "lemon/list_graph.h"

// Define static members for LEMON Tolerance specializations
// (normally defined in lemon/tolerance.cc, which we don't compile)
namespace lemon {
  float       Tolerance<float>::def_epsilon       = 1e-4f;
  double      Tolerance<double>::def_epsilon      = 1e-10;
  long double Tolerance<long double>::def_epsilon = 1e-14L;
}

namespace py = pybind11;
using namespace lemon;

typedef ListDigraph Graph;
typedef Graph::Node Node;
typedef Graph::Arc  Arc;

// Custom traits: all numeric types are double, so epsilon arithmetic
// stays in double rather than long long. Matches the float64 patch.
template <typename GR>
struct Float64Traits {
    typedef GR     Digraph;
    typedef double Value;      // supply / flow type
    typedef double Cost;       // edge cost type
    typedef double LargeCost;  // internal large-cost type
};

#include "lemon/cost_scaling.h"

typedef CostScaling<Graph, double, double, Float64Traits<Graph>> CS;

std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>>
solve_sparse(
    py::array_t<double,  py::array::c_style | py::array::forcecast> a,
    py::array_t<double,  py::array::c_style | py::array::forcecast> b,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> row_ptr_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> col_idx_arr,
    py::array_t<double,  py::array::c_style | py::array::forcecast> costs_arr,
    int numItermax
) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());

    const double*   ap = a.data();
    const double*   bp = b.data();
    const int32_t*  rp = row_ptr_arr.data();
    const int32_t*  ci = col_idx_arr.data();
    const double*   cp = costs_arr.data();

    Graph g;
    std::vector<Node> nodes;
    nodes.reserve(n + m);
    for (int k = 0; k < n + m; ++k)
        nodes.push_back(g.addNode());

    Graph::NodeMap<double>  supply(g);
    for (int i = 0; i < n; ++i)  supply[nodes[i]]     =  ap[i];
    for (int j = 0; j < m; ++j)  supply[nodes[n + j]] = -bp[j];

    Graph::ArcMap<double>   cost_map(g);
    Graph::ArcMap<int32_t>  arc_src(g), arc_dst(g);

    for (int i = 0; i < n; ++i) {
        for (int32_t ptr = rp[i]; ptr < rp[i + 1]; ++ptr) {
            int j   = ci[ptr];
            Arc arc = g.addArc(nodes[i], nodes[n + j]);
            cost_map[arc] = cp[ptr];
            arc_src[arc]  = i;
            arc_dst[arc]  = j;
        }
    }

    CS cs(g);
    cs.supplyMap(supply);
    cs.costMap(cost_map);
    cs.run(CS::PARTIAL_AUGMENT);

    std::vector<int32_t> out_rows, out_cols;
    std::vector<double>  out_vals;

    for (Graph::ArcIt arc(g); arc != INVALID; ++arc) {
        double flow = cs.flow(arc);
        if (flow > 0.0) {
            out_rows.push_back(arc_src[arc]);
            out_cols.push_back(arc_dst[arc]);
            out_vals.push_back(flow);
        }
    }

    const ssize_t sz = static_cast<ssize_t>(out_rows.size());
    py::array_t<int32_t> rows_out(sz), cols_out(sz);
    py::array_t<double>  vals_out(sz);
    std::copy(out_rows.begin(), out_rows.end(), rows_out.mutable_data());
    std::copy(out_cols.begin(), out_cols.end(), cols_out.mutable_data());
    std::copy(out_vals.begin(), out_vals.end(), vals_out.mutable_data());

    return {rows_out, cols_out, vals_out};
}

PYBIND11_MODULE(_lemon, m) {
    m.doc() = "LEMON CostScaling for sparse balanced OT (float64-patched)";
    m.def(
        "solve_sparse", &solve_sparse,
        py::arg("a"), py::arg("b"),
        py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
        py::arg("numItermax") = 100000,
        "Solve sparse balanced OT. Returns COO (row_indices, col_indices, values)."
    );
}
