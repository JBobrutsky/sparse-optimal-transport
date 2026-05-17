// src/cpp/lemon_solver.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

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

// CostScaling with int64 supply/flow and double costs.
// CostScaling requires integer supply/demand for correct termination.
// Float64 costs are supported natively (the float64 patch only affects
// the epsilon termination criterion, not supply types).
#include "lemon/cost_scaling.h"

typedef CostScaling<Graph, int64_t, double> CS;

// Scale factor for converting float64 supply values to int64.
// 1e12 gives ~12 decimal digits of precision for supply values in [0,1],
// comfortably below the 1e-9 marginal tolerance used in tests.
static const int64_t SUPPLY_SCALE = 1000000000000LL;

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

    // Convert float64 supply/demand to int64 by scaling.
    // Round each value and balance the total to ensure sum(a_int) == sum(b_int).
    std::vector<int64_t> a_int(n), b_int(m);
    int64_t sum_a = 0, sum_b = 0;
    for (int i = 0; i < n; ++i) {
        a_int[i] = static_cast<int64_t>(std::llround(ap[i] * SUPPLY_SCALE));
        sum_a += a_int[i];
    }
    for (int j = 0; j < m; ++j) {
        b_int[j] = static_cast<int64_t>(std::llround(bp[j] * SUPPLY_SCALE));
        sum_b += b_int[j];
    }
    // Absorb any rounding residual into the last sink to guarantee balance.
    if (m > 0) b_int[m - 1] += (sum_a - sum_b);

    Graph g;
    std::vector<Node> nodes;
    nodes.reserve(n + m);
    for (int k = 0; k < n + m; ++k)
        nodes.push_back(g.addNode());

    Graph::NodeMap<int64_t>  supply(g);
    for (int i = 0; i < n; ++i)  supply[nodes[i]]     =  a_int[i];
    for (int j = 0; j < m; ++j)  supply[nodes[n + j]] = -b_int[j];

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
    auto pt = cs.run(CS::PARTIAL_AUGMENT);
    if (pt != CS::OPTIMAL) {
        // The wrapper used to silently return all-zero flows on infeasibility,
        // which masked malformed problems. Raise so the caller hears about it.
        const char* what = (pt == CS::INFEASIBLE) ? "INFEASIBLE"
                         : (pt == CS::UNBOUNDED) ? "UNBOUNDED" : "UNKNOWN";
        throw std::runtime_error(
            std::string("LEMON CostScaling did not reach OPTIMAL (status=") + what + ")");
    }

    // Scale flow values back to float64 by dividing by SUPPLY_SCALE.
    const double inv_scale = 1.0 / static_cast<double>(SUPPLY_SCALE);

    std::vector<int32_t> out_rows, out_cols;
    std::vector<double>  out_vals;

    for (Graph::ArcIt arc(g); arc != INVALID; ++arc) {
        int64_t flow_int = cs.flow(arc);
        if (flow_int > 0) {
            out_rows.push_back(arc_src[arc]);
            out_cols.push_back(arc_dst[arc]);
            out_vals.push_back(static_cast<double>(flow_int) * inv_scale);
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
