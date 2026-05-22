#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <cstring>
#include "bonneel/full_bipartitegraph.h"
#include "bonneel/bipartite_sparse_digraph.h"
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;
using namespace lemon;

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
solve_dense(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<double, py::array::c_style | py::array::forcecast> M,
    int numItermax
) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    auto M_buf = M.request();
    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const double* ap = static_cast<const double*>(a_buf.ptr);
    const double* bp = static_cast<const double*>(b_buf.ptr);
    const double* Mp = static_cast<const double*>(M_buf.ptr);

    py::array_t<double> G({n, m});
    py::array_t<double> u(n);
    py::array_t<double> v(m);
    double* Gp = static_cast<double*>(G.request().ptr);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (n == 0 || m == 0) {
        std::fill(Gp, Gp + n * m, 0.0);
        std::fill(up, up + n, 0.0);
        std::fill(vp, vp + m, 0.0);
        return std::make_tuple(G, u, v);
    }

    FullBipartiteDigraph di(n, m);
    NetworkSimplexSimple<FullBipartiteDigraph, double, double, int64_t> net(
        di, true, n + m, (int64_t)n * m, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            net.setCost(di.arcFromId((int64_t)i * m + j), Mp[i * m + j]);
    net.run();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Gp[i * m + j] = net.flow(di.arcFromId((int64_t)i * m + j));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(G, u, v);
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    int numItermax
) {
    auto a_buf  = a.request();
    auto b_buf  = b.request();
    auto rp_buf = row_ptr.request();
    auto ci_buf = col_idx.request();
    auto c_buf  = costs.request();

    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const int64_t k = static_cast<int64_t>(c_buf.size);
    const double* ap  = static_cast<const double*>(a_buf.ptr);
    const double* bp  = static_cast<const double*>(b_buf.ptr);
    const int*    rp  = static_cast<const int*>(rp_buf.ptr);
    const int*    ci  = static_cast<const int*>(ci_buf.ptr);
    const double* cp  = static_cast<const double*>(c_buf.ptr);

    py::array_t<double> u(n);
    py::array_t<double> v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int>    rows(0);
        py::array_t<int>    cols(0);
        py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0);
        std::fill(vp, vp + m, 0.0);
        return std::make_tuple(rows, cols, vals, u, v);
    }

    BipartiteSparseDigraph di(n, m, rp, ci, k);
    NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(
        di, true, n + m, k, (size_t)numItermax
    );

    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);

    for (int64_t i = 0; i < k; i++)
        net.setCost(BipartiteSparseDigraph::arcFromId(i), cp[i]);

    net.run();

    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k);
    out_cols.reserve(k);
    out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }

    py::array_t<int>    rows_out(out_rows.size());
    py::array_t<int>    cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(),
                out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(),
                out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(),
                out_vals.size() * sizeof(double));

    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));

    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse_warm_potentials(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    py::array_t<double, py::array::c_style | py::array::forcecast> u0,
    py::array_t<double, py::array::c_style | py::array::forcecast> v0,
    int numItermax
) {
    auto a_buf  = a.request();  auto b_buf  = b.request();
    auto rp_buf = row_ptr.request(); auto ci_buf = col_idx.request();
    auto c_buf  = costs.request();
    auto u0_buf = u0.request(); auto v0_buf = v0.request();

    const int n = static_cast<int>(a_buf.size);
    const int m = static_cast<int>(b_buf.size);
    const int64_t k = static_cast<int64_t>(c_buf.size);
    const double* ap  = static_cast<const double*>(a_buf.ptr);
    const double* bp  = static_cast<const double*>(b_buf.ptr);
    const int*    rp  = static_cast<const int*>(rp_buf.ptr);
    const int*    ci  = static_cast<const int*>(ci_buf.ptr);
    const double* cp  = static_cast<const double*>(c_buf.ptr);
    const double* u0p = static_cast<const double*>(u0_buf.ptr);
    const double* v0p = static_cast<const double*>(v0_buf.ptr);

    py::array_t<double> u(n), v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int> rows(0), cols(0); py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0); std::fill(vp, vp + m, 0.0);
        return std::make_tuple(rows, cols, vals, u, v);
    }

    BipartiteSparseDigraph di(n, m, rp, ci, k);
    NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(
        di, true, n + m, k, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int64_t i = 0; i < k; i++)
        net.setCost(BipartiteSparseDigraph::arcFromId(i), cp[i]);

    net.runWarmPotentials(u0p, v0p, n, m);

    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k); out_cols.reserve(k); out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }
    py::array_t<int>    rows_out(out_rows.size()), cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(), out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(), out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(), out_vals.size() * sizeof(double));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
solve_sparse_warm_basis(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b,
    py::array_t<int,    py::array::c_style | py::array::forcecast> row_ptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> col_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> costs,
    py::array_t<double, py::array::c_style | py::array::forcecast> u0,
    py::array_t<double, py::array::c_style | py::array::forcecast> v0,
    py::array_t<int,    py::array::c_style | py::array::forcecast> arc_ids,
    py::array_t<int,    py::array::c_style | py::array::forcecast> warm_src,
    py::array_t<int,    py::array::c_style | py::array::forcecast> warm_tgt,
    py::array_t<double, py::array::c_style | py::array::forcecast> warm_flow,
    int numItermax
) {
    auto a_buf   = a.request();   auto b_buf   = b.request();
    auto rp_buf  = row_ptr.request(); auto ci_buf = col_idx.request();
    auto c_buf   = costs.request();
    auto u0_buf  = u0.request();  auto v0_buf  = v0.request();
    auto aid_buf = arc_ids.request();
    auto ws_buf  = warm_src.request(); auto wt_buf = warm_tgt.request();
    auto wf_buf  = warm_flow.request();

    const int n      = static_cast<int>(a_buf.size);
    const int m      = static_cast<int>(b_buf.size);
    const int64_t k  = static_cast<int64_t>(c_buf.size);
    const int n_warm = static_cast<int>(aid_buf.size);

    const double* ap   = static_cast<const double*>(a_buf.ptr);
    const double* bp   = static_cast<const double*>(b_buf.ptr);
    const int*    rp   = static_cast<const int*>(rp_buf.ptr);
    const int*    ci   = static_cast<const int*>(ci_buf.ptr);
    const double* cp   = static_cast<const double*>(c_buf.ptr);
    const double* u0p  = static_cast<const double*>(u0_buf.ptr);
    const double* v0p  = static_cast<const double*>(v0_buf.ptr);
    const int*    aidp = static_cast<const int*>(aid_buf.ptr);
    const int*    wsp  = static_cast<const int*>(ws_buf.ptr);
    const int*    wtp  = static_cast<const int*>(wt_buf.ptr);
    const double* wfp  = static_cast<const double*>(wf_buf.ptr);

    py::array_t<double> u(n), v(m);
    double* up = static_cast<double*>(u.request().ptr);
    double* vp = static_cast<double*>(v.request().ptr);

    if (k == 0) {
        py::array_t<int> rows(0), cols(0); py::array_t<double> vals(0);
        std::fill(up, up + n, 0.0); std::fill(vp, vp + m, 0.0);
        return std::make_tuple(rows, cols, vals, u, v);
    }

    BipartiteSparseDigraph di(n, m, rp, ci, k);
    NetworkSimplexSimple<BipartiteSparseDigraph, double, double, int64_t> net(
        di, true, n + m, k, (size_t)numItermax
    );
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);
    for (int64_t i = 0; i < k; i++)
        net.setCost(BipartiteSparseDigraph::arcFromId(i), cp[i]);

    net.runWarmBasis(u0p, v0p, aidp, wsp, wtp, wfp, n_warm, n, m);

    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;
    out_rows.reserve(k); out_cols.reserve(k); out_vals.reserve(k);
    const double eps = 1e-15;
    for (int64_t i = 0; i < k; i++) {
        double f = net.flow(BipartiteSparseDigraph::arcFromId(i));
        if (f > eps) {
            out_rows.push_back(static_cast<int>(di.source(i)));
            out_cols.push_back(ci[i]);
            out_vals.push_back(f);
        }
    }
    py::array_t<int>    rows_out(out_rows.size()), cols_out(out_cols.size());
    py::array_t<double> vals_out(out_vals.size());
    std::memcpy(rows_out.request().ptr, out_rows.data(), out_rows.size() * sizeof(int));
    std::memcpy(cols_out.request().ptr, out_cols.data(), out_cols.size() * sizeof(int));
    std::memcpy(vals_out.request().ptr, out_vals.data(), out_vals.size() * sizeof(double));
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));
    return std::make_tuple(rows_out, cols_out, vals_out, u, v);
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for balanced OT (dense and sparse)";
    m.def("solve_dense",  &solve_dense,
          py::arg("a"), py::arg("b"), py::arg("M"),
          py::arg("numItermax") = 100000,
          "Solve dense balanced OT. Returns (G, u, v).");
    m.def("solve_sparse", &solve_sparse,
          py::arg("a"), py::arg("b"),
          py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
          py::arg("numItermax") = 100000,
          "Solve sparse balanced OT on a CSR support. "
          "Returns (rows, cols, vals, u, v).");
    m.def("solve_sparse_warm_potentials", &solve_sparse_warm_potentials,
          py::arg("a"), py::arg("b"),
          py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
          py::arg("u0"), py::arg("v0"),
          py::arg("numItermax") = 100000,
          "Solve sparse OT with warm dual potentials (Mode C). "
          "Returns (rows, cols, vals, u, v).");
    m.def("solve_sparse_warm_basis", &solve_sparse_warm_basis,
          py::arg("a"), py::arg("b"),
          py::arg("row_ptr"), py::arg("col_idx"), py::arg("costs"),
          py::arg("u0"), py::arg("v0"),
          py::arg("arc_ids"), py::arg("warm_src"), py::arg("warm_tgt"),
          py::arg("warm_flow"),
          py::arg("numItermax") = 100000,
          "Solve sparse OT with full-basis warm start (Mode B). "
          "Returns (rows, cols, vals, u, v).");
}
