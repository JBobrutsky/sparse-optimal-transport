#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include "bonneel/full_bipartitegraph.h"
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;
using namespace lemon;
typedef FullBipartiteDigraph Digraph;
DIGRAPH_TYPEDEFS(Digraph);

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

    Digraph di(n, m);
    NetworkSimplexSimple<Digraph, double, double, int64_t> net(
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

    // POT convention u + v <= M, so flip pi sign on sources.
    for (int i = 0; i < n; i++) up[i] = -net.potential(di(i));
    for (int j = 0; j < m; j++) vp[j] =  net.potential(di(n + j));

    return std::make_tuple(G, u, v);
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for balanced OT";
    m.def("solve_dense", &solve_dense,
          py::arg("a"), py::arg("b"), py::arg("M"),
          py::arg("numItermax") = 100000,
          "Solve balanced OT. Returns (G, u, v).");
}
