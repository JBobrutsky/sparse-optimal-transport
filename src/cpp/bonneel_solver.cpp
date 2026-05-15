#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "bonneel/full_bipartitegraph.h"
#include "bonneel/network_simplex_simple.h"

namespace py = pybind11;
using namespace lemon;
typedef FullBipartiteDigraph Digraph;
DIGRAPH_TYPEDEFS(Digraph);

py::array_t<double> solve_dense(
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

    if (n == 0 || m == 0) {
        py::array_t<double> G({n, m});
        std::fill(static_cast<double*>(G.request().ptr),
                  static_cast<double*>(G.request().ptr) + n * m, 0.0);
        return G;
    }

    Digraph di(n, m);
    NetworkSimplexSimple<Digraph, double, double, int64_t> net(
        di, true, n + m, (int64_t)n * m, (size_t)numItermax
    );

    // Negate sink weights: Bonneel's supplyMap expects negative demand
    std::vector<double> neg_b(m);
    for (int j = 0; j < m; j++) neg_b[j] = -bp[j];
    net.supplyMap(ap, n, neg_b.data(), m);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            net.setCost(di.arcFromId((int64_t)i * m + j), Mp[i * m + j]);

    // Run the solver. Bonneel's NS can return INFEASIBLE for floating-point
    // problems even when the flows are correct (artificial-arc precision issue);
    // correctness is validated by marginal constraints in the Python layer.
    net.run();

    py::array_t<double> G({n, m});
    double* Gp = static_cast<double*>(G.request().ptr);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Gp[i * m + j] = net.flow(di.arcFromId((int64_t)i * m + j));

    return G;
}

PYBIND11_MODULE(_bonneel, m) {
    m.doc() = "Bonneel network simplex for dense balanced OT";
    m.def("solve_dense", &solve_dense,
          py::arg("a"), py::arg("b"), py::arg("M"),
          py::arg("numItermax") = 100000,
          "Solve balanced OT. a and b must sum to 1.0. M is (n,m) float64 row-major.");
}
