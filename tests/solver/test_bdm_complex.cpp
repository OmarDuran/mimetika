// The degree-2 complex hypre's ADS is given for a stabilized_bdm flux block.
//
//     P3 nodal --G--> N2E2 circulation --C--> BDM flux
//
// C, Pi_rt and Pi_nd come from exokal; G is built here and is topological on
// the edges. What has to hold:
//
//   C.G = 0        by Stokes, not by a fit -- the edge rows of G are
//                  integration by parts and the facet rows the in-plane
//                  divergence theorem, so nothing is approximated
//   G phi          for a CUBIC phi, G on phi's nodal dofs reproduces the
//                  circulation dofs of grad phi. Cubic because grad(P3) =
//                  [P2]^3 is exactly the reconstruction space; a lower degree
//                  would pass on a formula that is only first-order right
//   facet-local C  a facet's rows touch its own dofs only, which is what makes
//                  the per-cell blocks assemble into one global matrix
//   simplices      a polytope is refused rather than silently fitted

#include <algorithm>
#include <cmath>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/geometry/quadrature.hpp"
#include "mimetika/linear_solver/bdm_complex.hpp"
#include "mimetika/mesh/structured.hpp"

using graphos::Index;
using mimetika::mesh::box;
using mimetika::mesh::Family;
using mimetika::solver::bdm_complex;
using mimetika::solver::Sparse;
using P = exokal::Mesh::Point;

namespace {

// a fixed cubic and its gradient
struct Cubic {
  std::vector<std::array<int, 3>> a;
  std::vector<double> c;
  Cubic() {
    unsigned s = 12345u;
    for (int i = 0; i <= 3; ++i)
      for (int j = 0; i + j <= 3; ++j)
        for (int k = 0; i + j + k <= 3; ++k) {
          a.push_back({i, j, k});
          s = s * 1664525u + 1013904223u;
          c.push_back(double(s % 2000) / 1000.0 - 1.0);
        }
  }
  double at(const P& x) const {
    double v = 0.0;
    for (std::size_t t = 0; t < a.size(); ++t) {
      double m = c[t];
      for (int d = 0; d < 3; ++d)
        for (int p = 0; p < a[t][(std::size_t)d]; ++p) m *= x[(std::size_t)d];
      v += m;
    }
    return v;
  }
  P grad(const P& x) const {
    P g{0.0, 0.0, 0.0};
    for (std::size_t t = 0; t < a.size(); ++t)
      for (int d = 0; d < 3; ++d) {
        const int e = a[t][(std::size_t)d];
        if (e == 0) continue;
        double m = c[t] * e;
        for (int dd = 0; dd < 3; ++dd) {
          const int p = (dd == d) ? e - 1 : a[t][(std::size_t)dd];
          for (int q = 0; q < p; ++q) m *= x[(std::size_t)dd];
        }
        g[(std::size_t)d] += m;
      }
    return g;
  }
};

std::vector<double> mat_apply(const Sparse& M, const std::vector<double>& x) {
  std::vector<double> y((std::size_t)M.rows, 0.0);
  for (std::size_t k = 0; k < M.value.size(); ++k)
    y[(std::size_t)M.row[k]] += M.value[k] * x[(std::size_t)M.col[k]];
  return y;
}

const double kGp[4] = {0.0694318442029737, 0.3300094782075719, 0.6699905217924281,
                       0.9305681557970263};
const double kGw[4] = {0.1739274225687269, 0.3260725774312731, 0.3260725774312731,
                       0.1739274225687269};

}  // namespace

MIMETIKA_TEST(the_degree_2_complex_composes_to_zero) {
  for (const int n : {1, 2, 3}) {
    const exokal::Mesh mesh = box({n, n, n}, 3, Family::simplex, {1.0, 1.0, 1.0});
    const auto cx = bdm_complex(mesh, 3);

    std::vector<std::vector<std::pair<int, double>>> gcol((std::size_t)cx.n_nodal), crow(
        (std::size_t)cx.n_flux);
    for (std::size_t k = 0; k < cx.grad.value.size(); ++k)
      gcol[(std::size_t)cx.grad.col[k]].push_back({cx.grad.row[k], cx.grad.value[k]});
    for (std::size_t k = 0; k < cx.curl.value.size(); ++k)
      crow[(std::size_t)cx.curl.row[k]].push_back({cx.curl.col[k], cx.curl.value[k]});

    std::vector<double> tmp((std::size_t)cx.n_circ, 0.0);
    double worst = 0.0, ref = 0.0;
    for (int j = 0; j < cx.n_nodal; ++j) {
      for (const auto& [r, v] : gcol[(std::size_t)j]) tmp[(std::size_t)r] = v;
      for (int i = 0; i < cx.n_flux; ++i) {
        double s = 0.0, mag = 0.0;
        for (const auto& [cc, v] : crow[(std::size_t)i]) {
          s += v * tmp[(std::size_t)cc];
          mag += std::abs(v * tmp[(std::size_t)cc]);
        }
        worst = std::max(worst, std::abs(s));
        ref = std::max(ref, mag);
      }
      for (const auto& [r, v] : gcol[(std::size_t)j]) tmp[(std::size_t)r] = 0.0;
    }
    CHECK(worst < 1e-9 * std::max(ref, 1.0));
  }
}

MIMETIKA_TEST(the_topological_gradient_is_the_gradient) {
  for (const auto& lengths : std::vector<std::array<double, 3>>{{1.0, 1.0, 1.0}, {1.0, 0.3, 0.1}}) {
    const exokal::Mesh mesh = box({2, 2, 2}, 3, Family::simplex, lengths);
    const auto cx = bdm_complex(mesh, 3);
    const graphos::Complex& top = mesh.topology();
    const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));
    const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
    const Cubic phi;

    std::vector<double> nodal((std::size_t)cx.n_nodal, 0.0);
    for (Index v = 0; v < nV; ++v) nodal[(std::size_t)v] = phi.at(mesh.point(v));
    for (Index e = 0; e < nE; ++e) {
      const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
      const P pa = mesh.point(e2v.indices[b]), ph = mesh.point(e2v.indices[b + 1]);
      double m0 = 0.0, m1 = 0.0;
      for (int q = 0; q < 4; ++q) {
        const double s = kGp[q];
        const P x{pa[0] + s * (ph[0] - pa[0]), pa[1] + s * (ph[1] - pa[1]),
                  pa[2] + s * (ph[2] - pa[2])};
        const double f = phi.at(x);
        m0 += kGw[q] * f;
        m1 += kGw[q] * f * (2 * s - 1);
      }
      nodal[(std::size_t)(nV + 2 * int(e) + 0)] = m0;
      nodal[(std::size_t)(nV + 2 * int(e) + 1)] = m1;
    }
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
      const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, 3, f, 4);
      double acc = 0.0;
      for (std::size_t p = 0; p < q.weights.size(); ++p) acc += q.weights[p] * phi.at(q.points[p]);
      nodal[(std::size_t)(nV + 2 * nE + int(f))] = acc / area;
    }

    std::vector<double> ref((std::size_t)cx.n_circ, 0.0);
    for (Index e = 0; e < nE; ++e) {
      const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
      const P pa = mesh.point(e2v.indices[b]), ph = mesh.point(e2v.indices[b + 1]);
      const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
      for (int bb = 0; bb < 3; ++bb) {
        double acc = 0.0;
        for (int q = 0; q < 4; ++q) {
          const double s = kGp[q];
          const P x{pa[0] + s * tau[0], pa[1] + s * tau[1], pa[2] + s * tau[2]};
          const P g = phi.grad(x);
          const double chi = bb == 0 ? 1.0 : (bb == 1 ? (2 * s - 1) : (6 * s * s - 6 * s + 1));
          acc += kGw[q] * (g[0] * tau[0] + g[1] * tau[1] + g[2] * tau[2]) * chi;
        }
        ref[(std::size_t)(3 * int(e) + bb)] = acc;
      }
    }
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
      const P nrm{av[0] / area, av[1] / area, av[2] / area};
      const auto t = exokal::mimetic_curl_detail::tangent_frame(nrm);
      const P xf = exokal::centroid(mesh, 2, f);
      const double hf = std::sqrt(area);
      const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, 3, f, 4);
      double a0 = 0, a1 = 0, a2 = 0;
      for (std::size_t p = 0; p < q.weights.size(); ++p) {
        const P x = q.points[p];
        const P g = phi.grad(x);
        const double w = q.weights[p];
        const P r{(x[0] - xf[0]) / hf, (x[1] - xf[1]) / hf, (x[2] - xf[2]) / hf};
        a0 += w * (g[0] * t[0][0] + g[1] * t[0][1] + g[2] * t[0][2]);
        a1 += w * (g[0] * t[1][0] + g[1] * t[1][1] + g[2] * t[1][2]);
        a2 += w * (g[0] * r[0] + g[1] * r[1] + g[2] * r[2]);
      }
      const int r0 = 3 * nE + 3 * int(f);
      ref[(std::size_t)(r0 + 0)] = a0;
      ref[(std::size_t)(r0 + 1)] = a1;
      ref[(std::size_t)(r0 + 2)] = a2;
    }

    const std::vector<double> got = mat_apply(cx.grad, nodal);
    double worst = 0.0, rmax = 0.0;
    for (int i = 0; i < cx.n_circ; ++i) {
      worst = std::max(worst, std::abs(got[(std::size_t)i] - ref[(std::size_t)i]));
      rmax = std::max(rmax, std::abs(ref[(std::size_t)i]));
    }
    CHECK(worst < 1e-10 * std::max(rmax, 1.0));
  }
}

// A facet's curl row touches its own three edges' circulations and its own
// three interior moments -- twelve columns, no more. That is what lets the two
// cells sharing it agree, and so what makes a global C exist.
MIMETIKA_TEST(the_curl_rows_are_facet_local) {
  const exokal::Mesh mesh = box({2, 2, 2}, 3, Family::simplex, {1.0, 1.0, 1.0});
  const auto cx = bdm_complex(mesh, 3);
  std::vector<int> per_row((std::size_t)cx.n_flux, 0);
  for (std::size_t k = 0; k < cx.curl.value.size(); ++k) ++per_row[(std::size_t)cx.curl.row[k]];
  int worst = 0;
  for (const int c : per_row) worst = std::max(worst, c);
  CHECK(worst <= 12);
}

MIMETIKA_TEST(the_partition_is_inherited_entity_by_entity) {
  const exokal::Mesh mesh = box({2, 2, 2}, 3, Family::simplex, {1.0, 1.0, 1.0});
  const auto cx = bdm_complex(mesh, 3);
  const graphos::Complex& top = mesh.topology();
  const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));
  std::vector<std::vector<int>> eo(3);
  eo[0].resize((std::size_t)nV);
  eo[1].resize((std::size_t)nE);
  eo[2].resize((std::size_t)nF);
  for (int v = 0; v < nV; ++v) eo[0][(std::size_t)v] = (mesh.point(v)[0] < 0.5) ? 0 : 1;
  for (int e = 0; e < nE; ++e) eo[1][(std::size_t)e] = e % 2;
  for (int f = 0; f < nF; ++f) eo[2][(std::size_t)f] = (f / 3) % 2;

  const auto own = mimetika::solver::bdm_complex_owners(mesh, eo);
  CHECK(int(own.nodal.size()) == cx.n_nodal);
  CHECK(int(own.circulation.size()) == cx.n_circ);
  CHECK(int(own.flux.size()) == cx.n_flux);
  CHECK(int(own.vnodal.size()) == cx.n_vnodal);
  bool inherit = true;
  for (int e = 0; e < nE; ++e)
    for (int b = 0; b < 3; ++b)
      inherit &= own.circulation[(std::size_t)(3 * e + b)] == eo[1][(std::size_t)e];
  for (int f = 0; f < nF; ++f)
    for (int b = 0; b < 3; ++b) {
      inherit &= own.circulation[(std::size_t)(3 * nE + 3 * f + b)] == eo[2][(std::size_t)f];
      inherit &= own.flux[(std::size_t)(3 * f + b)] == eo[2][(std::size_t)f];
    }
  for (int v = 0; v < nV; ++v)
    for (int c = 0; c < 3; ++c)
      inherit &= own.vnodal[(std::size_t)(3 * v + c)] == eo[0][(std::size_t)v];
  CHECK(inherit);
}

// On a polytope D_edge > m, so the reconstruction is a least-squares fit and a
// facet's rows reach the whole cell -- no global C. Refused, not fitted.
MIMETIKA_TEST(a_polytope_is_refused) {
  const exokal::Mesh hexes = box({2, 2, 2}, 3, Family::cartesian, {1.0, 1.0, 1.0});
  CHECK(!mimetika::solver::all_tetrahedra(hexes, 3));
  CHECK_THROWS(bdm_complex(hexes, 3));
  const exokal::Mesh tets = box({2, 2, 2}, 3, Family::simplex, {1.0, 1.0, 1.0});
  CHECK(mimetika::solver::all_tetrahedra(tets, 3));
}

MIMETIKA_TEST_MAIN()
