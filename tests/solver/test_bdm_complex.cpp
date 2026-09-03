// The degree-2 complex hypre's ADS is given for a stabilized_bdm flux block.
//
//     P3 nodal --G--> N2E2 circulation --C--> BDM flux
//     [P1]^3 vector nodal --Pi_nd--> circulation ,  --Pi_rt--> flux
//
// NOTHING HERE NEEDS A SIMPLEX. G and C are closed form and read no cell: G is
// integration by parts, C is surface Stokes on a facet against exokal's own
// facet chart. Pi is the only piece that touches a cell, through the modes its
// vertex hats are written in. What has to hold:
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
//   C exactly d^1  C on a field's circulation dofs gives the flux dofs of its
//                  curl, on tetrahedra, hexahedra and prisms alike
//   Pi on P1       Lambda writes the vertex hats in the cell's modes: V^-1 on a
//                  tetrahedron, the pseudoinverse (V^T V)^-1 V^T beyond it.
//                  Either way Lambda V = I, so the hats reproduce every linear
//                  function -- non-interpolatory on a polytope, exact on P1 --
//                  and Pi of a linear field returns that field's own dofs. That
//                  is what ADS coarsens on: the near-nullspace is the constants
//
// dec/mimetic_curl.hpp computes C per CELL from a reconstruction instead. On a
// simplex D_edge = m and the fit is exact; on a polytope D_edge > m, it is
// least squares, it couples the whole cell, and the two cells sharing a facet
// disagree. That is a property of the device, not of d^1 -- which is why C is
// built facet-wise here and the complex is not restricted to tetrahedra.

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

// a quadratic vector field: its curl is linear, which is what the degree-2 rung
// reproduces exactly
struct Quadratic {
  std::vector<std::array<int, 3>> a;
  std::vector<std::array<double, 3>> c;
  Quadratic() {
    unsigned s = 20260902u;
    for (int i = 0; i <= 2; ++i)
      for (int j = 0; i + j <= 2; ++j)
        for (int k = 0; i + j + k <= 2; ++k) {
          a.push_back({i, j, k});
          std::array<double, 3> v{};
          for (int d = 0; d < 3; ++d) {
            s = s * 1664525u + 1013904223u;
            v[(std::size_t)d] = double(s % 2000) / 1000.0 - 1.0;
          }
          c.push_back(v);
        }
  }
  P at(const P& x) const {
    P u{0.0, 0.0, 0.0};
    for (std::size_t t = 0; t < a.size(); ++t) {
      double m = 1.0;
      for (int d = 0; d < 3; ++d)
        for (int q = 0; q < a[t][(std::size_t)d]; ++q) m *= x[(std::size_t)d];
      for (int d = 0; d < 3; ++d) u[(std::size_t)d] += c[t][(std::size_t)d] * m;
    }
    return u;
  }
  double d(std::size_t comp, int k, const P& x) const {
    double acc = 0.0;
    for (std::size_t t = 0; t < a.size(); ++t) {
      const int e = a[t][(std::size_t)k];
      if (e == 0) continue;
      double m = c[t][comp] * e;
      for (int dd = 0; dd < 3; ++dd) {
        const int q = (dd == k) ? e - 1 : a[t][(std::size_t)dd];
        for (int r = 0; r < q; ++r) m *= x[(std::size_t)dd];
      }
      acc += m;
    }
    return acc;
  }
  P curl(const P& x) const {
    return {d(2, 1, x) - d(1, 2, x), d(0, 2, x) - d(2, 0, x), d(1, 0, x) - d(0, 1, x)};
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

// THE DIFFERENTIALS DO NOT NEED A SIMPLEX. C is surface Stokes against the
// facet chart and G is integration by parts, so both are facet- and edge-local
// and the two cells sharing a facet build the same row. Checked two ways at
// once: C G = 0, and C applied to a field's exact circulation dofs against the
// exact stabilized_bdm flux dofs of its curl.
MIMETIKA_TEST(the_differentials_hold_on_a_polytope) {
  const Quadratic u;
  for (const Family fam : {Family::simplex, Family::cartesian, Family::prism}) {
    const exokal::Mesh mesh = box({2, 2, 2}, 3, fam, {1.0, 1.3, 0.7});
    const auto cx = mimetika::solver::bdm_complex_differentials(mesh, 3);
    const graphos::Complex& top = mesh.topology();
    const int nE = int(top.count(1)), nF = int(top.count(2));

    // C G = 0
    std::vector<std::vector<std::pair<int, double>>> grow((std::size_t)cx.grad.rows);
    for (std::size_t k = 0; k < cx.grad.value.size(); ++k)
      grow[(std::size_t)cx.grad.row[k]].push_back({cx.grad.col[k], cx.grad.value[k]});
    std::vector<double> acc((std::size_t)cx.n_nodal, 0.0);
    double cg = 0.0, terms = 0.0;
    for (int r = 0; r < cx.curl.rows; ++r) {
      for (std::size_t k = 0; k < cx.curl.value.size(); ++k) {
        if (cx.curl.row[k] != r) continue;
        for (const auto& [gc, gv] : grow[(std::size_t)cx.curl.col[k]]) {
          acc[(std::size_t)gc] += cx.curl.value[k] * gv;
          terms = std::max(terms, std::abs(cx.curl.value[k] * gv));
        }
      }
      for (double& v : acc) {
        cg = std::max(cg, std::abs(v));
        v = 0.0;
      }
    }
    CHECK(cg < 1e-10 * std::max(terms, 1.0));

    // C d(u) against the flux dofs of curl u
    const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
    std::vector<double> circ((std::size_t)cx.n_circ, 0.0), exact((std::size_t)(3 * nF), 0.0);
    for (Index e = 0; e < nE; ++e) {
      const auto b0 = (std::size_t)e2v.offsets[(std::size_t)e];
      const P pa = mesh.point(e2v.indices[b0]), ph = mesh.point(e2v.indices[b0 + 1]);
      const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
      for (int q = 0; q < 4; ++q) {
        const double sp = kGp[q];
        const P x{pa[0] + sp * tau[0], pa[1] + sp * tau[1], pa[2] + sp * tau[2]};
        const P uu = u.at(x);
        const double ut = uu[0] * tau[0] + uu[1] * tau[1] + uu[2] * tau[2];
        circ[(std::size_t)(3 * int(e) + 0)] += kGw[q] * ut;
        circ[(std::size_t)(3 * int(e) + 1)] += kGw[q] * ut * (2 * sp - 1);
        circ[(std::size_t)(3 * int(e) + 2)] += kGw[q] * ut * (6 * sp * sp - 6 * sp + 1);
      }
    }
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
      const P n{av[0] / area, av[1] / area, av[2] / area};
      const auto t = exokal::mimetic_curl_detail::tangent_frame(n);
      const P xf = exokal::centroid(mesh, 2, f);
      const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, 3, f, 4);
      const auto d3 = [](const P& x, const P& y) {
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
      };
      double m11 = 0.0, m12 = 0.0, m22 = 0.0;
      for (std::size_t pq = 0; pq < q.weights.size(); ++pq) {
        const P r{q.points[pq][0] - xf[0], q.points[pq][1] - xf[1], q.points[pq][2] - xf[2]};
        const double x1 = d3(r, t[0]), x2 = d3(r, t[1]);
        m11 += q.weights[pq] * x1 * x1;
        m12 += q.weights[pq] * x1 * x2;
        m22 += q.weights[pq] * x2 * x2;
      }
      const auto ch = exokal::hodge::facet_chart(area, m11, m12, m22, 3);
      const double c1[3] = {0.0, ch.a11, ch.a21}, c2[3] = {0.0, 0.0, ch.a22};
      for (std::size_t pq = 0; pq < q.weights.size(); ++pq) {
        const P x = q.points[pq];
        const P r{x[0] - xf[0], x[1] - xf[1], x[2] - xf[2]};
        const double cn = d3(u.curl(x), n);
        for (int a2 = 0; a2 < 3; ++a2) {
          const double chi =
              (a2 == 0 ? 1.0 : 0.0) + c1[a2] * d3(r, t[0]) + c2[a2] * d3(r, t[1]);
          exact[(std::size_t)(3 * int(f) + a2)] += q.weights[pq] * cn * chi;
        }
        circ[(std::size_t)(3 * nE + 3 * int(f) + 0)] += q.weights[pq] * d3(u.at(x), t[0]);
        circ[(std::size_t)(3 * nE + 3 * int(f) + 1)] += q.weights[pq] * d3(u.at(x), t[1]);
      }
    }
    const std::vector<double> got = mat_apply(cx.curl, circ);
    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(got[i] - exact[i]));
      scale = std::max(scale, std::abs(exact[i]));
    }
    CHECK(worst < 1e-10 * std::max(scale, 1.0));
  }
}

// Pi_rt and Pi_nd, on any cell shape.
//
// Lambda writes the vertex hats in the cell's modes: V^-1 on a tetrahedron, the
// pseudoinverse (V^T V)^-1 V^T beyond it. Either way Lambda V = I, so the hats
// reproduce every LINEAR function -- non-interpolatory on a polytope, but exact
// on P1. So for a linear vector field v, Pi applied to v's vertex values must
// return v's own dofs, because Pi carries sum_w lambda_w v(x_w) = v.
//
// That is the property ADS needs: the coarse space it hands BoomerAMG is the
// vector nodal one and its near-nullspace is the constants.
MIMETIKA_TEST(the_interpolations_reproduce_a_linear_field) {
  // v(x) = a + B x, a general linear vector field
  const P a{0.3, -0.7, 0.45};
  const double B[9] = {0.6, -0.2, 0.35, 0.15, 0.5, -0.4, -0.25, 0.3, 0.55};
  const auto v = [&](const P& x) {
    P r = a;
    for (int i2 = 0; i2 < 3; ++i2)
      for (int j2 = 0; j2 < 3; ++j2) r[(std::size_t)i2] += B[i2 * 3 + j2] * x[(std::size_t)j2];
    return r;
  };
  const auto d3 = [](const P& x, const P& y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };

  for (const Family fam : {Family::simplex, Family::cartesian, Family::prism}) {
    const exokal::Mesh mesh = box({2, 2, 2}, 3, fam, {1.0, 1.3, 0.7});
    const auto cx = bdm_complex(mesh, 3);
    const graphos::Complex& top = mesh.topology();
    const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));

    // v's vector-nodal coefficients
    std::vector<double> nodal((std::size_t)(3 * nV), 0.0);
    for (Index w = 0; w < nV; ++w) {
      const P vw = v(mesh.point(w));
      for (int cc = 0; cc < 3; ++cc) nodal[(std::size_t)(3 * int(w) + cc)] = vw[(std::size_t)cc];
    }

    // v's own flux and circulation dofs
    std::vector<double> flux((std::size_t)(3 * nF), 0.0), circ((std::size_t)cx.n_circ, 0.0);
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(d3(av, av));
      const P n{av[0] / area, av[1] / area, av[2] / area};
      const auto t = exokal::mimetic_curl_detail::tangent_frame(n);
      const P xf = exokal::centroid(mesh, 2, f);
      const double hf = std::sqrt(area);
      const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, 3, f, 4);
      double m11 = 0.0, m12 = 0.0, m22 = 0.0;
      for (std::size_t pq = 0; pq < q.weights.size(); ++pq) {
        const P r{q.points[pq][0] - xf[0], q.points[pq][1] - xf[1], q.points[pq][2] - xf[2]};
        const double x1 = d3(r, t[0]), x2 = d3(r, t[1]);
        m11 += q.weights[pq] * x1 * x1;
        m12 += q.weights[pq] * x1 * x2;
        m22 += q.weights[pq] * x2 * x2;
      }
      const auto ch = exokal::hodge::facet_chart(area, m11, m12, m22, 3);
      const double c1[3] = {0.0, ch.a11, ch.a21}, c2[3] = {0.0, 0.0, ch.a22};
      for (std::size_t pq = 0; pq < q.weights.size(); ++pq) {
        const P x = q.points[pq];
        const P r{x[0] - xf[0], x[1] - xf[1], x[2] - xf[2]};
        const P vv = v(x);
        const double vn = d3(vv, n), w = q.weights[pq];
        for (int a2 = 0; a2 < 3; ++a2)
          flux[(std::size_t)(3 * int(f) + a2)] +=
              w * vn * ((a2 == 0 ? 1.0 : 0.0) + c1[a2] * d3(r, t[0]) + c2[a2] * d3(r, t[1]));
        circ[(std::size_t)(3 * nE + 3 * int(f) + 0)] += w * d3(vv, t[0]);
        circ[(std::size_t)(3 * nE + 3 * int(f) + 1)] += w * d3(vv, t[1]);
        circ[(std::size_t)(3 * nE + 3 * int(f) + 2)] += w * d3(vv, r) / hf;
      }
    }
    const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
    for (Index e = 0; e < nE; ++e) {
      const auto b0 = (std::size_t)e2v.offsets[(std::size_t)e];
      const P pa = mesh.point(e2v.indices[b0]), ph = mesh.point(e2v.indices[b0 + 1]);
      const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
      for (int q = 0; q < 4; ++q) {
        const double sp = kGp[q];
        const P x{pa[0] + sp * tau[0], pa[1] + sp * tau[1], pa[2] + sp * tau[2]};
        const double vt = d3(v(x), tau);
        circ[(std::size_t)(3 * int(e) + 0)] += kGw[q] * vt;
        circ[(std::size_t)(3 * int(e) + 1)] += kGw[q] * vt * (2 * sp - 1);
        circ[(std::size_t)(3 * int(e) + 2)] += kGw[q] * vt * (6 * sp * sp - 6 * sp + 1);
      }
    }

    const std::vector<double> grt = mat_apply(cx.pi_rt, nodal);
    const std::vector<double> gnd = mat_apply(cx.pi_nd, nodal);
    double wrt = 0.0, srt = 0.0, wnd = 0.0, snd = 0.0;
    for (std::size_t i2 = 0; i2 < grt.size(); ++i2) {
      wrt = std::max(wrt, std::abs(grt[i2] - flux[i2]));
      srt = std::max(srt, std::abs(flux[i2]));
    }
    for (std::size_t i2 = 0; i2 < gnd.size(); ++i2) {
      wnd = std::max(wnd, std::abs(gnd[i2] - circ[i2]));
      snd = std::max(snd, std::abs(circ[i2]));
    }
    CHECK(wrt < 1e-10 * std::max(srt, 1.0));
    CHECK(wnd < 1e-10 * std::max(snd, 1.0));
  }
}

// Reported, not required: the complex builds on any cell shape now.
MIMETIKA_TEST(a_polytope_is_accepted) {
  const exokal::Mesh hexes = box({2, 2, 2}, 3, Family::cartesian, {1.0, 1.0, 1.0});
  CHECK(!mimetika::solver::all_tetrahedra(hexes, 3));
  CHECK(bdm_complex(hexes, 3).curl.rows == 3 * int(hexes.topology().count(2)));
  const exokal::Mesh tets = box({2, 2, 2}, 3, Family::simplex, {1.0, 1.0, 1.0});
  CHECK(mimetika::solver::all_tetrahedra(tets, 3));
}

MIMETIKA_TEST_MAIN()
