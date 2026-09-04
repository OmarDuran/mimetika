#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "exokal/dec/mimetic_curl.hpp"
#include "exokal/geometry/embedding.hpp"
#include "exokal/geometry/quadrature.hpp"
#include "exokal/hodge/mimetic_operators/facet_orthonormalization.hpp"
#include "exokal/hodge/mimetic_operators/stabilized/flux_bdm.hpp"
#include "graphos/core/incidence.hpp"

// The four matrices hypre's ADS needs to precondition a stabilized_bdm flux
// block on a TETRAHEDRAL mesh:
//
//     P3 nodal --G--> N2E2 circulation --C--> BDM flux
//     [P1]^3 vector nodal --Pi_nd--> circulation ,  --Pi_rt--> flux
//
// C, Pi_nd and Pi_rt come from exokal (dec/mimetic_curl.hpp kind=full, and
// hodge/.../flux_bdm.hpp). G does not exist there and is built here.
//
// G IS TOPOLOGICAL ON THE EDGES. A circulation dof is int_0^1 u.tau chi_b ds
// with tau = p_h - p_a, so for u = grad phi the integrand is d/ds phi(x(s)) and
// integration by parts removes the metric entirely:
//
//     b=0   phi(p_h) - phi(p_a)                 signed vertex-edge incidence
//     b=1   phi(p_h) + phi(p_a) - 2 m_e^0
//     b=2   phi(p_h) - phi(p_a) - 6 m_e^1
//
// with m_e^0 = int_0^1 phi ds and m_e^1 = int_0^1 phi (2s-1) ds the two nodal
// edge moments. The coefficients are numbers, not geometry.
//
// The facet-interior rows carry geometry but are still exact and local: the
// in-plane divergence theorem turns them into edge means of phi plus the facet
// mean, with nu_e the in-facet outward normal of edge e and d_e = (x-x_f).nu_e
// (constant along a straight edge),
//
//     int_f grad phi . t_i   =  sum_e (t_i.nu_e) |e| m_e^0
//     int_f grad phi . r/h_f = ( sum_e d_e |e| m_e^0 - 2 |f| m_f ) / h_f
//
// so C.G = 0 holds by Stokes rather than to the accuracy of a fit.
//
// TWO PER-CELL SCALINGS ARE REMOVED so the per-cell blocks assemble.
// dec/mimetic_curl.hpp nondimensionalizes by h_E = cbrt|E|:
//   columns  the facet-radial circulation dof carries 1/h_E; rescaled to the
//            facet-intrinsic h_f = sqrt|f| by the factor h_f/h_E
//   rows     the flux moments go to exokal's stabilized_bdm dofs through
//            T_f = N pinv(Pi_rt), which is exact (the two span the same
//            functionals) and absorbs h_E because that basis is facet-intrinsic
//
// A facet's curl rows depend only on that facet's own dofs -- Stokes again --
// so contributions from the two incident cells AGREE rather than add, and are
// averaged. The same holds for Pi, whose columns are global continuous P1 hats.

namespace mimetika::solver {

using graphos::Index;

struct Sparse {
  int rows{0}, cols{0};
  std::vector<int> row, col;
  std::vector<double> value;
  void push(int r, int c, double v) {
    if (v == 0.0) return;
    row.push_back(r);
    col.push_back(c);
    value.push_back(v);
  }
};

// The degree-2 complex of one tetrahedral mesh, in global dof numbering:
//   nodal        vertex v -> v ; edge e moment b<2 -> nV + 2e + b ; facet f -> nV + 2nE + f
//   circulation  edge e moment b<3 -> 3e + b ; facet f moment b<3 -> 3nE + 3f + b
//   flux         facet f moment b<3 -> 3f + b   (exokal's stabilized_bdm dof)
//   vector nodal vertex v component c -> 3v + c
struct BdmComplex {
  Sparse curl, grad, pi_rt, pi_nd;
  int n_flux{0}, n_circ{0}, n_nodal{0}, n_vnodal{0};
};

namespace bdm_detail {

using P = exokal::Mesh::Point;

// rows AGREE between cells rather than add: accumulate and divide by the count.
//
// Triples then one sort, not an ordered map: assembly was the dominant cost of
// the whole preconditioner -- 10 s against 1.4 s of setup and solve at 65k flux
// unknowns -- and most of it was the map's per-entry allocation and pointer
// chasing, which also made it superlinear.
struct Accum {
  // ONE PACKED KEY, NOT A PERMUTATION. Sorting an index array with a
  // comparator that dereferences two parallel vectors costs a cache miss per
  // comparison; (row << 32) | col is a scalar, so the sort is contiguous and
  // the compare is one instruction. Once exokal hoisted its adjacencies out of
  // the per-cell curl this was the largest single term in assembly.
  std::vector<std::uint64_t> key;
  std::vector<double> x;
  std::vector<int> hits;
  explicit Accum(int rows) : hits((std::size_t)rows, 0) {}
  void reserve(std::size_t n) {
    key.reserve(n);
    x.reserve(n);
  }
  void add(int rr, int cc, double v) {
    key.push_back((std::uint64_t)(std::uint32_t)rr << 32 | (std::uint32_t)cc);
    x.push_back(v);
  }
  void touched(int rr) { ++hits[(std::size_t)rr]; }

  // DROP RELATIVE TO THE ROW, NOT ABSOLUTELY.
  //
  // A facet's curl row is facet-local -- Stokes -- so it has at most 12 entries:
  // its three edges' three circulations, and its own three interior moments.
  // Everything else is the reconstruction's round-off, around 1e-12 of the row
  // but well above any fixed floor, and keeping it tripled the row count and
  // with it the density of C^T A C, which is where ADS actually spends: the
  // per-iteration cost fell 12-fold when it went.
  Sparse finish(int rows, int cols, double rel = 1e-9) const {
    std::vector<std::pair<std::uint64_t, double>> e(key.size());
    for (std::size_t i = 0; i < key.size(); ++i) e[i] = {key[i], x[i]};
    std::sort(e.begin(), e.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    Sparse s;
    s.rows = rows;
    s.cols = cols;
    // sum duplicates in place, then drop against each row's own peak -- two
    // passes over one contiguous array rather than a map walk
    std::size_t m = 0;
    std::vector<double> peak((std::size_t)rows, 0.0);
    for (std::size_t i = 0; i < e.size();) {
      std::size_t j = i;
      double acc = 0.0;
      while (j < e.size() && e[j].first == e[i].first) acc += e[j++].second;
      e[m] = {e[i].first, acc};
      const auto r = (std::size_t)(e[i].first >> 32);
      peak[r] = std::max(peak[r], std::fabs(acc));
      ++m;
      i = j;
    }
    s.row.reserve(m);
    s.col.reserve(m);
    s.value.reserve(m);
    for (std::size_t k = 0; k < m; ++k) {
      const auto r = (int)(e[k].first >> 32);
      if (std::fabs(e[k].second) <= rel * peak[(std::size_t)r]) continue;
      const int n = hits[(std::size_t)r] == 0 ? 1 : hits[(std::size_t)r];
      s.push(r, (int)(std::uint32_t)e[k].first, e[k].second / double(n));
    }
    return s;
  }
};

// solve A X = B for X, A n x n, B n x m, plain Gaussian elimination with
// partial pivoting -- n is 3 or 4 here
inline bool solve_dense(std::vector<double> A, std::vector<double>& B, int n, int m) {
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i)
      if (std::fabs(A[(std::size_t)(i * n + k)]) > std::fabs(A[(std::size_t)(p * n + k)])) p = i;
    if (std::fabs(A[(std::size_t)(p * n + k)]) < 1e-300) return false;
    if (p != k) {
      for (int j = 0; j < n; ++j) std::swap(A[(std::size_t)(k * n + j)], A[(std::size_t)(p * n + j)]);
      for (int j = 0; j < m; ++j) std::swap(B[(std::size_t)(k * m + j)], B[(std::size_t)(p * m + j)]);
    }
    const double d = A[(std::size_t)(k * n + k)];
    for (int j = 0; j < n; ++j) A[(std::size_t)(k * n + j)] /= d;
    for (int j = 0; j < m; ++j) B[(std::size_t)(k * m + j)] /= d;
    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      const double f = A[(std::size_t)(i * n + k)];
      if (f == 0.0) continue;
      for (int j = 0; j < n; ++j) A[(std::size_t)(i * n + j)] -= f * A[(std::size_t)(k * n + j)];
      for (int j = 0; j < m; ++j) B[(std::size_t)(i * m + j)] -= f * B[(std::size_t)(k * m + j)];
    }
  }
  return true;
}

}  // namespace bdm_detail

// The P3 nodal dofs' positions: a vertex, an edge midpoint twice, a facet
// centroid. ADS keeps the coordinates even when the interpolations supersede
// them, so they are stated rather than left as zeros.
inline std::vector<double> bdm_nodal_coordinates(const exokal::Mesh& mesh) {
  const graphos::Complex& top = mesh.topology();
  const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));
  const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
  std::vector<double> xyz((std::size_t)(nV + 2 * nE + nF) * 3, 0.0);
  for (Index v = 0; v < nV; ++v) {
    const auto& p = mesh.point(v);
    for (int d = 0; d < 3; ++d) xyz[(std::size_t)v * 3 + (std::size_t)d] = p[(std::size_t)d];
  }
  for (Index e = 0; e < nE; ++e) {
    const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
    const auto pa = mesh.point(e2v.indices[b]), ph = mesh.point(e2v.indices[b + 1]);
    for (int m = 0; m < 2; ++m)
      for (int d = 0; d < 3; ++d)
        xyz[(std::size_t)(nV + 2 * int(e) + m) * 3 + (std::size_t)d] =
            0.5 * (pa[(std::size_t)d] + ph[(std::size_t)d]);
  }
  for (Index f = 0; f < nF; ++f) {
    const auto c = exokal::centroid(mesh, 2, f);
    for (int d = 0; d < 3; ++d)
      xyz[(std::size_t)(nV + 2 * nE + int(f)) * 3 + (std::size_t)d] = c[(std::size_t)d];
  }
  return xyz;
}

// Is every cell a tetrahedron? Reported, not required: the complex is built for
// any cell shape. On a tetrahedron the vertex hats of Pi are interpolatory and
// the degree-2 reconstruction is exact; beyond it both are least squares.
inline bool all_tetrahedra(const exokal::Mesh& mesh, int cell_dim) {
  if (cell_dim != 3) return false;
  const graphos::Adjacency c2v = graphos::incidence(mesh.topology(), 3, 0);
  for (Index c = 0; c < mesh.count(3); ++c) {
    if (c2v.offsets[(std::size_t)c + 1] - c2v.offsets[(std::size_t)c] != 4) return false;
  }
  return true;
}

// Who owns each unknown of the four spaces, from the mesh entity owners.
//
// hypre renumbers every space by (owner, index), so all it needs is one owner
// per unknown. Each of the complex's unknowns sits on exactly one mesh entity
// -- a circulation moment on its edge or its facet, a nodal moment on its
// vertex, edge or facet, a flux moment on its facet, a vector nodal component
// on its vertex -- so the partition is inherited entity by entity and no two
// ranks can disagree about a shared one.
struct BdmOwners {
  std::vector<int> nodal, circulation, flux, vnodal;
};

inline BdmOwners bdm_complex_owners(const exokal::Mesh& mesh,
                                    const std::vector<std::vector<int>>& entity_owner) {
  if (entity_owner.size() < 3) {
    throw std::invalid_argument("bdm_complex_owners: need vertex, edge and facet owners");
  }
  const graphos::Complex& top = mesh.topology();
  const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));
  const auto& ov = entity_owner[0];
  const auto& oe = entity_owner[1];
  const auto& of = entity_owner[2];
  if (int(ov.size()) != nV || int(oe.size()) != nE || int(of.size()) != nF) {
    throw std::invalid_argument("bdm_complex_owners: the owners do not cover the entities");
  }
  BdmOwners o;
  o.nodal.resize((std::size_t)(nV + 2 * nE + nF));
  for (int v = 0; v < nV; ++v) o.nodal[(std::size_t)v] = ov[(std::size_t)v];
  for (int e = 0; e < nE; ++e)
    for (int b = 0; b < 2; ++b) o.nodal[(std::size_t)(nV + 2 * e + b)] = oe[(std::size_t)e];
  for (int f = 0; f < nF; ++f) o.nodal[(std::size_t)(nV + 2 * nE + f)] = of[(std::size_t)f];

  o.circulation.resize((std::size_t)(3 * nE + 3 * nF));
  for (int e = 0; e < nE; ++e)
    for (int b = 0; b < 3; ++b) o.circulation[(std::size_t)(3 * e + b)] = oe[(std::size_t)e];
  for (int f = 0; f < nF; ++f)
    for (int b = 0; b < 3; ++b) o.circulation[(std::size_t)(3 * nE + 3 * f + b)] = of[(std::size_t)f];

  o.flux.resize((std::size_t)(3 * nF));
  for (int f = 0; f < nF; ++f)
    for (int b = 0; b < 3; ++b) o.flux[(std::size_t)(3 * f + b)] = of[(std::size_t)f];

  o.vnodal.resize((std::size_t)(3 * nV));
  for (int v = 0; v < nV; ++v)
    for (int c = 0; c < 3; ++c) o.vnodal[(std::size_t)(3 * v + c)] = ov[(std::size_t)v];
  return o;
}

// Where assembly goes, when someone asks. Filled only if a pointer is passed,
// so the ordinary path pays nothing.
struct BdmComplexCost {
  double curl{0.0};      // gathering the cell's entity lists
  double flux{0.0};      // exokal's BDM flux product, per cell -- the Pi_rt basis
  double fit{0.0};       // T_f and the vertex-hat coefficients
  double scatter{0.0};   // writing the per-cell blocks into the accumulators
  double reduce{0.0};    // sort, sum duplicates, drop
  double grad{0.0};      // G, which is closed-form and touches no cell
  double total{0.0};
};

// The two differentials, for ANY cell shape.
//
// G and C are facet- and edge-local: neither reads a cell, so neither needs the
// degree-2 reconstruction and neither cares whether the mesh is simplicial.
// bdm_complex adds Pi_rt and Pi_nd, which do need a tetrahedron.
inline BdmComplex bdm_complex_differentials(const exokal::Mesh& mesh, int cell_dim) {
  using namespace bdm_detail;
  if (cell_dim != 3) throw std::invalid_argument("bdm_complex: 3D only");
  const graphos::Complex& top = mesh.topology();
  const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));

  BdmComplex out;
  out.n_flux = 3 * nF;
  out.n_circ = 3 * nE + 3 * nF;
  out.n_nodal = nV + 2 * nE + nF;
  out.n_vnodal = 3 * nV;

  const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
  const graphos::Adjacency f2e = graphos::incidence(top, 2, 1);

  const auto edge_ends = [&](Index e) {
    const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
    return std::pair<Index, Index>{e2v.indices[b], e2v.indices[b + 1]};
  };
  const auto nodal_edge = [&](Index e, int b) { return nV + 2 * int(e) + b; };
  const auto nodal_face = [&](Index f) { return nV + 2 * nE + int(f); };

  // ---------------- G : topological on the edges, Stokes on the facets ------
  {
    Sparse& G = out.grad;
    G.rows = out.n_circ;
    G.cols = out.n_nodal;
    for (Index e = 0; e < nE; ++e) {
      const auto [a, h] = edge_ends(e);
      const int r = 3 * int(e);
      G.push(r + 0, int(h), 1.0);
      G.push(r + 0, int(a), -1.0);
      G.push(r + 1, int(h), 1.0);
      G.push(r + 1, int(a), 1.0);
      G.push(r + 1, nodal_edge(e, 0), -2.0);
      G.push(r + 2, int(h), 1.0);
      G.push(r + 2, int(a), -1.0);
      G.push(r + 2, nodal_edge(e, 1), -6.0);
    }
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
      if (!(area > 0.0)) throw std::runtime_error("bdm_complex: degenerate facet");
      const P n{av[0] / area, av[1] / area, av[2] / area};
      const auto t = exokal::mimetic_curl_detail::tangent_frame(n);
      const P xf = exokal::centroid(mesh, 2, f);
      const double hf = std::sqrt(area);
      const int r = 3 * nE + 3 * int(f);
      for (auto k = f2e.offsets[(std::size_t)f]; k < f2e.offsets[(std::size_t)f + 1]; ++k) {
        const Index e = f2e.indices[(std::size_t)k];
        const auto [a, hh] = edge_ends(e);
        const P pa = mesh.point(a), ph = mesh.point(hh);
        const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
        const double len = std::sqrt(tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]);
        const P th{tau[0] / len, tau[1] / len, tau[2] / len};
        // w = edge midpoint - facet centroid, in the facet plane
        P w{0.5 * (pa[0] + ph[0]) - xf[0], 0.5 * (pa[1] + ph[1]) - xf[1],
            0.5 * (pa[2] + ph[2]) - xf[2]};
        const double wt = w[0] * th[0] + w[1] * th[1] + w[2] * th[2];
        P nu{w[0] - wt * th[0], w[1] - wt * th[1], w[2] - wt * th[2]};
        const double dl = std::sqrt(nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2]);
        if (!(dl > 0.0)) throw std::runtime_error("bdm_complex: degenerate facet edge");
        for (double& z : nu) z /= dl;
        const double de = dl;  // (x - x_f).nu is constant along e and equals |w_perp|
        const int col = nodal_edge(e, 0);
        G.push(r + 0, col, (t[0][0] * nu[0] + t[0][1] * nu[1] + t[0][2] * nu[2]) * len);
        G.push(r + 1, col, (t[1][0] * nu[0] + t[1][1] * nu[1] + t[1][2] * nu[2]) * len);
        G.push(r + 2, col, de * len / hf);
      }
      G.push(r + 2, nodal_face(f), -2.0 * area / hf);
    }
  }


  // ---------------- C : facet-wise, by surface Stokes -----------------------
  // d^1 is facet-local. On f the normal curl is the 2D surface curl of u's
  // TANGENTIAL TRACE, so against the facet chart chi_a
  //
  //     int_f (curl u).n chi_a = oint_df (u.tau) chi_a dl - int_f u . rot_s chi_a
  //
  // chi_a is affine, so the boundary term is each edge's chi_0 and chi_1
  // circulations, and rot_s chi_a = (-c_2, c_1) is a CONSTANT in-plane vector,
  // tested against the facet's own two tangential circulations. Every column is
  // a dof of f itself: no cell, no reconstruction, no shape assumption past a
  // planar facet with straight edges, and the two cells sharing f build the
  // same row because neither appears in the formula.
  //
  // dec/mimetic_curl.hpp computes the same operator per CELL from a
  // reconstruction. On a simplex D_edge = m and the fit is exact; on a polytope
  // D_edge > m, it is least squares, it couples the whole cell and the two
  // cells disagree. That is a property of the device, not of d^1 -- the earlier
  // claim here that no global C exists on a polytope was wrong.
  //
  // chi_a is exokal's own facet chart (facet_orthonormalization.hpp), so these
  // rows are already in the stabilized_bdm dof basis and need no change of
  // basis afterwards.
  {
    Sparse& Cf = out.curl;
    Cf.rows = out.n_flux;
    Cf.cols = out.n_circ;
    for (Index f = 0; f < nF; ++f) {
      const P av = exokal::face_area_vector(mesh, f);
      const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
      if (!(area > 0.0)) throw std::runtime_error("bdm_complex: degenerate facet");
      const P n{av[0] / area, av[1] / area, av[2] / area};
      const auto t = exokal::mimetic_curl_detail::tangent_frame(n);
      const P xf = exokal::centroid(mesh, 2, f);
      const auto dot3 = [](const P& u, const P& v) {
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      };
      // the chart, from the facet's own centred second moments
      double m11 = 0.0, m12 = 0.0, m22 = 0.0;
      const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, cell_dim, f, 4);
      for (std::size_t pq = 0; pq < q.weights.size(); ++pq) {
        const P r{q.points[pq][0] - xf[0], q.points[pq][1] - xf[1], q.points[pq][2] - xf[2]};
        const double x1 = dot3(r, t[0]), x2 = dot3(r, t[1]);
        m11 += q.weights[pq] * x1 * x1;
        m12 += q.weights[pq] * x1 * x2;
        m22 += q.weights[pq] * x2 * x2;
      }
      const exokal::hodge::FacetChart ch =
          exokal::hodge::facet_chart(area, m11, m12, m22, 3);
      // chi_0 = 1 ; chi_a = c1[a] xi_1 + c2[a] xi_2
      const double c1[3] = {0.0, ch.a11, ch.a21};
      const double c2[3] = {0.0, 0.0, ch.a22};
      const int r0 = 3 * int(f);
      const auto chi = [&](int a, const P& x) {
        const P r{x[0] - xf[0], x[1] - xf[1], x[2] - xf[2]};
        return (a == 0 ? 1.0 : 0.0) + c1[a] * dot3(r, t[0]) + c2[a] * dot3(r, t[1]);
      };
      for (auto k = f2e.offsets[(std::size_t)f]; k < f2e.offsets[(std::size_t)f + 1]; ++k) {
        const Index e = f2e.indices[(std::size_t)k];
        const auto [ea, eh] = edge_ends(e);
        const P pa = mesh.point(ea), ph = mesh.point(eh);
        const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
        // counterclockwise about n has (midpoint - x_f) x tau along +n
        const P w{0.5 * (pa[0] + ph[0]) - xf[0], 0.5 * (pa[1] + ph[1]) - xf[1],
                  0.5 * (pa[2] + ph[2]) - xf[2]};
        const P cr{w[1] * tau[2] - w[2] * tau[1], w[2] * tau[0] - w[0] * tau[2],
                   w[0] * tau[1] - w[1] * tau[0]};
        const double sgn = dot3(cr, n) >= 0.0 ? 1.0 : -1.0;
        for (int a = 0; a < 3; ++a) {
          // chi_a is affine along a straight edge, and int_0^1 (u.tau) s ds is
          // (chi_0 + chi_1)/2 in the shifted-Legendre circulation moments
          const double za = chi(a, pa), zb = chi(a, ph);
          Cf.push(r0 + a, 3 * int(e) + 0, sgn * (za + 0.5 * (zb - za)));
          Cf.push(r0 + a, 3 * int(e) + 1, sgn * 0.5 * (zb - za));
        }
      }
      for (int a = 1; a < 3; ++a) {
        Cf.push(r0 + a, 3 * nE + 3 * int(f) + 0, c2[a]);
        Cf.push(r0 + a, 3 * nE + 3 * int(f) + 1, -c1[a]);
      }
    }
  }
  return out;
}

// The complex of a mesh of any cell shape. G and C are facet-local (see
// bdm_complex_differentials); Pi_rt and Pi_nd write the vertex hats in the
// cell's modes, exactly on a tetrahedron and in least squares beyond it.
inline BdmComplex bdm_complex(const exokal::Mesh& mesh, int cell_dim,
                              BdmComplexCost* cost = nullptr) {
  using namespace bdm_detail;
  using Clock = std::chrono::steady_clock;
  const auto t_all = Clock::now();
  const auto tick = [](const auto& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  const auto t_g = Clock::now();
  BdmComplex out = bdm_complex_differentials(mesh, cell_dim);
  if (cost) cost->grad = tick(t_g);
  const graphos::Complex& top = mesh.topology();
  const int nE = int(top.count(1)), nF = int(top.count(2));
  const graphos::Adjacency c2v = graphos::incidence(top, 3, 0);
  const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
  const auto edge_ends = [&](Index e) {
    const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
    return std::pair<Index, Index>{e2v.indices[b], e2v.indices[b + 1]};
  };

  // ---------------- C, Pi_rt, Pi_nd : per cell, averaged over the cells -----
  Accum Prt(out.n_flux), Pnd(out.n_circ);
  double t_curl = 0.0, t_flux = 0.0, t_fit = 0.0, t_scatter = 0.0;
  // THE MESH-WIDE ADJACENCIES, ONCE. graphos::incidence builds a fresh CSR over
  // the whole complex and does not cache, so taking the mesh-only overload per
  // cell is O(cells x mesh); the cache makes the loop O(mesh).
  // THE FACET GEOMETRY Pi_nd NEEDS, once. area, centroid, the frame the
  // circulation dofs are stated in, and the ambient second moment
  // S = int_f (x - x_f) (x - x_f)^T, which is what the radial dof of an affine
  // field integrates to.
  struct FacetGeom {
    P xf{}, t0{}, t1{};
    double area{0.0};
    std::array<double, 9> S{};
  };
  std::vector<FacetGeom> fg((std::size_t)nF);
  for (Index f = 0; f < nF; ++f) {
    FacetGeom& G2 = fg[(std::size_t)f];
    const P av = exokal::face_area_vector(mesh, f);
    G2.area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
    const P n{av[0] / G2.area, av[1] / G2.area, av[2] / G2.area};
    const auto t = exokal::mimetic_curl_detail::tangent_frame(n);
    G2.t0 = t[0];
    G2.t1 = t[1];
    G2.xf = exokal::centroid(mesh, 2, f);
    const exokal::QuadratureRule q = exokal::facet_quadrature(mesh, cell_dim, f, 2);
    for (std::size_t p2 = 0; p2 < q.weights.size(); ++p2) {
      const P r{q.points[p2][0] - G2.xf[0], q.points[p2][1] - G2.xf[1],
                q.points[p2][2] - G2.xf[2]};
      for (int i2 = 0; i2 < 3; ++i2)
        for (int j2 = 0; j2 < 3; ++j2)
          G2.S[(std::size_t)(i2 * 3 + j2)] +=
              q.weights[p2] * r[(std::size_t)i2] * r[(std::size_t)j2];
    }
  }
  const graphos::Adjacency c2e = graphos::incidence(top, 3, 1);
  const graphos::Adjacency c2f = graphos::incidence(top, 3, 2);
  for (Index c = 0; c < mesh.count(cell_dim); ++c) {
    const auto vb = (std::size_t)c2v.offsets[(std::size_t)c];
    const auto ve = (std::size_t)c2v.offsets[(std::size_t)c + 1];
    const int nv = int(ve - vb);
    if (nv < 4) throw std::invalid_argument("bdm_complex: cell has fewer than four vertices");

    // PI_ND IN CLOSED FORM, so the degree-2 reconstruction is not built here.
    //
    // dec/mimetic_curl.hpp computes C, Pi_rt and Pi_nd together, and Pi_nd is
    // populated only for kind=full -- so asking for it used to drag in the
    // whole least-squares fit whose C this file no longer uses (it builds C
    // facet-wise). Measured at 125.9k cells that was 4.01 s of the complex's
    // 8.16 s. But Pi_nd is only the circulation dofs of the AFFINE modes, and
    // every one of those integrals is elementary -- the same argument that
    // writes C and G directly.
    const auto t0 = Clock::now();
    // THE FACET ORDER IS exokal's, NOT graphos'. fx.N is indexed by 3*f + b
    // with f the LOCAL facet position the flux product used, so the local order
    // has to be fx.faces -- taking graphos' incidence order instead misaligns
    // every Pi_rt row, which is what the linear-field assertion caught.
    std::vector<Index> ge;
    for (auto k = c2e.offsets[(std::size_t)c]; k < c2e.offsets[(std::size_t)c + 1]; ++k)
      ge.push_back(c2e.indices[(std::size_t)k]);
    const auto t1 = Clock::now();
    const auto fx = exokal::hodge::stabilized_bdm_flux_inner_product(
        mesh, cell_dim, c, exokal::hodge::SymTensor<>::isotropic(1.0));
    const auto t2 = Clock::now();
    t_curl += std::chrono::duration<double>(t1 - t0).count();
    t_flux += std::chrono::duration<double>(t2 - t1).count();
    const int cE = int(ge.size()), cF = int(fx.faces.size());
    const double hE = std::cbrt(exokal::measure(mesh, cell_dim, c));
    const P xE = exokal::centroid(mesh, cell_dim, c);

    // global column of each local circulation dof, and its rescaling
    std::vector<int> col((std::size_t)(3 * cE + 3 * cF));
    std::vector<double> cscale((std::size_t)(3 * cE + 3 * cF), 1.0);
    for (int e = 0; e < cE; ++e)
      for (int b = 0; b < 3; ++b) col[(std::size_t)(3 * e + b)] = 3 * int(ge[(std::size_t)e]) + b;
    for (int f = 0; f < cF; ++f) {
      const Index fid = fx.faces[(std::size_t)f];
      const P av = exokal::face_area_vector(mesh, fid);
      const double hf = std::sqrt(std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]));
      for (int b = 0; b < 3; ++b) {
        col[(std::size_t)(3 * cE + 3 * f + b)] = 3 * nE + 3 * int(fid) + b;
        // THE RADIAL DOF ALONE IS SCALED. exokal states it as int_f u.(x - x_f)
        // / h_E, the cell's length; G states the global one over the FACET's,
        // 1 / h_f, and Pi_nd has to land in G's convention -- so the local value
        // is carried over by h_E / h_f. Everything else agrees already.
        if (b == 2) cscale[(std::size_t)(3 * cE + 3 * f + b)] = hE / hf;
      }
    }

    // Lambda writes each vertex hat in the cell's modes {1, xi}, so that
    // lambda_w = sum_s Lambda[s][w] mode_s.
    //
    // On a tetrahedron V is 4 x 4 and Lambda = V^-1: the hats are the
    // barycentric coordinates, interpolatory by construction. On a polytope
    // nv > 4 and no nv interpolatory functions fit inside P1, so Lambda is the
    // pseudoinverse (V^T V)^-1 V^T. The hats stop being interpolatory, but
    // Lambda V = I still holds EXACTLY, so they reproduce every linear function
    // and in particular sum to one. That is what Pi has to carry: the coarse
    // space ADS hands BoomerAMG is the vector nodal one, and its near-nullspace
    // is the constants.
    std::vector<Index> vs;
    for (std::size_t i = vb; i < ve; ++i) vs.push_back(c2v.indices[i]);
    std::vector<double> V((std::size_t)(nv * 4), 0.0);
    for (int w = 0; w < nv; ++w) {
      const P x = mesh.point(vs[(std::size_t)w]);
      V[(std::size_t)(w * 4 + 0)] = 1.0;
      for (int j = 0; j < 3; ++j)
        V[(std::size_t)(w * 4 + 1 + j)] = (x[(std::size_t)j] - xE[(std::size_t)j]) / hE;
    }
    // the normal equations (V^T V) Lambda = V^T ; Lambda is 4 x nv row-major
    std::vector<double> VtV(16, 0.0), L((std::size_t)(4 * nv), 0.0);
    for (int s = 0; s < 4; ++s) {
      for (int s2 = 0; s2 < 4; ++s2) {
        double acc = 0.0;
        for (int w = 0; w < nv; ++w)
          acc += V[(std::size_t)(w * 4 + s)] * V[(std::size_t)(w * 4 + s2)];
        VtV[(std::size_t)(s * 4 + s2)] = acc;
      }
      for (int w = 0; w < nv; ++w)
        L[(std::size_t)(s * nv + w)] = V[(std::size_t)(w * 4 + s)];
    }
    // singular exactly when the cell's vertices are coplanar
    if (!solve_dense(VtV, L, 4, nv)) throw std::runtime_error("bdm_complex: degenerate cell");
    const auto t3 = Clock::now();
    t_fit += std::chrono::duration<double>(t3 - t2).count();

    for (int f = 0; f < cF; ++f) {
      const Index fid = fx.faces[(std::size_t)f];
      for (int b = 0; b < 3; ++b) {
        const int gr = 3 * int(fid) + b;
        Prt.touched(gr);
        // Pi_rt in exokal's dof basis IS fx.N; only the modes -> vertex hats
        for (int w = 0; w < nv; ++w)
          for (int cc = 0; cc < 3; ++cc) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s)
              acc += L[(std::size_t)(s * nv + w)] * fx.N((std::size_t)(3 * f + b), (std::size_t)(s * 3 + cc));
            Prt.add(gr, 3 * int(vs[(std::size_t)w]) + cc, acc);
          }
      }
    }
    // Pi_nd: the circulation dofs of the vertex hats.
    //
    // pnd[s][cc] is dof i applied to the mode field mode_s * e_cc, with the
    // modes {1, xi_1, xi_2, xi_3}, xi = (x - x_E)/h_E. Every entry is a closed
    // form:
    //
    //   edge, chi_b   mode_0 gives tau_cc on chi_0 and nothing else; mode_j is
    //                 affine along the edge, a + b t, and against the shifted
    //                 Legendre chi_b integrates to a + b/2, b/6, 0
    //   facet 0, 1    int_f mode_s (e_cc . t_a): |f| t_a[cc] for the constant,
    //                 and |f| t_a[cc] (x_f - x_E)_j / h_E for mode j, since
    //                 int_f (x - x_f) vanishes at the centroid
    //   facet 2       the radial dof, int_f mode_s (e_cc . (x - x_f)) / h_E:
    //                 zero for the constant, and S_{j,cc} / h_E^2 for mode j
    //
    // stated in exokal's own scaling (h_E on the radial dof) so cscale carries
    // it to G's convention exactly as before.
    double pnd[4][3];
    for (int i = 0; i < 3 * cE + 3 * cF; ++i) {
      const int gr = col[(std::size_t)i];
      Pnd.touched(gr);
      if (i < 3 * cE) {
        const int le = i / 3, b = i % 3;
        const auto [ea, eh] = edge_ends(ge[(std::size_t)le]);
        const P pa = mesh.point(ea), ph = mesh.point(eh);
        const P tau{ph[0] - pa[0], ph[1] - pa[1], ph[2] - pa[2]};
        for (int cc = 0; cc < 3; ++cc) {
          pnd[0][cc] = b == 0 ? tau[(std::size_t)cc] : 0.0;
          for (int j = 0; j < 3; ++j) {
            const double a0 = (pa[(std::size_t)j] - xE[(std::size_t)j]) / hE;
            const double a1 = tau[(std::size_t)j] / hE;
            pnd[j + 1][cc] = b == 0   ? tau[(std::size_t)cc] * (a0 + 0.5 * a1)
                             : b == 1 ? tau[(std::size_t)cc] * a1 / 6.0
                                      : 0.0;
          }
        }
      } else {
        const int lf = (i - 3 * cE) / 3, b = (i - 3 * cE) % 3;
        const FacetGeom& G2 = fg[(std::size_t)fx.faces[(std::size_t)lf]];
        const P& ta = b == 0 ? G2.t0 : G2.t1;
        for (int cc = 0; cc < 3; ++cc) {
          pnd[0][cc] = b == 2 ? 0.0 : G2.area * ta[(std::size_t)cc];
          for (int j = 0; j < 3; ++j)
            pnd[j + 1][cc] =
                b == 2 ? G2.S[(std::size_t)(j * 3 + cc)] / (hE * hE)
                       : G2.area * ta[(std::size_t)cc] *
                             (G2.xf[(std::size_t)j] - xE[(std::size_t)j]) / hE;
        }
      }
      for (int w = 0; w < nv; ++w)
        for (int cc = 0; cc < 3; ++cc) {
          double acc = 0.0;
          for (int s = 0; s < 4; ++s) acc += L[(std::size_t)(s * nv + w)] * pnd[s][cc];
          Pnd.add(gr, 3 * int(vs[(std::size_t)w]) + cc, acc * cscale[(std::size_t)i]);
        }
    }
  }

  if (cost) {
    cost->curl = t_curl;
    cost->flux = t_flux;
    cost->fit = t_fit;
    cost->scatter = t_scatter;
  }
  const auto t_red = Clock::now();
  out.pi_rt = Prt.finish(out.n_flux, out.n_vnodal);
  out.pi_nd = Pnd.finish(out.n_circ, out.n_vnodal);
  if (cost) {
    cost->reduce = tick(t_red);
    cost->total = tick(t_all);
    cost->scatter = cost->total - cost->curl - cost->flux - cost->fit -
                    cost->reduce - cost->grad;
  }
  return out;
}

}  // namespace mimetika::solver
