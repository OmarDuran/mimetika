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

// Is every cell a tetrahedron? The degree-2 curl is facet-local only there --
// on a polytope D_edge > m, the reconstruction is a least-squares fit and a
// facet's rows reach the whole cell, so no global C exists without a choice.
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
  double curl{0.0};      // exokal's degree-2 curl, per cell
  double flux{0.0};      // exokal's BDM flux product, per cell -- the Pi_rt basis
  double fit{0.0};       // T_f and the vertex-hat coefficients
  double scatter{0.0};   // writing the per-cell blocks into the accumulators
  double reduce{0.0};    // sort, sum duplicates, drop
  double grad{0.0};      // G, which is closed-form and touches no cell
  double total{0.0};
};

// The complex of a tetrahedral mesh. Throws if a cell is not a tetrahedron --
// on a general polytope the degree-2 reconstruction is a least-squares fit
// (D_edge > m) and a facet's curl rows stop being facet-local, so no global C
// exists without a choice.
inline BdmComplex bdm_complex(const exokal::Mesh& mesh, int cell_dim,
                              BdmComplexCost* cost = nullptr) {
  using namespace bdm_detail;
  using exokal::MimeticCurlKind;
  using Clock = std::chrono::steady_clock;
  const auto t_all = Clock::now();
  const auto tick = [](const auto& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
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
  const graphos::Adjacency c2v = graphos::incidence(top, 3, 0);

  const auto edge_ends = [&](Index e) {
    const auto b = (std::size_t)e2v.offsets[(std::size_t)e];
    return std::pair<Index, Index>{e2v.indices[b], e2v.indices[b + 1]};
  };
  const auto nodal_edge = [&](Index e, int b) { return nV + 2 * int(e) + b; };
  const auto nodal_face = [&](Index f) { return nV + 2 * nE + int(f); };

  // ---------------- G : topological on the edges, Stokes on the facets ------
  const auto t_g = Clock::now();
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

  if (cost) cost->grad = tick(t_g);

  // ---------------- C, Pi_rt, Pi_nd : per cell, averaged over the cells -----
  Accum C(out.n_flux), Prt(out.n_flux), Pnd(out.n_circ);
  double t_curl = 0.0, t_flux = 0.0, t_fit = 0.0, t_scatter = 0.0;
  // THE MESH-WIDE ADJACENCIES, ONCE. graphos::incidence builds a fresh CSR over
  // the whole complex and does not cache, so taking the mesh-only overload per
  // cell is O(cells x mesh); the cache makes the loop O(mesh).
  const exokal::MimeticCurlCache curl_cache(mesh, cell_dim);
  for (Index c = 0; c < mesh.count(cell_dim); ++c) {
    const auto vb = (std::size_t)c2v.offsets[(std::size_t)c];
    const auto ve = (std::size_t)c2v.offsets[(std::size_t)c + 1];
    if (ve - vb != 4) throw std::invalid_argument("bdm_complex: cell is not a tetrahedron");

    const auto t0 = Clock::now();
    const auto g = exokal::mimetic_curl(mesh, curl_cache, c, MimeticCurlKind::full);
    const auto t1 = Clock::now();
    const auto fx = exokal::hodge::stabilized_bdm_flux_inner_product(
        mesh, cell_dim, c, exokal::hodge::SymTensor<>::isotropic(1.0));
    const auto t2 = Clock::now();
    t_curl += std::chrono::duration<double>(t1 - t0).count();
    t_flux += std::chrono::duration<double>(t2 - t1).count();
    const int cE = int(g.edges.size()), cF = int(g.faces.size());
    const double hE = std::cbrt(exokal::measure(mesh, cell_dim, c));
    const P xE = exokal::centroid(mesh, cell_dim, c);

    // global column of each local circulation dof, and its rescaling
    std::vector<int> col((std::size_t)(3 * cE + 3 * cF));
    std::vector<double> cscale((std::size_t)(3 * cE + 3 * cF), 1.0);
    for (int e = 0; e < cE; ++e)
      for (int b = 0; b < 3; ++b) col[(std::size_t)(3 * e + b)] = 3 * int(g.edges[(std::size_t)e]) + b;
    for (int f = 0; f < cF; ++f) {
      const Index fid = g.faces[(std::size_t)f];
      const P av = exokal::face_area_vector(mesh, fid);
      const double hf = std::sqrt(std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]));
      for (int b = 0; b < 3; ++b) {
        col[(std::size_t)(3 * cE + 3 * f + b)] = 3 * nE + 3 * int(fid) + b;
        // the radial dof alone carries 1/h_E; move it to the facet's own scale
        if (b == 2) cscale[(std::size_t)(3 * cE + 3 * f + b)] = hf / hE;
      }
    }

    // the 4 x 4 that writes each vertex hat in the cell's modes {1, xi}
    std::vector<double> V(16), L(16, 0.0);
    std::vector<Index> vs;
    for (std::size_t i = vb; i < ve; ++i) vs.push_back(c2v.indices[i]);
    for (int w = 0; w < 4; ++w) {
      const P x = mesh.point(vs[(std::size_t)w]);
      V[(std::size_t)(w * 4 + 0)] = 1.0;
      for (int j = 0; j < 3; ++j) V[(std::size_t)(w * 4 + 1 + j)] = (x[(std::size_t)j] - xE[(std::size_t)j]) / hE;
      L[(std::size_t)(w * 4 + w)] = 1.0;
    }
    // V (4x4) * Lambda = I  =>  lambda_w = sum_s Lambda[s][w] mode_s
    if (!solve_dense(V, L, 4, 4)) throw std::runtime_error("bdm_complex: degenerate tetrahedron");
    const auto t3 = Clock::now();
    t_fit += std::chrono::duration<double>(t3 - t2).count();

    for (int f = 0; f < cF; ++f) {
      const Index fid = g.faces[(std::size_t)f];
      // T_f : the mimetic-curl flux moments -> exokal's stabilized_bdm dofs.
      // Both are 3 x 12 on the same modes, so T = N A^T (A A^T)^-1.
      std::vector<double> Gm(9, 0.0), BA(9, 0.0);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          double gg = 0.0, ba = 0.0;
          for (int k = 0; k < 12; ++k) {
            const double a_i = g.pi_rt((std::size_t)(3 * f + i), (std::size_t)k);
            const double a_j = g.pi_rt((std::size_t)(3 * f + j), (std::size_t)k);
            const double b_i = fx.N((std::size_t)(3 * f + i), (std::size_t)k);
            gg += a_i * a_j;
            ba += b_i * a_j;
          }
          Gm[(std::size_t)(i * 3 + j)] = gg;
          BA[(std::size_t)(i * 3 + j)] = ba;
        }
      // solve T Gm = BA  <=>  Gm^T T^T = BA^T ; Gm is symmetric
      std::vector<double> Tt(9);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Tt[(std::size_t)(i * 3 + j)] = BA[(std::size_t)(j * 3 + i)];
      if (!solve_dense(Gm, Tt, 3, 3)) throw std::runtime_error("bdm_complex: singular facet basis");
      const auto T = [&](int i, int j) { return Tt[(std::size_t)(j * 3 + i)]; };

      for (int b = 0; b < 3; ++b) {
        const int gr = 3 * int(fid) + b;
        C.touched(gr);
        Prt.touched(gr);
        for (std::size_t j = 0; j < g.C.cols(); ++j) {
          double acc = 0.0;
          for (int q = 0; q < 3; ++q) acc += T(b, q) * g.C((std::size_t)(3 * f + q), j);
          if (acc != 0.0) C.add(gr, col[j], acc * cscale[j]);
        }
        // Pi_rt in exokal's dof basis IS fx.N; only the modes -> vertex hats
        for (int w = 0; w < 4; ++w)
          for (int cc = 0; cc < 3; ++cc) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s)
              acc += L[(std::size_t)(s * 4 + w)] * fx.N((std::size_t)(3 * f + b), (std::size_t)(s * 3 + cc));
            Prt.add(gr, 3 * int(vs[(std::size_t)w]) + cc, acc);
          }
      }
    }
    // Pi_nd: the circulation dofs of the vertex hats
    for (int i = 0; i < 3 * cE + 3 * cF; ++i) {
      const int gr = col[(std::size_t)i];
      Pnd.touched(gr);
      for (int w = 0; w < 4; ++w)
        for (int cc = 0; cc < 3; ++cc) {
          double acc = 0.0;
          for (int s = 0; s < 4; ++s)
            acc += L[(std::size_t)(s * 4 + w)] * g.pi_nd((std::size_t)i, (std::size_t)(s * 3 + cc));
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
  out.curl = C.finish(out.n_flux, out.n_circ);
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
