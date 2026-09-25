#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/stress_operators.hpp"
#include "exokal/preprocess/mixed_dimensional_mesh.hpp"
#include "mimetika/model/boundary.hpp"

// The two face-local matrices the interface rows are built from.
//
//     (A_c,f)_ij = int_f  D_c psi_j . psi_i df      the compliance
//     (G_f)_ij   = int_f      psi_j . psi_i df      the chart's Gram
//
// both over T_h(f), the face's traction space, and both symmetric. They are
// face-local because the formulation takes Lambda_c,h and U_c,h broken face by
// face: no continuity is asked of them across a face boundary, so A_c and J are
// block diagonal and this holds one block per fracture cell.
//
// WHY THE GRAM IS HERE AND NOT ASSUMED. j(s_c, v_c) = -<s_c, [[v_c]]> pairs two
// expansions in the same chart, so its matrix is that chart's Gram -- not the
// identity, and not |f| times it unless the chart is L^2-orthonormal, which is
// a property of the chart and not of the formulation. The mimetic facet charts
// are orthonormal up to |f|, so G = |f| I there; the wrench modes of the
// strongly symmetric family are a vector basis and carry their own Gram. Both
// are stated here rather than guessed in a kernel.
//
// THE CHART IS THE AMBIENT FACET'S. u_c and t_c are expanded in the basis the
// carrying facet's traction moments are taken against -- that is what makes
// terms/contact_trace.hpp an identity -- so these matrices must be built in
// that same chart, from the ambient side, and not from a chart rebuilt on the
// fracture's own geometry.

namespace mimetika::physics {

// The smallest eigenvalue of a small symmetric matrix, by cyclic Jacobi.
//
// Eigenvalues rather than a Cholesky because the answer wanted is not "is it
// definite" but HOW negative it is: a compliance that misses positivity by
// 1e-16 of its norm is round-off in the chart's quadrature and a compliance
// that misses it by its whole norm is a sign error, and only a number tells
// them apart in the message. k is 6 or 9, so this is O(k^3) per face, once, at
// setup.
inline double smallest_eigenvalue(const double* M, std::size_t k) {
  std::vector<double> a(M, M + k * k);
  for (int sweep = 0; sweep < 30; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < k; ++p) {
      for (std::size_t q = p + 1; q < k; ++q) off += a[p * k + q] * a[p * k + q];
    }
    if (off <= 1e-30) break;
    for (std::size_t p = 0; p < k; ++p) {
      for (std::size_t q = p + 1; q < k; ++q) {
        const double apq = a[p * k + q];
        if (apq == 0.0) continue;
        const double theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
        const double t =
            (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
        for (std::size_t r = 0; r < k; ++r) {
          const double arp = a[r * k + p], arq = a[r * k + q];
          a[r * k + p] = c * arp - sn * arq;
          a[r * k + q] = sn * arp + c * arq;
        }
        for (std::size_t r = 0; r < k; ++r) {
          const double apr = a[p * k + r], aqr = a[q * k + r];
          a[p * k + r] = c * apr - sn * aqr;
          a[q * k + r] = sn * apr + c * aqr;
        }
      }
    }
  }
  double lo = a[0];
  for (std::size_t i = 1; i < k; ++i) lo = std::min(lo, a[i * k + i]);
  return lo;
}

inline double worst_asymmetry(const double* M, std::size_t k, double& scale) {
  double worst = 0.0;
  scale = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      scale = std::max(scale, std::fabs(M[i * k + j]));
      worst = std::max(worst, std::fabs(M[i * k + j] - M[j * k + i]));
    }
  }
  return worst;
}

struct ContactInterface {
  std::size_t k{0};               // dim T_h(f)
  std::vector<double> gram;       // k*k per fracture cell, row-major
  std::vector<double> compliance; // k*k per fracture cell, row-major

  std::size_t faces() const { return k == 0 ? 0 : gram.size() / (k * k); }

  const double* gram_at(std::size_t face) const { return block(gram, face, "gram"); }
  const double* compliance_at(std::size_t face) const {
    return block(compliance, face, "compliance");
  }

  void validate() const {
    if (k == 0) throw std::invalid_argument("ContactInterface: the face traction space is empty");
    if (gram.size() != compliance.size()) {
      throw std::invalid_argument("ContactInterface: gram and compliance disagree on the faces");
    }
    if (gram.size() % (k * k) != 0) {
      throw std::invalid_argument("ContactInterface: the blocks are not k by k");
    }

    // A_c IS A COMPLIANCE AND IS POSITIVE. Both blocks are integrals of an
    // inner product over a face, so both are symmetric, the Gram is definite,
    // and the compliance is at worst semi-definite. That is not bookkeeping:
    //
    //   physically, D_c positive is what makes a traction and the jump it
    //   produces point the same way. Negate it and compression OPENS the
    //   fracture -- the discrete problem stays solvable and answers a
    //   different question, which is the failure a solver never reports;
    //
    //   algebraically, [A_c J; J^T 0] is a saddle point only while A_c is
    //   positive semi-definite. Indefinite, it has neither the inertia MINRES
    //   assumes nor the sign structure a block preconditioner is built on.
    //
    // A_c = 0 IS ADMITTED, and is the one case that matters most: it is the
    // bonded limit, c_n and c_T infinite, where the cut problem is the uncut
    // problem and the patch test is reproduced exactly. Semi-definite, not
    // definite, is therefore the condition -- a definiteness test here would
    // reject the very configuration that validates the formulation.
    const std::size_t stride = k * k;
    const std::size_t faces = gram.size() / stride;
    for (std::size_t f = 0; f < faces; ++f) {
      double gs = 0.0, as = 0.0;
      const double* G = gram.data() + f * stride;
      const double* A = compliance.data() + f * stride;
      const double g_skew = worst_asymmetry(G, k, gs);
      const double a_skew = worst_asymmetry(A, k, as);
      if (g_skew > 1e-12 * std::max(gs, 1.0)) {
        throw std::invalid_argument("ContactInterface: the chart Gram on face " +
                                    std::to_string(f) + " is not symmetric, by " +
                                    std::to_string(g_skew));
      }
      if (a_skew > 1e-12 * std::max(as, 1.0)) {
        throw std::invalid_argument("ContactInterface: the compliance on face " +
                                    std::to_string(f) + " is not symmetric, by " +
                                    std::to_string(a_skew));
      }
      if (!(gs > 0.0) || smallest_eigenvalue(G, k) <= 1e-12 * gs) {
        throw std::invalid_argument(
            "ContactInterface: the chart Gram on face " + std::to_string(f) +
            " is not positive definite -- it is <psi_i, psi_j> for a basis, so a non-positive "
            "eigenvalue means the face traction space is degenerate and the jump J u_c = 0 does "
            "not say u_c^+ = u_c^-");
      }
      // exactly zero is the bonded interface and is the point; anything else
      // must be a compliance
      if (as > 0.0 && smallest_eigenvalue(A, k) < -1e-10 * as) {
        throw std::invalid_argument(
            "ContactInterface: the compliance on face " + std::to_string(f) +
            " is not positive semi-definite, its smallest eigenvalue being " +
            std::to_string(smallest_eigenvalue(A, k)) + " against a norm of " +
            std::to_string(as) +
            " -- D_c is a compliance, so a negative direction is one in which traction and jump "
            "oppose each other and compression opens the fracture. A_c = 0 is admitted and is "
            "the bonded limit");
      }
    }
  }

 private:
  const double* block(const std::vector<double>& v, std::size_t face, const char* what) const {
    const std::size_t stride = k * k;
    if (stride == 0 || (face + 1) * stride > v.size()) {
      throw std::out_of_range(std::string("ContactInterface: no ") + what + " for that face");
    }
    return v.data() + face * stride;
  }
};


// ---------------------------------------------------------------------------
// The chart, from the ambient facet, and the law on top of it.
//
// These are separate on purpose. The chart is a property of the DISCRETIZATION:
// which basis the face's tractions are written in, and the two integrals of it
// that any interface law needs. The law is what turns those into a compliance,
// and the linear one below is an example of a law, not the only one -- a
// nonlinear normal compliance or a Coulomb friction law consumes the same
// chart, replaces only `contact_compliance_cell`, and leaves the chart, the
// jump and the trace coupling untouched.

// Per fracture face, in the basis of the ambient facet that carries it:
//
//     G_ij = int_f psi_i . psi_j ,      N_ij = int_f (psi_i . n)(psi_j . n)
//
// G is what the jump pairs against; N is the part of any normal/tangential
// split that is not G, and the two together express every D_c of the form
// alpha (n (x) n) + beta (I - n (x) n) without further geometry.
struct InterfaceChart {
  std::size_t k{0};
  std::vector<double> gram;         // k*k per fracture cell
  std::vector<double> normal_gram;  // k*k per fracture cell
};

// The two families differ in what a facet's traction moments ARE, so the chart
// is built per family and nothing here is shared by accident:
//
//   wrench      one component, d(d+1)/2 rigid-motion modes, the first
//               d(d-1)/2 + ... tangential and the rest normal. The basis is
//               normalized so that its Gram is |f| I -- exokal states it where
//               the strong displacement datum is expanded -- so G = |f| I and
//               N is |f| on the normal modes, nothing on the tangential ones,
//               the modes being purely one or the other.
//
//   componentwise   d ambient components against a scalar chart chi_b, so
//               psi_(b,k) = chi_b e_k. Then G = S (x) I with S the scalar chart
//               Gram the operators already hold, and N = S (x) (n (x) n).
//
// Both order a facet's unknowns as b * components + k, which is the ordering
// the boundary terms index with.
inline InterfaceChart interface_chart(const exokal::MixedDimensionalMesh& md,
                                      const exokal::spaces::ProductSpace& ambient_space,
                                      const std::string& stress_field,
                                      const exokal::hodge::StressOperators& ops, int dim) {
  const std::vector<const exokal::Stratum*> fractures = md.strata_of_codim(1);
  if (fractures.size() != 1) {
    throw std::invalid_argument("interface_chart: expected exactly one fracture stratum");
  }
  const exokal::Mesh& top = md.top.mesh;
  const graphos::Complex& tc = top.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(tc, dim - 1);

  const std::size_t fi = ambient_space.index_of(stress_field);
  const exokal::spaces::DofLayout& lay = ambient_space.map(fi).layout();
  const std::size_t nb = static_cast<std::size_t>(lay.on(dim - 1));
  const std::size_t nc = static_cast<std::size_t>(lay.components);
  const bool wrench = nc == 1 && (nb == 6 || nb == 3);
  const std::size_t n_tangential = nb == 6 ? 3 : 1;  // as free slip pins them

  InterfaceChart chart;
  chart.k = nb * nc;
  const std::size_t stride = chart.k * chart.k;
  const auto& facet_to_ambient = md.top.to_ambient.index[static_cast<std::size_t>(dim - 1)];
  const auto& cell_to_ambient = fractures[0]->to_ambient.index[static_cast<std::size_t>(dim - 1)];
  const graphos::Index n_faces = fractures[0]->mesh.topology().count(dim - 1);
  chart.gram.assign(static_cast<std::size_t>(n_faces) * stride, 0.0);
  chart.normal_gram.assign(chart.gram.size(), 0.0);

  for (graphos::Index face = 0; face < n_faces; ++face) {
    // any carrying copy with a cell of its own: the chart is the facet's, and
    // the two sides share it
    const graphos::Index ambient_facet = cell_to_ambient[static_cast<std::size_t>(face)];
    graphos::Index facet = graphos::invalid_index, cell = graphos::invalid_index;
    for (graphos::Index f = 0; f < tc.count(dim - 1) && facet == graphos::invalid_index; ++f) {
      if (facet_to_ambient[static_cast<std::size_t>(f)] != ambient_facet) continue;
      const auto b = cob.offsets[static_cast<std::size_t>(f)];
      if (cob.offsets[static_cast<std::size_t>(f) + 1] == b) continue;  // the detached copy
      facet = f;
      cell = cob.indices[static_cast<std::size_t>(b)];
    }
    if (facet == graphos::invalid_index) {
      throw std::invalid_argument("interface_chart: a fracture face is carried by no bulk cell");
    }
    const FacetFrame fr = FacetFrame::of(top, dim, cell, facet);
    const exokal::hodge::StressOperators::Cell& c = ops.compact(cell);
    double* G = chart.gram.data() + static_cast<std::size_t>(face) * stride;
    double* N = chart.normal_gram.data() + static_cast<std::size_t>(face) * stride;

    if (wrench) {
      for (std::size_t b = 0; b < nb; ++b) {
        G[b * chart.k + b] = fr.measure;
        if (b >= n_tangential) N[b * chart.k + b] = fr.measure;
      }
      continue;
    }

    // the scalar chart's Gram, as the operators hold it; |f| I where a product
    // stores none, which is the fallback the displacement datum already takes
    std::size_t slot = c.faces.size();
    for (std::size_t j = 0; j < c.faces.size(); ++j) {
      if (c.faces[j] == facet) slot = j;
    }
    if (slot == c.faces.size()) {
      throw std::invalid_argument("interface_chart: the carrying facet is not a face of its cell");
    }
    const bool has_gram = slot < c.facet_gram.size() && c.facet_gram[slot].rows() == nb;
    for (std::size_t b = 0; b < nb; ++b) {
      for (std::size_t bb = 0; bb < nb; ++bb) {
        const double S = has_gram ? c.facet_gram[slot](b, bb) : (b == bb ? fr.measure : 0.0);
        if (S == 0.0) continue;
        for (std::size_t k = 0; k < nc; ++k) {
          for (std::size_t kk = 0; kk < nc; ++kk) {
            const std::size_t i = b * nc + k, j = bb * nc + kk;
            if (k == kk) G[i * chart.k + j] += S;
            N[i * chart.k + j] += S * fr.normal[k] * fr.normal[kk];
          }
        }
      }
    }
  }
  return chart;
}

// D_c = (1/c_n) n (x) n + (1/c_T) (I - n (x) n), AN EXAMPLE OF A CONTACT LAW:
// the linear compliance of Martins and Oden, which is what makes the whole
// problem linear. In the chart that is
//
//     A_c = (1/c_T) G + (1/c_n - 1/c_T) N ,
//
// one line, because the chart already carries the geometry. A law that is not
// linear supplies its own A_c from the same two integrals and its own state;
// only this function and the compliance term change.
inline ContactInterface linear_contact(const InterfaceChart& chart, double c_n, double c_T) {
  if (!(c_n > 0.0) || !(c_T > 0.0)) {
    throw std::invalid_argument(
        "linear_contact: the normal and tangential stiffnesses must be positive, or D_c is not an "
        "inner product and the interface row is singular");
  }
  ContactInterface out;
  out.k = chart.k;
  out.gram = chart.gram;
  out.compliance.assign(chart.gram.size(), 0.0);
  const double a = 1.0 / c_T, b = 1.0 / c_n - 1.0 / c_T;
  for (std::size_t i = 0; i < out.compliance.size(); ++i) {
    out.compliance[i] = a * chart.gram[i] + b * chart.normal_gram[i];
  }
  out.validate();
  return out;
}

// The rigid limit of that family: D_c = 0, so the interface is bonded and the
// compliance row reads [[u_c]] = 0.
//
// It is not a degenerate case to be guarded against but the one configuration
// with a closed form: a bonded fracture transmits everything and deforms not at
// all, so the cut problem is the uncut problem and any exact solution of the
// latter must be reproduced to round-off. That is what makes a patch test
// possible, and it is the only check that pins the SIGNS of the coupling --
// T against J -- rather than their pattern. Flip either and a bonded interface
// either goes singular or transmits the wrong side's traction.
//
// The Gram is kept: J is the Gram-weighted jump, and with G positive definite
// J u_c = 0 is exactly u_c^+ = u_c^-. Zeroing it too would delete the
// constraint rather than stiffen them.
inline ContactInterface rigid_contact(const InterfaceChart& chart) {
  ContactInterface ci;
  ci.k = chart.k;
  ci.gram = chart.gram;
  ci.compliance.assign(chart.gram.size(), 0.0);
  ci.validate();
  return ci;
}

}  // namespace mimetika::physics
