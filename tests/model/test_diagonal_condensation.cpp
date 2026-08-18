#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/linear_solver/condense.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/model/single_phase_model.hpp"

// THE DIAGONAL PRODUCTS CONDENSE, AND WHAT IS LEFT IS THE FINITE VOLUME METHOD.
//
// A diagonal star makes the first block of the saddle point diagonal, so the
// flux or the stress is eliminated CELL BY CELL -- no factorization, no fill,
// one division per unknown -- and the Schur complement is the whole system:
//
//     [ M  A01 ] [ x0 ]   [ b0 ]                      -1
//     [ A10 C  ] [ y  ] = [ b1 ]  =>  S = C - A10 M     A01
//
// For diagonal_tpfa S is the pressure alone and IS the two-point flux
// approximation: seven entries a row on a Cartesian mesh in space. For
// diagonal_tpsa it is the displacement, the rotation and the total pressure,
// which is the FV-TPSA system of Nordbotten & Keilegavlen (their Eq. 3.9) --
// the same three cell-centered unknowns, reached by eliminating a stress those
// authors never introduce.
//
// WHAT THIS IS NOT. exokal owns the inner product M and tests it: diagonal
// under condensation, and positive. Nothing here re-litigates that -- the
// premise below only CONFIRMS the block is diagonal, so that the specialization
// is applied to the products it belongs to. What is measured here is the
// GLOBAL Schur complement, which is M together with the divergence, the weak
// symmetry, the trace and the total-pressure row: a different matrix, and the
// one a solver is actually handed.
//
// This file asserts, on meshes small enough to take dense linear algebra to:
//
//   the first block is diagonal   -- and is NOT for the de Rham and stabilized
//                                    products, so the specialization is theirs
//   S has the two-point stencil   -- every entry couples two cells sharing a
//                                    facet, and no others
//   S is symmetric               -- which the assembled matrix is not, since a
//                                    pinned row is one the constraint set
//                                    rewrote; the symmetry REAPPEARS under the
//                                    elimination
//   S solves the same problem     -- condensed answer against the saddle
//                                    point's own, to round-off
//
// WHAT KIND OF SYMMETRIC IS NOT THE SAME FOR THE TWO PRODUCTS, and it is the
// one thing a solver has to be told:
//
//   diagonal_tpfa   S is the pressure alone and is POSITIVE DEFINITE, as
//                   assembled, with no sign to flip: conjugate gradients and
//                   an algebraic multigrid, and the matrix IS the two-point
//                   flux approximation -- seven entries a row on a Cartesian
//                   mesh in space.
//
//   diagonal_tpsa   S is displacement, rotation and total pressure, and is
//                   SYMMETRIC QUASI-DEFINITE: (u, r) positive definite, p
//                   negative definite. That is not SPD and no scaling makes it
//                   so -- a diagonal similarity cannot change the sign of a
//                   diagonal block -- so it is MINRES or an LDL^T, not CG.
//                   Negating the pressure EQUATION alone trades the symmetry
//                   for coercivity and gives the form the finite volume
//                   literature writes: Nordbotten & Keilegavlen Eq. (3.9),
//                   whose weak form (their 5.5) pairs +(p, div u') against
//                   -(div u, p'). Both are asserted below.
//
//                   On TETRAHEDRA it is invertible but not quasi-definite --
//                   see Case below for the ratio that decides it.
//
// BOUNDARY DATA. The cases here prescribe the pressure and the displacement,
// which are natural in a mixed form. An ESSENTIAL condition on the first field
// -- a traction, or the tangential half of a roller -- pins stress unknowns,
// and for diagonal_tpsa that costs [Dv; As] its row rank, which leaves S
// singular rather than definite. That is a property of the rotation closure and
// not of the condensation, and it is not asserted here.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::SinglePhaseModel;
using mimetika::mesh::Family;
using mimetika::solver::SparseSystem;
using Formulation = CauchyElasticityModel::Formulation;
using Stress = CauchyElasticityModel::Realization;
using Flux = SinglePhaseModel::Realization;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;

// ---- dense algebra, for systems of a few hundred unknowns -------------------

struct Dense {
  std::size_t n{0};
  std::vector<double> a;

  Dense() = default;
  Dense(std::size_t rows, std::size_t cols) : n(cols), a(rows * cols, 0.0) {}

  double& at(std::size_t i, std::size_t j) { return a[i * n + j]; }
  double at(std::size_t i, std::size_t j) const { return a[i * n + j]; }
  std::size_t rows() const { return n == 0 ? 0 : a.size() / n; }
};

// The triplets summed, which is where duplicates are resolved: the assembly
// emits one per contribution and the matrix is their sum.
Dense dense_of(const SparseSystem& A) {
  Dense out(A.n, A.n);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    out.at(static_cast<std::size_t>(A.row[k]), static_cast<std::size_t>(A.col[k])) += A.value[k];
  }
  return out;
}

// in place, and false the moment a pivot is not positive: the factorization IS
// the definiteness test, so no eigenvalue is asked for
bool cholesky(Dense& L) {
  const std::size_t m = L.rows();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = L.at(i, j);
      for (std::size_t k = 0; k < j; ++k) s -= L.at(i, k) * L.at(j, k);
      if (i == j) {
        if (!(s > 0.0)) return false;
        L.at(i, i) = std::sqrt(s);
      } else {
        L.at(i, j) = s / L.at(j, j);
      }
    }
    for (std::size_t j = i + 1; j < m; ++j) L.at(i, j) = 0.0;
  }
  return true;
}

// a diagonal block of S, on the unknowns a predicate selects
template <class Keep>
Dense diagonal_block(const Dense& S, const Keep& keep) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < S.rows(); ++i) {
    if (keep(i)) idx.push_back(i);
  }
  Dense out(idx.size(), idx.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    for (std::size_t j = 0; j < idx.size(); ++j) out.at(i, j) = S.at(idx[i], idx[j]);
  }
  return out;
}

Dense negated(const Dense& S) {
  Dense out = S;
  for (double& v : out.a) v = -v;
  return out;
}

Dense symmetric_part(const Dense& S) {
  Dense out(S.rows(), S.rows());
  for (std::size_t i = 0; i < S.rows(); ++i) {
    for (std::size_t j = 0; j < S.rows(); ++j) out.at(i, j) = 0.5 * (S.at(i, j) + S.at(j, i));
  }
  return out;
}

// the smallest pivot the elimination met, over the largest: a singular matrix
// has no pivot left at the end, and this is what says so
double pivot_ratio = 0.0;
std::size_t tiny_pivots = 0;

// Gaussian elimination with partial pivoting: the quasi-definite system is not
// a Cholesky's to take, and this is a few hundred unknowns
std::vector<double> lu_solve(Dense A, std::vector<double> b) {
  const std::size_t m = A.rows();
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < m; ++i) {
      if (std::abs(A.at(i, k)) > std::abs(A.at(pivot, k))) pivot = i;
    }
    for (std::size_t j = 0; j < m; ++j) std::swap(A.at(k, j), A.at(pivot, j));
    std::swap(b[k], b[pivot]);
    for (std::size_t i = k + 1; i < m; ++i) {
      const double f = A.at(i, k) / A.at(k, k);
      if (f == 0.0) continue;
      for (std::size_t j = k; j < m; ++j) A.at(i, j) -= f * A.at(k, j);
      b[i] -= f * b[k];
    }
  }
  double lo = 1e300, hi = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    lo = std::min(lo, std::abs(A.at(i, i)));
    hi = std::max(hi, std::abs(A.at(i, i)));
  }
  pivot_ratio = hi > 0.0 ? lo / hi : 0.0;
  tiny_pivots = 0;
  for (std::size_t i = 0; i < m; ++i) {
    if (std::abs(A.at(i, i)) < 1e-10 * hi) ++tiny_pivots;
  }
  for (std::size_t i = m; i-- > 0;) {
    for (std::size_t j = i + 1; j < m; ++j) b[i] -= A.at(i, j) * b[j];
    b[i] /= A.at(i, i);
  }
  return b;
}

std::vector<double> cholesky_solve(const Dense& L, std::vector<double> b) {
  const std::size_t m = L.rows();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < i; ++k) b[i] -= L.at(i, k) * b[k];
    b[i] /= L.at(i, i);
  }
  for (std::size_t i = m; i-- > 0;) {
    for (std::size_t k = i + 1; k < m; ++k) b[i] -= L.at(k, i) * b[k];
    b[i] /= L.at(i, i);
  }
  return b;
}

// ---- the condensation ------------------------------------------------------

// The first field's unknowns, and everything else in ascending order. The first
// field is the flux or the stress: the one the diagonal star is a star ON.
struct Split {
  std::vector<std::size_t> first, rest;
  std::vector<int> cell_of;   // which cell each `rest` unknown belongs to
  std::vector<int> field_of;  // and which field, counted from the second
  std::vector<std::string> names;
};

template <class Model>
Split split_of(const Model& m) {
  const auto blocks = mimetika::solver::field_blocks(m.simulation().epoch());
  const auto cells = static_cast<std::size_t>(m.n_cells());
  Split s;
  for (const Index g : blocks[0].indices()) s.first.push_back(static_cast<std::size_t>(g));
  for (std::size_t f = 1; f < blocks.size(); ++f) {
    // a cell field is laid out entity-major, so dof k of a field with c
    // components per cell belongs to cell k / c
    const std::size_t components = blocks[f].size() / cells;
    s.names.push_back(blocks[f].name);
    std::size_t k = 0;
    for (const Index g : blocks[f].indices()) {
      s.rest.push_back(static_cast<std::size_t>(g));
      s.cell_of.push_back(static_cast<int>(k / components));
      s.field_of.push_back(static_cast<int>(f) - 1);
      ++k;
    }
  }
  return s;
}

double worst_off_diagonal(const Dense& A, const std::vector<std::size_t>& idx) {
  double worst = 0.0;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    for (std::size_t j = 0; j < idx.size(); ++j) {
      if (i != j) worst = std::max(worst, std::abs(A.at(idx[i], idx[j])));
    }
  }
  return worst;
}

// S = C - A10 M^-1 A01, with M the diagonal first block. A10 and A01 are taken
// separately rather than as one transpose: a pinned unknown is a row the
// constraint set rewrote, so the assembled matrix is not symmetric even where
// the operator is, and S is where the symmetry has to REAPPEAR.
Dense condense(const Dense& A, const Split& s, const std::vector<double>& b,
               std::vector<double>* rhs = nullptr) {
  Dense S(s.rest.size(), s.rest.size());
  std::vector<double> inv(s.first.size());
  for (std::size_t k = 0; k < s.first.size(); ++k) inv[k] = 1.0 / A.at(s.first[k], s.first[k]);

  for (std::size_t i = 0; i < s.rest.size(); ++i) {
    for (std::size_t j = 0; j < s.rest.size(); ++j) S.at(i, j) = A.at(s.rest[i], s.rest[j]);
  }
  for (std::size_t k = 0; k < s.first.size(); ++k) {
    const std::size_t g = s.first[k];
    for (std::size_t i = 0; i < s.rest.size(); ++i) {
      const double left = A.at(s.rest[i], g);
      if (left == 0.0) continue;
      for (std::size_t j = 0; j < s.rest.size(); ++j) {
        S.at(i, j) -= left * inv[k] * A.at(g, s.rest[j]);
      }
    }
  }
  if (rhs != nullptr) {
    rhs->assign(s.rest.size(), 0.0);
    for (std::size_t i = 0; i < s.rest.size(); ++i) {
      (*rhs)[i] = b[s.rest[i]];
      for (std::size_t k = 0; k < s.first.size(); ++k) {
        (*rhs)[i] -= A.at(s.rest[i], s.first[k]) * inv[k] * b[s.first[k]];
      }
    }
  }
  return S;
}

double asymmetry(const Dense& S) {
  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < S.rows(); ++i) {
    for (std::size_t j = 0; j < S.rows(); ++j) {
      worst = std::max(worst, std::abs(S.at(i, j) - S.at(j, i)));
      scale = std::max(scale, std::abs(S.at(i, j)));
    }
  }
  return scale > 0.0 ? worst / scale : worst;
}

// cells sharing a facet, from the cell-facet incidence and its inverse
std::vector<std::vector<char>> facet_neighbours(const exokal::Mesh& mesh, int dim) {
  const graphos::Complex& c = mesh.topology();
  const auto cells = static_cast<std::size_t>(c.count(dim));
  const auto facets = static_cast<std::size_t>(c.count(dim - 1));
  const graphos::BoundaryOperator& d = c.boundary(dim);
  std::vector<std::vector<int>> on_facet(facets);
  for (std::size_t e = 0; e < cells; ++e) {
    for (Index k = d.offsets[e]; k < d.offsets[e + 1]; ++k) {
      on_facet[static_cast<std::size_t>(d.indices[static_cast<std::size_t>(k)])].push_back(
          static_cast<int>(e));
    }
  }
  std::vector<std::vector<char>> near(cells, std::vector<char>(cells, 0));
  for (std::size_t e = 0; e < cells; ++e) near[e][e] = 1;
  for (const std::vector<int>& side : on_facet) {
    for (const int a : side) {
      for (const int b : side) near[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = 1;
    }
  }
  return near;
}

// every entry of S couples two cells that share a facet -- Definition 3.1 of
// Nordbotten & Keilegavlen, which is what "two-point" means
std::size_t entries_beyond_the_neighbours(const Dense& S, const Split& s,
                                          const std::vector<std::vector<char>>& near,
                                          double tol) {
  std::size_t bad = 0;
  for (std::size_t i = 0; i < S.rows(); ++i) {
    for (std::size_t j = 0; j < S.rows(); ++j) {
      if (std::abs(S.at(i, j)) <= tol) continue;
      const auto a = static_cast<std::size_t>(s.cell_of[i]);
      const auto b = static_cast<std::size_t>(s.cell_of[j]);
      if (near[a][b] == 0) ++bad;
    }
  }
  return bad;
}

// ---- the two models, on a handful of cells ---------------------------------

exokal::Mesh box_of(int n, int dim, Family family) {
  return mimetika::mesh::box({n, n, dim == 3 ? n : 1}, dim, family);
}

// THE TETRAHEDRA COME FROM THE ANNULUS, NOT THE BOX. box(simplex) is the
// Kuhn subdivision -- six congruent tetrahedra a cube, every cell a translate
// of every other -- and that pattern is degenerate for diagonal_tpsa: it
// carries exactly one spurious rotation mode per interior CUBE face, which the
// last test in this file measures. Tetrahedra as such are not the problem, and
// a mesh whose cells differ from one another shows it.
struct Case {
  const char* name;
  int dim;
  exokal::Mesh mesh;
  // QUASI-DEFINITE WHERE THE PAIRING HAS SLACK. A cell carries d unknowns per
  // facet against the d + d(d-1)/2 rows the divergence and the weak symmetry
  // impose: 4 against 3 on a quadrilateral, 9 against 6 on a hexahedron, 7.5
  // against 6 on a prism -- and exactly 6 against 6 on a TETRAHEDRON. At that
  // ratio the displacement block and the rotation block stay positive definite
  // separately and their union does not, so the matrix is symmetric and
  // invertible but not quasi-definite: MINRES rather than an LDL^T.
  bool quasi_definite;
};

std::vector<Case> cases() {
  std::vector<Case> out;
  out.push_back({"2D cartesian", 2, box_of(3, 2, Family::cartesian), true});
  out.push_back({"2D simplex", 2, box_of(3, 2, Family::simplex), true});
  out.push_back({"3D cartesian", 3, box_of(3, 3, Family::cartesian), true});
  out.push_back({"3D prism", 3, box_of(3, 3, Family::prism), true});
  out.push_back(
      {"3D simplex", 3, mimetika::mesh::annulus(4, 2, 3, Family::simplex, 1.0, 3.0, 1.0), false});
  return out;
}

// p prescribed on the whole boundary: natural data, nothing pinned
void build_flow(SinglePhaseModel& prob, const exokal::Mesh& m, int dim) {
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    const auto x = exokal::centroid(m, dim - 1, f);
    prob.flow().emplace<mimetika::PressureBC>(std::vector<Index>{f}, x[0] + 2.0 * x[1]);
  }
  prob.build();
}

// u = x on the whole boundary, as an affine datum: also natural
void build_elasticity(CauchyElasticityModel& prob, const exokal::Mesh& m, int dim) {
  std::array<double, 9> grad{};
  for (int k = 0; k < dim; ++k) grad[static_cast<std::size_t>(k * 3 + k)] = 1.0;
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    const auto x = exokal::centroid(m, dim, mimetika::cofacet_of(m, dim, f));
    prob.prescribe_displacement({f}, {x[0], x[1], dim == 3 ? x[2] : 0.0}, grad);
  }
  prob.build();
}

}  // namespace

// ---- the premise -----------------------------------------------------------

// The star is what makes the block diagonal, so this is a statement about the
// products and not about the models: a de Rham or stabilized star couples the
// facets of a cell, and no arrangement of the assembly makes it diagonal.
MIMETIKA_TEST(only_the_diagonal_products_leave_a_diagonal_first_block) {
  const exokal::Mesh m = box_of(3, 3, Family::cartesian);

  for (const auto [how, name, diagonal] :
       {std::tuple{Flux::diagonal_tpfa, "diagonal_tpfa", true},
        std::tuple{Flux::stabilized_rt, "stabilized_rt", false},
        std::tuple{Flux::derham_bdm, "derham_bdm", false}}) {
    SinglePhaseModel prob(m, 3, 1.0, how);
    build_flow(prob, m, 3);
    const double worst = worst_off_diagonal(dense_of(prob.system()), split_of(prob).first);
    std::printf("  flow  %-14s off-diagonal in the flux block %.2e\n", name, worst);
    CHECK((worst == 0.0) == diagonal);
  }

  for (const auto [how, name, form, diagonal] :
       {std::tuple{Stress::diagonal_tpsa, "diagonal_tpsa", Formulation::weak_symmetry_total, true},
        std::tuple{Stress::stabilized_bdm, "stabilized_bdm", Formulation::weak_symmetry_total,
                   false},
        std::tuple{Stress::derham_bdm, "derham_bdm", Formulation::weak_symmetry, false}}) {
    CauchyElasticityModel prob(m, 3, ElasticMaterial{kMu, kLam}, how, form);
    build_elasticity(prob, m, 3);
    const double worst = worst_off_diagonal(dense_of(prob.system()), split_of(prob).first);
    std::printf("  solid %-14s off-diagonal in the stress block %.2e\n", name, worst);
    CHECK((worst == 0.0) == diagonal);
  }
}

// ---- what the condensation leaves ------------------------------------------

MIMETIKA_TEST(eliminating_the_flux_leaves_an_spd_two_point_pressure_system) {
  for (const Case& c : cases()) {
    {
      const exokal::Mesh& m = c.mesh;
      const int dim = c.dim;
      SinglePhaseModel prob(m, dim, 1.0, Flux::diagonal_tpfa);
      build_flow(prob, m, dim);
      const Split s = split_of(prob);
      const Dense S = condense(dense_of(prob.system()), s, prob.rhs());

      Dense L = S;
      const bool definite = cholesky(L);
      const std::size_t stray =
          entries_beyond_the_neighbours(S, s, facet_neighbours(m, dim), 1e-12);
      std::printf("  %-13s %3zu cells -> %4zu x %4zu   asymmetry %.1e  SPD %d  "
                  "beyond the two cells %zu\n",
                  c.name, static_cast<std::size_t>(prob.n_cells()),
                  S.rows(), S.rows(), asymmetry(S), definite ? 1 : 0, stray);
      CHECK(asymmetry(S) < 1e-12);
      CHECK(definite);
      CHECK(stray == 0);
    }
  }
}

// QUASI-DEFINITE EVERYWHERE BUT ON TETRAHEDRA, and the exception is the space
// rather than the condensation. diagonal_tpsa carries d unknowns a facet, so a
// cell holds d x (facets/cell) of them against the d + d(d-1)/2 rows the
// divergence and the weak symmetry impose: 4 against 3 on a quadrilateral, 9
// against 6 on a hexahedron -- and exactly 6 against 6 on a TETRAHEDRON, where
// the pairing has no slack at all. There the displacement block and the
// rotation block are each positive definite and their union is not, which is
// the same marginality that costs [Dv; As] its row rank once a traction is
// imposed. The system remains symmetric and nonsingular -- the solve below
// says so on every cell type -- so MINRES is what applies generally, and the
// LDL^T a quasi-definite matrix admits is available on the rest.
MIMETIKA_TEST(eliminating_the_stress_leaves_displacement_rotation_and_pressure) {
  for (const Case& c : cases()) {
    {
      const exokal::Mesh& m = c.mesh;
      const int dim = c.dim;
      CauchyElasticityModel prob(m, dim, ElasticMaterial{kMu, kLam}, Stress::diagonal_tpsa,
                                 Formulation::weak_symmetry_total);
      build_elasticity(prob, m, dim);
      const Split s = split_of(prob);
      const Dense S = condense(dense_of(prob.system()), s, prob.rhs());

      // QUASI-DEFINITE: the kinematic half positive, the pressure negative.
      const auto is_pressure = [&](std::size_t i) {
        return s.names[static_cast<std::size_t>(s.field_of[i])][0] == 'p';
      };
      Dense kinematic = diagonal_block(S, [&](std::size_t i) { return !is_pressure(i); });
      Dense pressure = negated(diagonal_block(S, is_pressure));
      const bool quasi = cholesky(kinematic) && cholesky(pressure);

      // and the finite volume form: the pressure EQUATION negated is coercive
      Dense flipped = S;
      for (std::size_t i = 0; i < S.rows(); ++i) {
        if (!is_pressure(i)) continue;
        for (std::size_t j = 0; j < S.rows(); ++j) flipped.at(i, j) = -flipped.at(i, j);
      }
      Dense coercive = symmetric_part(flipped);
      const bool positive = cholesky(coercive);

      Dense spd = S;
      const bool definite = cholesky(spd);
      const std::size_t stray =
          entries_beyond_the_neighbours(S, s, facet_neighbours(m, dim), 1e-12);
      // d + d(d-1)/2 + 1 unknowns a cell: displacement, rotation, pressure
      const std::size_t per_cell = static_cast<std::size_t>(dim + dim * (dim - 1) / 2 + 1);
      std::printf("  %-13s %3zu cells -> %4zu x %4zu (%zu a cell)   asymmetry %.1e  "
                  "quasi-definite %d  coercive %d  definite %d  beyond the two cells %zu\n",
                  c.name, static_cast<std::size_t>(prob.n_cells()),
                  S.rows(), S.rows(), per_cell, asymmetry(S), quasi ? 1 : 0, positive ? 1 : 0,
                  definite ? 1 : 0, stray);
      CHECK(S.rows() == static_cast<std::size_t>(prob.n_cells()) * per_cell);
      CHECK(asymmetry(S) < 1e-12);
      CHECK(quasi == c.quasi_definite);
      CHECK(positive == c.quasi_definite);
      CHECK(!definite);  // it is NOT SPD, and a solver told otherwise will fail
      CHECK(stray == 0);
    }
  }
}

// ---- and it is the same system ---------------------------------------------
//
// Definite and small is worth nothing if it is a different problem. The
// condensed solve is compared against the saddle point's own, unknown by
// unknown, and the eliminated field is recovered and compared too -- x0 =
// M^-1 (b0 - A01 y) is the whole cost of getting the flux or the stress back.
MIMETIKA_TEST(the_condensed_solve_is_the_saddle_point_solve) {
  // The condensed answer against the saddle point's own, unknown by unknown,
  // and the eliminated field recovered too: x0 = M^-1 (b0 - A01 y) is the whole
  // cost of getting the flux or the stress back. `singular` marks the case
  // where there is nothing to agree with -- see the header -- and the
  // conditioning is asserted there instead.
  const auto compare = [&](const SparseSystem& A, const std::vector<double>& b, const Split& s,
                           const std::string& what, bool singular) {
    const Dense full = dense_of(A);
    std::vector<double> rhs;
    const Dense S = condense(full, s, b, &rhs);
    const std::vector<double> y = lu_solve(S, rhs);
    const double pivots = pivot_ratio;

    mimetika::solver::PetscSolver petsc;
    std::vector<double> x;
    const auto rep = petsc.solve(A, b, x);
    CHECK(rep.converged);

    // the residual of BOTH answers in the condensed system, which is what says
    // whether a disagreement is the condensation or the solve
    double mine = 0.0, theirs = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < s.rest.size(); ++i) {
      double a = -rhs[i], c = -rhs[i];
      for (std::size_t j = 0; j < s.rest.size(); ++j) {
        a += S.at(i, j) * y[j];
        c += S.at(i, j) * x[s.rest[j]];
      }
      mine = std::max(mine, std::abs(a));
      theirs = std::max(theirs, std::abs(c));
      scale = std::max(scale, std::abs(rhs[i]));
    }

    double worst = 0.0, size = 0.0, worst_first = 0.0;
    for (std::size_t i = 0; i < s.rest.size(); ++i) {
      worst = std::max(worst, std::abs(y[i] - x[s.rest[i]]));
      size = std::max(size, std::abs(x[s.rest[i]]));
    }
    for (std::size_t k = 0; k < s.first.size(); ++k) {
      const std::size_t g = s.first[k];
      double acc = b[g];
      for (std::size_t j = 0; j < s.rest.size(); ++j) acc -= full.at(g, s.rest[j]) * y[j];
      worst_first = std::max(worst_first, std::abs(acc / full.at(g, g) - x[g]));
    }
    std::printf("  %-20s pivots %.1e   |S y - b| %.1e   |S x - b| %.1e   agreement %.1e / %.1e\n",
                what.c_str(), pivots, mine / scale, theirs / scale, worst / std::max(1.0, size),
                worst_first);

    CHECK((pivots < 1e-12) == singular);
    if (singular) return;
    CHECK(mine < 1e-9 * scale);
    CHECK(worst < 1e-9 * std::max(1.0, size));
    CHECK(worst_first < 1e-9 * std::max(1.0, size));
  };

  for (const Case& c : cases()) {
    {
      const exokal::Mesh& m = c.mesh;
      const int dim = c.dim;
      const std::string where = c.name;

      SinglePhaseModel flow(m, dim, 1.0, Flux::diagonal_tpfa);
      build_flow(flow, m, dim);
      compare(flow.system(), flow.rhs(), split_of(flow), where + " tpfa", false);

      CauchyElasticityModel solid(m, dim, ElasticMaterial{kMu, kLam}, Stress::diagonal_tpsa,
                                  Formulation::weak_symmetry_total);
      build_elasticity(solid, m, dim);
      // the tetrahedron is the ratio at which the rotation is not determined
      compare(solid.system(), solid.rhs(), split_of(solid), where + " tpsa", false);
    }
  }
}


// ---- one mesh the pattern of which diagonal_tpsa cannot carry ---------------
//
// box(simplex) is the Kuhn subdivision: six congruent tetrahedra to a cube,
// every cell a translate or reflection of every other. TETRAHEDRA ARE NOT THE
// PROBLEM -- the annulus tetrahedra above condense to an invertible system --
// but that PATTERN is, and by an exact count: the condensed operator loses one
// dimension per interior CUBE face, 3 n^2 (n - 1) of them, which is 12, 54 and
// 144 at n = 2, 3 and 4. The modes are rotation, with the stress and the
// pressure identically zero.
//
// The condensation is what makes this visible at all. Handed the saddle point,
// the direct solver reports CONVERGED and returns 1e16; handed the condensed
// operator, the elimination runs out of pivots and says so. That is the case
// for specializing the solver rather than for trusting a factorization.
//
// This test is expected to FAIL the day the rotation is closed -- TPSA's alpha
// term, which exokal does not carry -- and that failure is the notification.
MIMETIKA_TEST(the_kuhn_tetrahedra_lose_one_rotation_per_interior_cube_face) {
  for (const int n : {2, 3}) {
    const exokal::Mesh m = box_of(n, 3, Family::simplex);
    CauchyElasticityModel prob(m, 3, ElasticMaterial{kMu, kLam}, Stress::diagonal_tpsa,
                               Formulation::weak_symmetry_total);
    build_elasticity(prob, m, 3);
    const Split s = split_of(prob);
    std::vector<double> rhs;
    const Dense S = condense(dense_of(prob.system()), s, prob.rhs(), &rhs);
    lu_solve(S, rhs);
    const std::size_t interior_cube_faces = static_cast<std::size_t>(3 * n * n * (n - 1));
    std::printf("  %d^3 cubes, %3zu tetrahedra -> %4zu x %4zu   pivot ratio %.1e   "
                "pivots lost %2zu   interior cube faces %2zu\n",
                n, static_cast<std::size_t>(prob.n_cells()), S.rows(), S.rows(), pivot_ratio,
                tiny_pivots, interior_cube_faces);
    CHECK(pivot_ratio < 1e-12);
    CHECK(tiny_pivots == interior_cube_faces);
  }
}

// ---- and the solver does it ------------------------------------------------
//
// Everything above is arithmetic done in the test. This is the solver's own
// path: told which unknowns may be divided out, PetscSolver eliminates them,
// solves S, and puts the eliminated field back -- and the answer has to be the
// one the saddle point gives, on every unknown INCLUDING the eliminated ones.
//
// Naming the field is a permission, not an instruction. The same call on a de
// Rham or stabilized product must leave the saddle point alone, because their
// star couples a cell's facets and the block is not diagonal; the report says
// which happened.
MIMETIKA_TEST(the_solver_condenses_when_it_is_allowed_to_and_only_then) {
  const exokal::Mesh m = box_of(3, 3, Family::cartesian);

  for (const auto [how, name, diagonal] :
       {std::tuple{Flux::diagonal_tpfa, "diagonal_tpfa", true},
        std::tuple{Flux::stabilized_rt, "stabilized_rt", false}}) {
    SinglePhaseModel prob(m, 3, 1.0, how);
    build_flow(prob, m, 3);
    const std::vector<int> first = mimetika::solver::first_field_dofs(prob.simulation().epoch());

    mimetika::solver::PetscSolver saddle;
    std::vector<double> plain;
    const auto a = saddle.solve(prob.system(), prob.rhs(), plain);

    mimetika::solver::PetscSolver condensing;
    condensing.set_condensable(first);
    std::vector<double> reduced;
    const auto b = condensing.solve(prob.system(), prob.rhs(), reduced);

    double worst = 0.0;
    for (std::size_t i = 0; i < plain.size(); ++i) {
      worst = std::max(worst, std::abs(plain[i] - reduced[i]));
    }
    std::printf("  flow  %-14s condensed %d  %zu of %zu unknowns solved  |difference| %.1e\n",
                name, b.condensed ? 1 : 0, b.condensed ? b.condensed_dofs : prob.simulation().n_dofs(),
                prob.simulation().n_dofs(), worst);
    CHECK(a.converged && b.converged);
    CHECK(b.condensed == diagonal);
    CHECK(a.condensed == false);  // never without the permission
    CHECK(worst < 1e-9);
    if (diagonal) CHECK(b.condensed_dofs == static_cast<std::size_t>(prob.n_cells()));
  }

  // the stress, four fields, and the same statement
  CauchyElasticityModel solid(m, 3, ElasticMaterial{kMu, kLam}, Stress::diagonal_tpsa,
                              Formulation::weak_symmetry_total);
  build_elasticity(solid, m, 3);
  mimetika::solver::PetscSolver saddle;
  std::vector<double> plain;
  const auto a = saddle.solve(solid.system(), solid.rhs(), plain);

  mimetika::solver::PetscSolver condensing;
  condensing.set_condensable(mimetika::solver::first_field_dofs(solid.simulation().epoch()));
  std::vector<double> reduced;
  const auto b = condensing.solve(solid.system(), solid.rhs(), reduced);

  double worst = 0.0, stress = 0.0;
  const auto sigma = static_cast<std::size_t>(
      mimetika::solver::field_blocks(solid.simulation().epoch())[0].size());
  for (std::size_t i = 0; i < plain.size(); ++i) {
    const double d = std::abs(plain[i] - reduced[i]);
    worst = std::max(worst, d);
    if (i < sigma) stress = std::max(stress, d);
  }
  std::printf("  solid %-14s condensed %d  %zu of %zu unknowns solved  |difference| %.1e "
              "(%.1e on the recovered stress)\n",
              "diagonal_tpsa", b.condensed ? 1 : 0, b.condensed_dofs,
              solid.simulation().n_dofs(), worst, stress);
  CHECK(a.converged && b.converged);
  CHECK(b.condensed);
  CHECK(worst < 1e-9);

  // and the option turns it off, permission or not
  mimetika::solver::SolverOptions off;
  off.condense = false;
  mimetika::solver::PetscSolver refusing(off);
  refusing.set_condensable(mimetika::solver::first_field_dofs(solid.simulation().epoch()));
  std::vector<double> again;
  const auto c = refusing.solve(solid.system(), solid.rhs(), again);
  CHECK(!c.condensed);
}

MIMETIKA_TEST_MAIN()
