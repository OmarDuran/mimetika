#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/matrix_free.hpp"
#include "mimetika/linear_solver/minres.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"

// The matrix-free mechanics solve: Pazner, Kolev & Vassilevski carried from
// H(div) flow to mixed elasticity with weak symmetry.
//
// WHAT TRANSFERS, AND WHY IT IS NOT OBVIOUS. Their saddle point is
//
//     [ M   B^T ]        B = D, the divergence,
//     [ B   0   ]
//
// and the whole construction is B = diag(diag(M), S~) with S~ = D diag(M)^-1
// D^T. Mixed elasticity does not look like that: weak symmetry carries THREE
// fields, because the space holds no symmetry constraint and a rotation
// multiplier gamma pairs sigma against the rigid rotations.
//
// It is that shape all the same. Negating every non-stress row gives exactly
// their (10) -- measured below, to round-off, on both cell families -- with
//
//     B = [ D ]     the momentum balance over the divergence,
//         [ A ]     the symmetry over the asymmetry,
//
// and an empty (2, 2) block. The second block is composite; nothing else moved.
//
// THE THEORY NEEDS NO EXTENSION. Their Propositions 1, 2, 3 and 5 assume only
// that the (1, 1) block is SPD -- section 2.2 says so -- and make NO assumption
// about B. The one hypothesis that matters is M spectrally equivalent to its
// own diagonal, and the equivalence of S~ to the exact S follows from it for
// ANY B by operator antitonicity:
//
//     c diag(M) <= M <= C diag(M)
//       =>  (1/C) B diag(M)^-1 B^T  <=  B M^-1 B^T  <=  (1/c) B diag(M)^-1 B^T.
//
// At lowest order the hypothesis is free: M is assembled from cell-local blocks
// and a facet is shared by two cells, so the equivalence constant is a bound in
// the CELL SHAPE and not in h (tests/solver/test_matrix_free.cpp). Measured
// here: kappa(B^-1 A) is 9.14 then 9.31 over 3^3 and the constant it tracks,
// cond(M/diag M), is 8.86 then 8.80.
//
// WHAT DOES NOT TRANSFER IS PROPOSITION 4, and it costs nothing. S~ is not an
// M-matrix here -- 540 positive off-diagonals on 27 cartesian cells, row sums
// that do not vanish -- because u and gamma are cell VECTORS, so S~ is a system
// and not a scalar equation. That proposition is what licenses one algebraic
// multigrid cycle, and it is the near-null space it is really about: an
// M-matrix with zero row sums has the constants in its kernel, which is what
// classical AMG's coarsening and interpolation reproduce.
//
// Measured, CG on S~ to 1e-8 over cartesian 2^3 to 8^3, a sixty-fourfold growth
// in the cells: 4, 5, 6, 8, 8 iterations with BoomerAMG's defaults, and 4, 5,
// 6, 7, 7 told the unknowns come six to a cell. The property was sufficient and
// not necessary. The reason looks structural: A diag(M)^-1 A^T is a MASS-like
// zeroth-order block, so S~ has no near-kernel for a coarse space to have to
// represent -- its smallest eigenvalue grows under refinement rather than
// shrinking. The rotation multiplier, which is the apparent complication, is
// what makes the Schur complement easy.

using mimetika::CauchyMechanicsModel;
using mimetika::mesh::Family;
using mimetika::solver::field_blocks;
using mimetika::solver::minres;
using mimetika::solver::MinresOptions;
using mimetika::solver::SparseSystem;
using Realization = CauchyMechanicsModel::Realization;
using Formulation = CauchyMechanicsModel::Formulation;

namespace {

struct Problem {
  std::unique_ptr<exokal::Mesh> mesh;
  std::unique_ptr<CauchyMechanicsModel> model;
  std::vector<graphos::Index> stress, rest;
  std::vector<double> sign;  // +1 on the stress rows, -1 on the rest
  std::size_t pinned{0};
};

// A column: a displacement prescribed over most of the boundary, which is
// NATURAL in this form, and optionally a traction on the top face, which is the
// ESSENTIAL one and therefore pins rows. The mechanics roles are the flow roles
// swapped, so a mechanics run carries strong conditions where a flow run
// usually does not -- the pinned columns are the ordinary case here.
Problem build(int nb, Family family, Realization how, bool traction, bool tangent = true) {
  const int dim = 3;
  const std::array<double, 9> gradient{1.0, 0, 0, 0, -0.3, 0, 0, 0, -0.3};
  Problem p;
  p.mesh = std::make_unique<exokal::Mesh>(mimetika::mesh::box({nb, nb, nb}, dim, family));
  p.model = std::make_unique<CauchyMechanicsModel>(*p.mesh, dim, mimetika::ElasticMaterial{1.0, 1.0},
                                                   how, Formulation::weak_symmetry);
  for (const graphos::Index f : mimetika::boundary_facets(p.mesh->topology(), dim)) {
    const auto x = exokal::centroid(*p.mesh, dim - 1, f);
    if (traction && x[2] > 1.0 - 1e-12) {
      p.model->mechanics().emplace<mimetika::TractionBC>(
          std::vector<graphos::Index>{f}, std::array<double, 9>{0, 0, 0, 0, 0, 0, 0, 0, -0.5});
    } else {
      p.model->prescribe_displacement(
          {f}, {gradient[0] * x[0], gradient[4] * x[1], gradient[8] * x[2]}, gradient);
    }
  }
  p.model->build(tangent);
  for (const auto& b : field_blocks(p.model->simulation().epoch())) {
    auto& into = b.name.rfind("s", 0) == 0 ? p.stress : p.rest;
    for (const auto i : b.indices()) into.push_back(i);
  }
  const std::size_t n = p.model->simulation().n_dofs();
  p.sign.assign(n, 1.0);
  for (const graphos::Index i : p.rest) p.sign[static_cast<std::size_t>(i)] = -1.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (p.model->simulation().constraints().pinned(i)) ++p.pinned;
  }
  return p;
}

const std::vector<std::pair<std::string, Realization>> kProducts{
    {"derham_bdm", Realization::derham_bdm},
    {"stabilized_bdm", Realization::stabilized_bdm},
};

}  // namespace

// ---- the shape ------------------------------------------------------------
MIMETIKA_TEST(negating_the_non_stress_rows_makes_the_operator_symmetric) {
  for (const auto& [name, how] : kProducts) {
    for (const Family family : {Family::cartesian, Family::simplex}) {
      const Problem p = build(2, family, how, false);
      const auto& sim = p.model->simulation();
      const std::size_t n = sim.n_dofs();
      CHECK(p.pinned == 0);  // a displacement is natural: the operator, alone

      std::vector<double> e(n, 0.0), y, t;
      double asym = 0.0, scale = 0.0;
      std::vector<std::vector<double>> col(n);
      for (std::size_t j = 0; j < n; ++j) {
        e.assign(n, 0.0);
        e[j] = 1.0;
        sim.apply(e, t);
        col[j].resize(n);
        for (std::size_t i = 0; i < n; ++i) col[j][i] = p.sign[i] * t[i];
      }
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          asym = std::max(asym, std::fabs(col[j][i] - col[i][j]));
          scale = std::max(scale, std::fabs(col[j][i]));
        }
      }
      std::printf("  %-15s %-8s asymmetry %.2e of %.2e\n", name.c_str(),
                  family == Family::cartesian ? "cart" : "simplex", asym, scale);
      CHECK(asym <= 1e-12 * scale);
    }
  }
}

// ---- the solve, and the refinement ----------------------------------------
//
// CONVERGED IS NOT CORRECT, so the answer is checked against a direct solve of
// the system as assembled -- which the sign flip and the elimination both leave
// unchanged, being row operations on a system whose solution they preserve.
MIMETIKA_TEST(the_block_preconditioner_is_h_robust_on_mixed_elasticity) {
  for (const auto& [name, how] : kProducts) {
    for (const Family family : {Family::cartesian, Family::simplex}) {
      for (const bool traction : {false, true}) {
        std::vector<int> its;
        for (const int nb : {2, 3}) {
          const Problem p = build(nb, family, how, traction);
          const auto& sim = p.model->simulation();
          const SparseSystem& A = p.model->system();
          const std::size_t n = A.n;
          CHECK(traction == (p.pinned > 0));
          // every strong form in this family is a single pin, so the columns
          // can be eliminated and the operator handed to MINRES is symmetric
          CHECK(sim.pins_only());

          // S~, off the same kernels the operator applies. SchurSink carries a
          // LIST per stress dof and assumes no arity, so B = [D; A] with its
          // twelve entries a dof drops in where the flux had two.
          const std::vector<int> in_stress = mimetika::solver::slot_of(n, p.stress);
          mimetika::solver::SchurSink sink(in_stress, mimetika::solver::slot_of(n, p.rest),
                                           p.stress.size(), sim.constraints().mask());
          sim.assemble_into(sink);
          const SparseSystem S = mimetika::solver::schur_of(sink, p.rest.size());

          mimetika::solver::SolverOptions inner;
          inner.method = "preonly";  // one V-cycle: Prop. 4 fails, the cycle does not
          inner.preconditioner = "hypre";
          inner.rtol = 0.0;
          inner.max_iterations = 1;
          inner.condense = false;
          mimetika::solver::PetscSolver schur(inner);

          std::vector<double> b = p.model->rhs();
          sim.fold_pinned_columns(b);
          for (std::size_t i = 0; i < n; ++i) b[i] *= p.sign[i];

          std::vector<double> t, vp, yp;
          auto apply_a = [&](const std::vector<double>& v, std::vector<double>& y) {
            sim.apply_symmetric(v, t);
            y.resize(t.size());
            for (std::size_t i = 0; i < t.size(); ++i) y[i] = p.sign[i] * t[i];
          };
          auto apply_b = [&](const std::vector<double>& v, std::vector<double>& y) {
            y.assign(v.size(), 0.0);
            for (std::size_t i = 0; i < v.size(); ++i) {
              if (in_stress[i] >= 0) {
                y[i] = v[i] / sink.mass[static_cast<std::size_t>(in_stress[i])];
              }
            }
            vp.assign(p.rest.size(), 0.0);
            for (std::size_t i = 0; i < p.rest.size(); ++i) {
              vp[i] = v[static_cast<std::size_t>(p.rest[i])];
            }
            yp.assign(p.rest.size(), 0.0);
            schur.solve(S, vp, yp);
            for (std::size_t i = 0; i < p.rest.size(); ++i) {
              y[static_cast<std::size_t>(p.rest[i])] = yp[i];
            }
          };

          MinresOptions o;
          o.rtol = 1e-10;
          o.max_iterations = 600;
          std::vector<double> x;
          const auto rep = minres(apply_a, apply_b, b, x, o);

          mimetika::solver::PetscSolver direct;
          std::vector<double> x_direct;
          const auto d = direct.solve(A, p.model->rhs(), x_direct);
          if (!d.converged) throw std::runtime_error("the direct reference failed: " + d.reason);
          double err = 0.0, scale = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            err = std::max(err, std::fabs(x[i] - x_direct[i]));
            scale = std::max(scale, std::fabs(x_direct[i]));
          }
          std::printf("  %-15s %-8s %s %2d^3 n=%6zu pinned %4zu  %4d its  err %.2e\n",
                      name.c_str(), family == Family::cartesian ? "cart" : "simplex",
                      traction ? "traction   " : "displacement", nb, n, p.pinned, rep.iterations,
                      err / scale);
          CHECK(rep.converged);
          CHECK(err <= 1e-7 * scale);
          its.push_back(rep.iterations);
        }
        // WHAT IS FLAT IS THE CONDITIONING. As in the flow test, a bounded
        // kappa bounds the convergence FACTOR and not the count at a given
        // tolerance; the count drifts as the spectrum fills in. Measured over
        // cartesian 2^3 to 6^3 -- a twentyfold growth in the unknowns -- the
        // count goes 69, 80, 86, 89 and on simplices 235, 243, 243, 242. What
        // is asserted is that the drift over the pair run here stays far below
        // the growth in the problem.
        CHECK(its.back() <= 2 * its.front());
      }
    }
  }
}

// ---- and the tangent is never formed --------------------------------------
MIMETIKA_TEST(build_without_a_tangent_leaves_the_system_empty) {
  const Problem p = build(2, Family::cartesian, Realization::derham_bdm, true, false);
  CHECK(p.model->system().nnz() == 0);
  CHECK(p.model->system().n == 0);
  // the load is still there: it is the residual at the zero state, not the
  // tangent, and the strongly imposed rows still carry their own datum
  CHECK(p.model->rhs().size() == p.model->simulation().n_dofs());
  double worst = 0.0;
  for (const double v : p.model->rhs()) worst = std::max(worst, std::fabs(v));
  CHECK(worst > 0.0);
}

// ---- the scale is the same whichever path measured it ----------------------
//
// Constraints requires one scale on all three paths, or the residual, the
// tangent and the action impose three different equations. Which path measures
// it first is decided by the build: with a tangent the scale is read off the
// assembled triplets, without one the residual is reached first and a
// DiagonalSink probes for it. Those two must agree entry for entry -- and the
// load, which is the scale times the datum, is where a disagreement would show.
MIMETIKA_TEST(the_scale_does_not_depend_on_which_path_measured_it) {
  for (const Family family : {Family::cartesian, Family::simplex}) {
    const Problem tangent = build(2, family, Realization::derham_bdm, true, true);
    const Problem probed = build(2, family, Realization::derham_bdm, true, false);
    const auto& from_tangent = tangent.model->simulation().constraints();
    const auto& from_probe = probed.model->simulation().constraints();
    const std::size_t n = tangent.model->simulation().n_dofs();
    CHECK(tangent.pinned > 0);  // otherwise no scale is measured and this is vacuous
    CHECK(tangent.pinned == probed.pinned);

    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!from_tangent.pinned(i)) continue;
      worst = std::max(worst, std::fabs(from_tangent.scale_at(i) - from_probe.scale_at(i)));
      scale = std::max(scale, std::fabs(from_tangent.scale_at(i)));
    }
    double load = 0.0, load_scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      load = std::max(load, std::fabs(tangent.model->rhs()[i] - probed.model->rhs()[i]));
      load_scale = std::max(load_scale, std::fabs(tangent.model->rhs()[i]));
    }
    std::printf("  %-8s scale differs by %.2e of %.2e, load by %.2e of %.2e\n",
                family == Family::cartesian ? "cart" : "simplex", worst, scale, load, load_scale);
    CHECK(scale > 0.0);
    CHECK(worst <= 1e-14 * scale);
    CHECK(load <= 1e-14 * load_scale);
  }
}

MIMETIKA_TEST_MAIN()
