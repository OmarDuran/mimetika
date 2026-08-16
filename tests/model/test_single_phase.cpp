#include <cmath>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/single_phase_model.hpp"
#include "mimetika/solver/petsc.hpp"

// THE SINGLE-PHASE SOLVER, AGAINST TWO CLOSED FORMS, ON EVERY CELL TYPE.
//
//   column    p linear between the ends              the Laplace solution
//   annulus   p_a + (p_b - p_a) ln(r/a)/ln(b/a)      Dupuit
//
// Both are exact for the discretization -- the flux space reproduces the
// gradient of a linear field, and the Dupuit profile is the radial harmonic --
// so the comparison is against machine precision for the column and against
// the resolution of a curved boundary for the annulus.
//
// Six mesh configurations each -- two dimensions by three families -- against
// ALL THREE of exokal's flux inner products. The prism is the cell that matters
// most: neither a simplex nor a tensor-product cell, two triangles and three
// quadrilaterals, so a construction that handles it is polytopal rather than
// hex-and-tet with extra steps. In the plane a prism over an interval IS a
// quadrilateral, so that row coincides with cartesian and is reported rather
// than skipped.
//
// THE THREE PRODUCTS ARE DIFFERENT DISCRETIZATIONS, not a refinement ladder:
//
//   derham      d moments per facet. On a simplex it IS BDM_1 -- 3 edges x 2
//               in the plane, 4 facets x 3 in space -- unisolvent with no
//               enrichment; on a polytope the moments are completed with
//               div-free curl modes. Any cell, either dimension.
//   derham_rt   one flux per facet: RT_0, the minimal de Rham pair, whose
//               radial mode x - x_E is what lets div reach P_0 at all. d+1
//               modes against d+1 fluxes is the whole argument, so it is
//               SIMPLICES only -- triangle or tetrahedron -- and refuses the
//               rest rather than stabilizing them.
//   stabilized  one flux per facet on any polytope, in either dimension:
//               consistency plus a stabilization sized to trace(M1)/(d+2),
//               which is the scale at which it IS the conforming RT_0 element
//               on a simplex (exokal hodge.test_rt_equivalence, vs basix).
//
// What they share is CONSISTENCY -- every one of them reproduces a constant
// flux exactly -- and that is what the column asserts of all three. Where they
// differ is everything else, which is what the annulus measures.

using graphos::Index;
using mimetika::SinglePhaseModel;
using mimetika::mesh::Family;

namespace {

struct Outcome {
  double max_err{0.0};
  double rms_err{0.0};
  std::size_t cells{0};
  std::size_t dofs{0};
};

using Realization = mimetika::SinglePhaseModel::Realization;

Outcome run(const exokal::Mesh& m, int dim, const std::vector<Index>& high,
            const std::vector<Index>& low, const std::vector<Index>& sealed, double p_high,
            double p_low, const std::function<double(const exokal::Mesh::Point&)>& exact,
            Realization how = Realization::derham_bdm) {
  SinglePhaseModel prob(m, dim, 1.0, how);
  prob.flow().emplace<mimetika::NormalFluxBC>(sealed);
  prob.flow().emplace<mimetika::PressureBC>(high, p_high);
  prob.flow().emplace<mimetika::PressureBC>(low, p_low);
  prob.build();

  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(prob.system(), prob.rhs(), x);
  if (!rep.converged) throw std::runtime_error("single phase: " + rep.reason);
  prob.accept(x);

  Outcome out;
  out.cells = prob.n_cells();
  out.dofs = prob.simulation().n_dofs();
  double acc = 0.0;
  for (Index e = 0; e < static_cast<Index>(out.cells); ++e) {
    const double d = prob.cell_pressure(e) - exact(exokal::centroid(m, dim, e));
    out.max_err = std::max(out.max_err, std::abs(d));
    acc += d * d;
  }
  out.rms_err = std::sqrt(acc / static_cast<double>(out.cells));
  return out;
}

// the facet sets of a column: the two ends, and everything else sealed
Outcome column_case(int n, int dim, Family family, Realization how = Realization::derham_bdm) {
  const double h = 1.0, p_hi = 2.0, p_lo = 1.0;
  const exokal::Mesh m = mimetika::mesh::column(n, dim, family, h);
  const graphos::Complex& c = m.topology();
  const int axis = dim - 1;
  std::vector<Index> top, base, side;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    const double z = exokal::centroid(m, dim - 1, f)[static_cast<std::size_t>(axis)];
    if (std::abs(z - h) < 1e-9) {
      top.push_back(f);
    } else if (std::abs(z) < 1e-9) {
      base.push_back(f);
    } else {
      side.push_back(f);
    }
  }
  // p = p_lo at the base, p_hi at the top: linear in between
  return run(
      m, dim, top, base, side, p_hi, p_lo,
      [&](const exokal::Mesh::Point& x) {
        return p_lo + (p_hi - p_lo) * x[static_cast<std::size_t>(axis)] / h;
      },
      how);
}

// the facet sets of a quarter annulus: the two radii, symmetry planes sealed
Outcome annulus_case(int nr, int nt, int dim, Family family,
                     Realization how = Realization::derham_bdm) {
  const double a = 1.0, b = 10.0, hz = 1.0, p_a = 2.0, p_b = 1.0;
  const exokal::Mesh m = mimetika::mesh::annulus(nr, nt, dim, family, a, b, hz);
  const graphos::Complex& c = m.topology();
  const double rmid = std::sqrt(a * b);
  std::vector<Index> inner, outer, sealed;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    const auto x = exokal::centroid(m, dim - 1, f);
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const bool sym = std::abs(x[0]) < 1e-8 || std::abs(x[1]) < 1e-8 ||
                     (dim == 3 && (std::abs(x[2]) < 1e-8 || std::abs(x[2] - hz) < 1e-8));
    if (sym) {
      sealed.push_back(f);
    } else if (r < rmid) {
      inner.push_back(f);
    } else {
      outer.push_back(f);
    }
  }
  return run(
      m, dim, inner, outer, sealed, p_a, p_b,
      [&](const exokal::Mesh::Point& x) {
        const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        return p_a + (p_b - p_a) * std::log(r / a) / std::log(b / a);
      },
      how);
}

const Family kFamilies[] = {Family::cartesian, Family::simplex, Family::prism};

const Realization kProducts[] = {Realization::derham_bdm, Realization::derham_rt,
                                 Realization::stabilized_rt};

const char* product_name(Realization r) {
  switch (r) {
    case Realization::derham_bdm: return "derham_bdm";
    case Realization::derham_rt: return "derham_rt";
    case Realization::stabilized_rt: return "stabilized_rt";
  }
  return "?";
}

// WHAT EACH PRODUCT CLAIMS, stated once. A configuration outside a product's
// claim is not a failure and not a silent skip -- it is reported as refused,
// and a separate test checks that the refusal is an exception rather than a
// wrong answer.
bool supported(Realization r, int dim, Family f) {
  switch (r) {
    case Realization::derham_bdm: return true;
    // RT_0's unisolvence argument is d+1 modes against d+1 facets: a simplex,
    // in either dimension
    case Realization::derham_rt: return f == Family::simplex && !(dim == 3 && f == Family::prism);
    // the stabilized construction is polytopal in both dimensions
    case Realization::stabilized_rt: return true;
  }
  return false;
}

}  // namespace

// A LINEAR PRESSURE IS REPRODUCED EXACTLY. The flux space contains the constant
// fields, so the gradient of a linear pressure is represented with no error at
// all -- on any cell type, in any dimension. Anything above round-off here is a
// broken space, not a coarse mesh, and no amount of refinement would fix it.
MIMETIKA_TEST(the_column_reproduces_the_linear_solution_exactly) {
  for (const Realization r : kProducts) {
    for (const int dim : {2, 3}) {
      for (const Family f : kFamilies) {
        if (!supported(r, dim, f)) {
          std::printf("  column  %-10s %dD %-10s   not claimed\n", product_name(r), dim,
                      mimetika::mesh::name(f));
          continue;
        }
        const Outcome o = column_case(6, dim, f, r);
        std::printf("  column  %-10s %dD %-10s %5zu cells %7zu dofs   max %.2e   rms %.2e\n",
                    product_name(r), dim, mimetika::mesh::name(f), o.cells, o.dofs, o.max_err,
                    o.rms_err);
        CHECK(o.max_err < 1e-10);
      }
    }
  }
}

// AND THE DUPUIT PROFILE, which is not in the space: the radial harmonic is
// approximated, so the error is a resolution and not a defect. It must fall
// with refinement, which the second size checks.
MIMETIKA_TEST(the_annulus_reproduces_dupuit) {
  for (const Realization r : kProducts) {
    for (const int dim : {2, 3}) {
      for (const Family f : kFamilies) {
        if (!supported(r, dim, f)) continue;
        const Outcome coarse = annulus_case(8, 4, dim, f, r);
        const Outcome fine = annulus_case(16, 8, dim, f, r);
        std::printf(
            "  annulus %-10s %dD %-10s %5zu -> %5zu cells   max %.2e -> %.2e   rms %.2e -> %.2e\n",
            product_name(r), dim, mimetika::mesh::name(f), coarse.cells, fine.cells, coarse.max_err,
            fine.max_err, coarse.rms_err, fine.rms_err);
        CHECK(coarse.max_err < 5e-2);
        CHECK(fine.rms_err < coarse.rms_err);  // refinement helps, so it is resolution
      }
    }
  }
}

// ---- HOW THE THREE PRODUCTS DIFFER ----------------------------------------

// THE SPACES ARE NOT THE SAME SIZE, which is the concrete content of "different
// discretizations". d moments per facet against one: on the same tetrahedral
// column the de Rham/BDM space carries 330 unknowns where the two lowest-order
// products carry 134, and all three are exact on a linear pressure.
MIMETIKA_TEST(the_products_lay_out_different_spaces) {
  const Outcome bdm = column_case(6, 3, Family::simplex, Realization::derham_bdm);
  const Outcome rt = column_case(6, 3, Family::simplex, Realization::derham_rt);
  const Outcome mfd = column_case(6, 3, Family::simplex, Realization::stabilized_rt);
  std::printf("  3D simplex column   derham %zu   derham_rt %zu   stabilized %zu dofs\n", bdm.dofs,
              rt.dofs, mfd.dofs);
  CHECK(rt.dofs < bdm.dofs);
  CHECK(mfd.dofs == rt.dofs);  // both are one flux per facet
}

// RT AND THE STABILIZED PRODUCT RETURN THE SAME SOLVED FIELD ON A SIMPLEX --
// and they are NOT the same operator, which is the more interesting half.
//
// exokal measures the operators against basix (hodge.test_rt_equivalence):
// derham_rt IS the conforming RT_0 element, while the stabilized product
// reconstructs on the CONSTANTS ALONE -- three modes against four facet fluxes
// -- and carries a one-dimensional stabilization. Their matrices differ by
// about 3%.
//
// Yet the pressures agree to round-off, and that is worth pinning rather than
// explaining away: both spaces contain the constants, so both are consistent,
// and on this problem the stabilization does not reach the cell pressures. It
// is also the reason a model-level comparison must never be used to conclude
// two operators are the same -- this test would have said so, and it would have
// been wrong. That conclusion belongs to the exokal test, against a conforming
// element.
MIMETIKA_TEST(rt_and_the_stabilized_product_coincide_on_a_simplex) {
  for (const int nr : {8, 16}) {
    const Outcome rt = annulus_case(nr, nr / 2, 3, Family::simplex, Realization::derham_rt);
    const Outcome mfd = annulus_case(nr, nr / 2, 3, Family::simplex, Realization::stabilized_rt);
    std::printf("  annulus %4zu cells   RT max %.6e   MFD max %.6e   |diff| %.2e\n", rt.cells,
                rt.max_err, mfd.max_err, std::abs(rt.max_err - mfd.max_err));
    CHECK(std::abs(rt.max_err - mfd.max_err) < 1e-12);
    CHECK(std::abs(rt.rms_err - mfd.rms_err) < 1e-12);
  }
}

// EVERY UNCLAIMED CONFIGURATION RAISES. A product that is not unisolvent on a
// cell must say so: RT_0's whole argument is four modes against four facet
// fluxes, so a hexahedron is a different case and not a coarser one, and the
// stabilized construction is written for volumes. Refusing is the correct
// behaviour, and a silent wrong answer is the one thing that is not.
MIMETIKA_TEST(every_product_refuses_what_it_does_not_claim) {
  for (const Realization r : kProducts) {
    for (const int dim : {2, 3}) {
      for (const Family f : kFamilies) {
        if (supported(r, dim, f)) continue;
        bool refused = false;
        try {
          column_case(2, dim, f, r);
        } catch (const std::exception&) {
          refused = true;
        }
        std::printf("  %-10s %dD %-10s  %s\n", product_name(r), dim, mimetika::mesh::name(f),
                    refused ? "refused" : "ACCEPTED -- unclaimed but did not raise");
        CHECK(refused);
      }
    }
  }
}

MIMETIKA_TEST_MAIN()
