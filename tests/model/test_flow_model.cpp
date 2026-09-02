#include <cmath>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"
#include "mimetika/linear_solver/petsc.hpp"

// The single-phase solver, against two closed forms, on every cell type.
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
// all three of exokal's flux inner products. The prism is the cell that matters
// most: neither a simplex nor a tensor-product cell, two triangles and three
// quadrilaterals, so a construction that handles it is polytopal rather than
// hex-and-tet with extra steps. In the plane a prism over an interval is a
// quadrilateral, so that row coincides with cartesian and is reported rather
// than skipped.
//
// The three products are different discretizations, not a refinement ladder:
//
//   derham      d moments per facet. On a simplex it is BDM_1 -- 3 edges x 2
//               in the plane, 4 facets x 3 in space -- unisolvent with no
//               enrichment; on a polytope the moments are completed with
//               div-free curl modes. Any cell, either dimension.
//   derham_rt   one flux per facet: RT_0, the minimal de Rham pair, whose
//               radial mode x - x_E is what lets div reach P_0 at all. d+1
//               modes against d+1 fluxes is the whole argument, so it is
//               simplices only -- triangle or tetrahedron -- and refuses the
//               rest rather than stabilizing them.
//   stabilized  one flux per facet on any polytope, in either dimension:
//               consistency plus a stabilization sized to trace(M1)/(d+2),
//               which is the scale at which it is the conforming RT_0 element
//               on a simplex (exokal hodge.test_rt_equivalence, vs basix).
//
// What they share is consistency -- every one of them reproduces a constant
// flux exactly -- and that is what the column asserts of all three. Where they
// differ is everything else, which is what the annulus measures.

using graphos::Index;
using mimetika::FlowModel;
using mimetika::mesh::Family;

namespace {

struct Outcome {
  double max_err{0.0};
  double rms_err{0.0};
  std::size_t cells{0};
  std::size_t dofs{0};
};

using Realization = mimetika::FlowModel::Realization;

Outcome run(const exokal::Mesh& m, int dim, const std::vector<Index>& high,
            const std::vector<Index>& low, const std::vector<Index>& sealed, double p_high,
            double p_low, const std::function<double(const exokal::Mesh::Point&)>& exact,
            Realization how = Realization::derham_bdm) {
  FlowModel prob(m, dim, 1.0, how);
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

// EVERY MEMBER, NOT ONLY THE ONES kProducts EXERCISES. The switch has no
// default on purpose: -Wswitch is then what reports the next realization added
// to the enum, and a default would silence it and let the new member print
// someone else's name. The unreachable return is for a value outside the enum.
const char* product_name(Realization r) {
  switch (r) {
    case Realization::derham_bdm: return "derham_bdm";
    case Realization::derham_rt: return "derham_rt";
    case Realization::stabilized_bdm: return "stabilized_bdm";
    case Realization::stabilized_rt: return "stabilized_rt";
    case Realization::diagonal_tpfa: return "diagonal_tpfa";
    case Realization::adaptive_rt: return "adaptive_rt";
  }
  return "?";
}

// What each product claims.
//
// All three claim every structured family in both dimensions. RT_0's argument is
// d+1 modes against d+1 facets, a simplex, but the consistency-only families
// enrich with curl-type divergence-free fields until the facet moments are
// unisolvent, and that reaches the tensor cells too.
//
// The claim is therefore bounded by facet count, not by cell type: past exokal's
// default_max_facets a cell is refused before any search is attempted, which is
// what a_cell_past_the_facet_limit_is_refused pins down.
bool supported(Realization, int, Family) { return true; }

// An n-gonal prism as a single cell: n side quads and two caps, so n + 2
// facets. The one family whose facet count is a free parameter.
exokal::Mesh drum(int n) {
  std::vector<exokal::Mesh::Point> pts;
  for (const double z : {0.0, 1.0}) {
    for (int i = 0; i < n; ++i) {
      const double a = 2.0 * M_PI * i / n;
      pts.push_back({std::cos(a), std::sin(a), z});
    }
  }
  std::vector<Index> bottom, top;
  for (int i = n - 1; i >= 0; --i) bottom.push_back(i);
  for (int i = 0; i < n; ++i) top.push_back(n + i);
  std::vector<std::vector<Index>> faces{bottom, top};
  for (int i = 0; i < n; ++i) {
    faces.push_back({i, (i + 1) % n, (i + 1) % n + n, i + n});
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), {faces});
}

}  // namespace

// A linear pressure is reproduced exactly. The flux space contains the constant
// fields, so the gradient of a linear pressure is represented with no error, on
// any cell type in any dimension. Anything above round-off here is a broken
// space rather than a coarse mesh, and no refinement would fix it.
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

// And the Dupuit profile, which is not in the space: the radial harmonic is
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

// ---- the two-point product --------------------------------------------------
//
// diagonal_tpfa is exokal's, and so is the question of where it is consistent:
// it reconstructs nothing, its M is the diagonal primal-dual star, and it is
// strongly consistent only where the mesh is K-orthogonal. exokal tests that.
// What is tested here is that mimetika reaches it -- that a model built with it
// lays out the space it should and solves the problem it claims.
//
// Its space is RT's: one flux per facet, so a model that mixed the two up
// would still assemble and still converge, and only the count would say so.
MIMETIKA_TEST(the_two_point_product_lays_out_one_flux_per_facet) {
  for (const int dim : {2, 3}) {
    for (const Family f : kFamilies) {
      const Outcome tpfa = column_case(6, dim, f, Realization::diagonal_tpfa);
      const Outcome rt = column_case(6, dim, f, Realization::derham_rt);
      std::printf("  tpfa    %dD %-10s %7zu dofs (rt %zu)   max %.2e\n", dim,
                  mimetika::mesh::name(f), tpfa.dofs, rt.dofs, tpfa.max_err);
      CHECK(tpfa.dofs == rt.dofs);
      CHECK(tpfa.cells == rt.cells);
    }
  }
}

// And it reproduces a linear pressure where it claims to. The hexahedral and
// prismatic columns are K-orthogonal -- the segment between two cell centroids
// meets their shared facet squarely -- and there the two-point flux is exact.
// The tetrahedral column is not, and is not asserted here: that boundary
// belongs to exokal, which tests it against the geometry rather than against a
// model.
MIMETIKA_TEST(the_two_point_product_is_exact_where_the_column_is_orthogonal) {
  for (const int dim : {2, 3}) {
    for (const Family f : {Family::cartesian, Family::prism}) {
      const Outcome o = column_case(6, dim, f, Realization::diagonal_tpfa);
      std::printf("  tpfa    %dD %-10s   max %.2e   rms %.2e\n", dim, mimetika::mesh::name(f),
                  o.max_err, o.rms_err);
      CHECK(o.max_err < 1e-10);
    }
  }
}

// ---- how the three products differ ----------------------------------------

// The spaces are not the same size, which is the concrete content of "different
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

// RT and the stabilized product return the same solved field on a simplex, and
// they are not the same operator.
//
// exokal measures the operators against basix (hodge.test_rt_equivalence):
// derham_rt is the conforming RT_0 element, while the stabilized product
// reconstructs on the constants alone -- three modes against four facet fluxes
// -- and carries a one-dimensional stabilization. Their matrices differ by
// about 3%.
//
// Yet the pressures agree to round-off: both spaces contain the constants, so
// both are consistent, and on this problem the stabilization does not reach the
// cell pressures. A model-level comparison therefore cannot be used to conclude
// two operators are the same; that conclusion belongs to the exokal test,
// against a conforming element.
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

// Where the consistency-only family stops, and that it stops by refusing.
//
// The enrichment is not unbounded: a cell with more facets than the search is
// willing to chase is refused at once, before any search, because F is known
// and discovering the refusal the slow way costs tens of milliseconds a cell.
// An n-gonal prism walks the boundary one facet at a time -- n sides and two
// caps -- so the limit is located rather than assumed.
//
// The stabilized construction has a fallback and takes every one of them, which
// is what makes the pair the test: the refusal is a property of the
// consistency-only argument and not of the cell being difficult.
MIMETIKA_TEST(a_cell_past_the_facet_limit_is_refused) {
  const auto builds = [](int n, Realization r) {
    try {
      (void)exokal::hodge::FluxOperators::build(drum(n), 3,
                                                exokal::hodge::Coefficient::uniform(1.0), r);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
  // the last accepted cell and the first refused one, on either side of the cap
  CHECK(builds(11, Realization::derham_rt));   // 13 facets
  CHECK(!builds(12, Realization::derham_rt));  // 14
  CHECK(!builds(16, Realization::derham_rt));
  // the stabilized product stabilizes instead of enriching, and has no limit
  CHECK(builds(12, Realization::stabilized_rt));
  CHECK(builds(16, Realization::stabilized_rt));
  std::printf("  derham_rt accepts 13 facets and refuses 14; stabilized_rt takes both\n");
}

MIMETIKA_TEST_MAIN()
