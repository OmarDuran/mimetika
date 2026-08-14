#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/solver/petsc.hpp"

// THE SAME PROBLEM IN EVERY DIMENSION AND EVERY CELL FAMILY.
//
// Confined uniaxial compression has no freedom left to get wrong: rollers on
// the base and sides, a uniform compressive traction on top, zero lateral
// strain forced by geometry alone. So elasticity gives the whole answer in
// closed form, in ANY dimension, with the same two lines:
//
//     sigma_nn = lam/(lam + 2 mu) sigma_axial     on the confined facets
//     eps_axial = sigma_axial / K_oed,            K_oed = lam + 2 mu
//
// Running it on quadrilaterals, triangles, hexahedra and tetrahedra is what
// separates "the code has a 2D branch" from "the 2D discretization is the same
// method". Each family exercises something different:
//
//   triangle   3 edges x 2 moments = 6 = dim[P_1]^2, so the mimetic-BDM is
//              BDM_1 with NO enrichment and N is square by construction
//   quad       8 > 6, so the rot enrichment must supply exactly 2
//   tet        4 facets x 3 = 12 = dim[P_1]^3, again no enrichment
//   hex        18 > 12, so the curl enrichment supplies 6
//
// and the mimetic-AFW is d copies of whichever of those it sits on, with the
// weak-symmetry pairing carrying d(d-1)/2 rows -- three on a tet, ONE on a
// triangle, where skew(2) is a line.

using exokal::hodge::DeRhamGeometryCache;
using exokal::hodge::FluxHodge;
using exokal::hodge::StressOperators;
using graphos::Index;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;

namespace {

constexpr double kMu = 1.0, kLam = 1.0, kLoad = 1.0;

// n x n cells over the unit square, as quadrilaterals or as triangles
exokal::Mesh square(int n, bool simplex) {
  std::vector<exokal::Mesh::Point> p;
  const auto vid = [n](int i, int j) { return static_cast<Index>(j * (n + 1) + i); };
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      p.push_back({static_cast<double>(i) / n, static_cast<double>(j) / n, 0.0});
    }
  }
  std::vector<std::vector<Index>> cells;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (simplex) {
        cells.push_back({vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)});
        cells.push_back({vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)});
      } else {
        cells.push_back({vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
      }
    }
  }
  return simplex ? exokal::Mesh::from_simplices(2, std::move(p), cells)
                 : exokal::Mesh::from_polygons(std::move(p), cells);
}

// n x n x n cells over the unit cube, as hexahedra or as tetrahedra
exokal::Mesh cube(int n, bool simplex) {
  std::vector<exokal::Mesh::Point> p;
  const auto vid = [n](int i, int j, int k) {
    return static_cast<Index>((k * (n + 1) + j) * (n + 1) + i);
  };
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        p.push_back({static_cast<double>(i) / n, static_cast<double>(j) / n,
                     static_cast<double>(k) / n});
      }
    }
  }
  if (simplex) {
    // the six-tetrahedron (Kuhn) subdivision of each cube: it tiles, and every
    // tetrahedron is positively oriented
    static constexpr int kuhn[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 4, 5, 7},
                                       {0, 4, 6, 7}, {0, 2, 6, 7}, {0, 2, 3, 7}};
    std::vector<std::vector<Index>> cells;
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const auto corner = [&](int l) {
            return vid(i + (l & 1), j + ((l >> 1) & 1), k + ((l >> 2) & 1));
          };
          for (const auto& t : kuhn) {
            cells.push_back({corner(t[0]), corner(t[1]), corner(t[2]), corner(t[3])});
          }
        }
      }
    }
    return exokal::Mesh::from_simplices(3, std::move(p), cells);
  }
  static constexpr int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                      {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<std::vector<std::vector<Index>>> cells;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        std::vector<std::vector<Index>> cell;
        for (const auto& f : faces) {
          std::vector<Index> cyc;
          for (const int l : f) {
            cyc.push_back(vid(i + (l & 1), j + ((l >> 1) & 1), k + ((l >> 2) & 1)));
          }
          cell.push_back(std::move(cyc));
        }
        cells.push_back(std::move(cell));
      }
    }
  }
  return exokal::Mesh::from_polyhedra(std::move(p), cells);
}

struct Result {
  double sigma_lateral{0.0};  // worst error against lam/(lam+2mu) sigma_axial
  double displacement{0.0};   // worst error against eps_axial * x_axial
  std::size_t n_stabilized{0};
  std::size_t n_dofs{0};
};

// Solve confined compression on `m` and measure both closed-form quantities.
Result confined(const exokal::Mesh& m, int d) {
  const graphos::Complex& c = m.topology();
  const int axis = d - 1;  // the column axis is the last coordinate

  const DeRhamGeometryCache geo = DeRhamGeometryCache::build(m, d);
  const StressOperators ops =
      StressOperators::build(m, d, kMu, kLam, StressOperators::Realization::derham, &geo);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);

  Simulation sim(Catalogue::instance().build("linear_elasticity", {}),
                 {StratumSpec{"ambient", &c, d, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();

  std::vector<Index> loaded, confined_f;
  for (const Index f : mimetika::boundary_facets(c, d)) {
    const auto x = exokal::centroid(m, d - 1, f);
    (std::abs(x[static_cast<std::size_t>(axis)] - 1.0) < 1e-9 ? loaded : confined_f).push_back(f);
  }
  std::array<double, 9> applied{};
  applied[static_cast<std::size_t>(axis * 3 + axis)] = -kLoad;
  mimetika::impose_traction(sim.constraints(), sp, "s_0", d, m, loaded, applied);
  mimetika::impose_free_slip(sim.constraints(), sp, "s_0", d, m, confined_f);
  sim.freeze_constraints();

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const auto A = mimetika::solver::SparseSystem::from(jac);
  std::vector<double> b(sim.n_dofs(), 0.0), x;
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    if (sim.constraints().pinned(i)) {
      b[i] = sim.constraints().scale_at(i) * sim.constraints().rhs_at(i);
    }
  }
  mimetika::solver::PetscSolver petsc;
  const auto rep = petsc.solve(A, b, x);
  if (!rep.converged) throw std::runtime_error("confined: " + rep.reason);

  const double k_oed = kLam + 2.0 * kMu;
  const double s_exact = kLam / k_oed * (-kLoad);
  const double e_exact = -kLoad / k_oed;

  Result out;
  out.n_stabilized = ops.n_stabilized();
  out.n_dofs = sim.n_dofs();

  // the normal traction on every confined facet, read through the FORM that
  // imposed nothing there: n . (sigma n), divided by the measure it was
  // integrated against
  const auto& ms = sp.map(sp.index_of("s_0"));
  const auto s_off = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
  for (const Index f : confined_f) {
    const auto fr = mimetika::FacetFrame::of(m, d, mimetika::cofacet_of(m, d, f), f);
    // only the LATERAL facets carry lam/(lam+2mu) sigma_axial. The base is
    // confined too, but the axial load transmits straight through it, so its
    // normal traction is sigma_axial itself -- a different closed form, and
    // checking it against the lateral one would fail by exactly 2/3 here.
    if (std::abs(fr.normal[static_cast<std::size_t>(axis)]) > 1e-9) continue;
    double t = 0.0;
    for (int k = 0; k < d; ++k) {
      t += fr.normal[static_cast<std::size_t>(k)] *
           x[s_off + static_cast<std::size_t>(ms.global(d - 1, f, 0, k))];
    }
    out.sigma_lateral = std::max(out.sigma_lateral, std::abs(t / fr.measure - s_exact));
  }

  // u is stored as the cell integral, with the opposite sign of the convention
  // the closed form is written in
  const auto& mu_ = sp.map(sp.index_of("u_0"));
  const auto u_off = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
  for (Index e = 0; e < c.count(d); ++e) {
    const double vol = exokal::measure(m, d, e);
    const double z = exokal::centroid(m, d, e)[static_cast<std::size_t>(axis)];
    const double uz =
        -x[u_off + static_cast<std::size_t>(mu_.global(d, e, 0, axis))] / vol;
    out.displacement = std::max(out.displacement, std::abs(uz - e_exact * z));
  }
  return out;
}

}  // namespace

// THE 2 x 2 MATRIX: two dimensions, two cell families, one closed form.
MIMETIKA_TEST(confined_compression_is_exact_in_every_dimension_and_family) {
  struct Case {
    const char* name;
    exokal::Mesh mesh;
    int dim;
    std::size_t expect_stabilized;
  };
  std::vector<Case> cases;
  cases.push_back({"2D quadrilateral", square(3, false), 2, 0});
  cases.push_back({"2D triangle", square(3, true), 2, 0});
  cases.push_back({"3D hexahedron", cube(2, false), 3, 0});
  cases.push_back({"3D tetrahedron", cube(2, true), 3, 0});

  for (const Case& k : cases) {
    const Result r = confined(k.mesh, k.dim);
    std::printf("  %-18s %6zu dofs   sigma_lat %.2e   u %.2e   stabilized %zu\n", k.name,
                r.n_dofs, r.sigma_lateral, r.displacement, r.n_stabilized);
    // the de Rham realization is unisolvent by construction: nothing stabilizes
    CHECK(r.n_stabilized == k.expect_stabilized);
    // and the closed form is reproduced, not approximated
    CHECK(r.sigma_lateral < 1e-11);
    CHECK(r.displacement < 1e-11);
  }
}

MIMETIKA_TEST_MAIN()
