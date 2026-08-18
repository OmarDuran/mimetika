// TWO BENCHMARKS, ONE MODEL, TWO CONFIGURATIONS.
//
//   consolidation   Terzaghi's column (Coussy Sect. 5.2.2)
//   borehole        drilling of a borehole (Coussy Sect. 5.2.3), plane strain
//
// Nothing below chooses a discretization, a term or a package. Both problems
// are the `consolidation` model of the catalogue -- Biot poroelasticity on the
// mimetic-AFW-BDM de Rham spaces -- and they differ only in what a
// PoroelasticModel carries: a domain, five material numbers, an initial
// state and a list of boundary forms.
//
// The borehole is the one that could not have been written before the boundary
// conditions became forms. Its wall is curved, so every facet there has its own
// normal, and "the radial stress is -p1" is a single statement about all of
// them rather than a component to pin.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "mimetika/model/poroelastic_model.hpp"
#include "mimetika/linear_solver/petsc.hpp"

using namespace mimetika;
using graphos::Index;

namespace {

constexpr double kPi = 3.14159265358979323846;

enum class Family { cartesian, simplex };

// ---- the domains, in either dimension and either cell family -------------
//
// One generator per (problem, dimension); the family is a split inside it. A
// simplex mesh is not a different problem and must not be a different code
// path above the mesh: everything downstream -- the operators, the boundary
// forms, the solver -- sees only a complex.

// EXTRUSION, shared by both three-dimensional domains. A layer of quads
// becomes hexahedra; a layer of triangles becomes prisms, each split into
// three tetrahedra by the standard rule that tiles and keeps orientation.
exokal::Mesh extrude(const std::vector<exokal::Mesh::Point>& plane,
                     const std::vector<std::vector<Index>>& faces, double h, Family family) {
  const auto n = static_cast<Index>(plane.size());
  std::vector<exokal::Mesh::Point> pts;
  for (int layer = 0; layer < 2; ++layer) {
    for (const auto& q : plane) pts.push_back({q[0], q[1], layer * h});
  }
  const auto lo = [](Index v) { return v; };
  const auto hi = [n](Index v) { return v + n; };

  if (family == Family::simplex) {
    // A PRISM SPLITS INTO THREE TETRAHEDRA, but only one choice of the three
    // quadrilateral-face diagonals makes neighbouring prisms agree. Getting it
    // wrong does not produce a warning: it produces a mesh with internal holes,
    // whose "boundary" is most of the facets, and the first solve fails on a
    // system that was never a discretization of anything.
    //
    // Ordering each triangle's vertices by GLOBAL INDEX before applying a fixed
    // rule is what makes the choice agree: two prisms sharing a quad face see
    // the same two global indices on it and cut it the same way.
    std::vector<std::vector<Index>> cells;
    for (const auto& f : faces) {
      std::array<Index, 3> v{f[0], f[1], f[2]};
      std::sort(v.begin(), v.end());
      const Index a = v[0], b = v[1], c = v[2];
      cells.push_back({lo(a), lo(b), lo(c), hi(c)});
      cells.push_back({lo(a), lo(b), hi(b), hi(c)});
      cells.push_back({lo(a), hi(a), hi(b), hi(c)});
    }
    return exokal::Mesh::from_simplices(3, std::move(pts), cells);
  }
  std::vector<std::vector<std::vector<Index>>> cells;
  for (const auto& f : faces) {
    const Index a = f[0], b = f[1], c = f[2], d = f[3];
    cells.push_back({{lo(a), lo(d), lo(c), lo(b)},  // bottom, inward cycle
                     {hi(a), hi(b), hi(c), hi(d)},  // top
                     {lo(a), lo(b), hi(b), hi(a)},
                     {lo(b), lo(c), hi(c), hi(b)},
                     {lo(c), lo(d), hi(d), hi(c)},
                     {lo(d), lo(a), hi(a), hi(d)}});
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

// A COLUMN: one cell across, n tall. Terzaghi's domain.
exokal::Mesh column(int n, int dim, Family family, double h, double width) {
  std::vector<exokal::Mesh::Point> plane;  // the cross-section, in x-y
  std::vector<std::vector<Index>> faces;
  if (dim == 2) {
    // a strip of width `width`, n cells tall: the axis is y
    const auto vid = [](int i, int j) { return static_cast<Index>(j * 2 + i); };
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= 1; ++i) plane.push_back({i * width, j * h / n, 0.0});
    }
    std::vector<std::vector<Index>> cells;
    for (int j = 0; j < n; ++j) {
      const Index a = vid(0, j), b = vid(1, j), c = vid(1, j + 1), d = vid(0, j + 1);
      if (family == Family::simplex) {
        cells.push_back({a, b, c});
        cells.push_back({a, c, d});
      } else {
        cells.push_back({a, b, c, d});
      }
    }
    return family == Family::simplex ? exokal::Mesh::from_simplices(2, std::move(plane), cells)
                                     : exokal::Mesh::from_polygons(std::move(plane), cells);
  }
  // three dimensions: a 1 x 1 cross-section extruded n times is awkward to
  // express through `extrude`, so the column is built directly along z
  std::vector<exokal::Mesh::Point> pts;
  const auto vid = [](int i, int j, int k) { return static_cast<Index>((k * 2 + j) * 2 + i); };
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= 1; ++i) pts.push_back({i * width, j * width, k * h / n});
    }
  }
  if (family == Family::simplex) {
    static constexpr int kuhn[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 4, 5, 7},
                                       {0, 4, 6, 7}, {0, 2, 6, 7}, {0, 2, 3, 7}};
    std::vector<std::vector<Index>> cells;
    for (int k = 0; k < n; ++k) {
      const auto corner = [&](int l) { return vid(l & 1, (l >> 1) & 1, k + ((l >> 2) & 1)); };
      for (const auto& q : kuhn) {
        cells.push_back({corner(q[0]), corner(q[1]), corner(q[2]), corner(q[3])});
      }
    }
    return exokal::Mesh::from_simplices(3, std::move(pts), cells);
  }
  static constexpr int faces3[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                       {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<std::vector<std::vector<Index>>> cells;
  for (int k = 0; k < n; ++k) {
    std::vector<std::vector<Index>> cell;
    for (const auto& f : faces3) {
      std::vector<Index> cyc;
      for (const int l : f) cyc.push_back(vid(l & 1, (l >> 1) & 1, k + ((l >> 2) & 1)));
      cell.push_back(std::move(cyc));
    }
    cells.push_back(std::move(cell));
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

// A QUARTER ANNULUS, graded geometrically in the radius, and in three
// dimensions one layer of it: the plane-strain closure is rollers and seals on
// the two faces normal to the axis, so the answer must not move.
exokal::Mesh annulus(int nr, int nt, int dim, Family family, double a, double b, double h) {
  std::vector<exokal::Mesh::Point> plane;
  const auto vid = [nt](int i, int j) { return static_cast<Index>(i * (nt + 1) + j); };
  const double growth = std::pow(b / a, 1.0 / nr);
  for (int i = 0; i <= nr; ++i) {
    const double r = a * std::pow(growth, i);
    for (int j = 0; j <= nt; ++j) {
      const double th = 0.5 * kPi * static_cast<double>(j) / nt;
      plane.push_back({r * std::cos(th), r * std::sin(th), 0.0});
    }
  }
  std::vector<std::vector<Index>> faces;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nt; ++j) {
      const Index p0 = vid(i, j), p1 = vid(i + 1, j), p2 = vid(i + 1, j + 1), p3 = vid(i, j + 1);
      if (family == Family::simplex) {
        faces.push_back({p0, p1, p2});
        faces.push_back({p0, p2, p3});
      } else {
        faces.push_back({p0, p1, p2, p3});
      }
    }
  }
  if (dim == 3) return extrude(plane, faces, h, family);
  return family == Family::simplex ? exokal::Mesh::from_simplices(2, std::move(plane), faces)
                                   : exokal::Mesh::from_polygons(std::move(plane), faces);
}

std::array<double, 9> isotropic(double s, int d) {
  std::array<double, 9> out{};
  for (int k = 0; k < d; ++k) out[static_cast<std::size_t>(k * 3 + k)] = s;
  return out;
}

// march a built problem, reporting at the requested time factors
void march(PoroelasticModel& prob, const std::vector<double>& report_at, double t_scale,
           const std::function<void(double, const PoroelasticModel&)>& report) {
  // BIND ONCE: the tangent is constant for a linear model at constant dt, so
  // the factorization is paid one time and every step is a back-substitution
  solver::PetscSolver petsc;
  petsc.factorize(prob.system());
  std::size_t next = 0;
  const int steps = static_cast<int>(std::ceil(report_at.back() * t_scale / prob.dt()));
  std::vector<double> x;
  for (int s = 1; s <= steps && next < report_at.size(); ++s) {
    const auto b = prob.step_rhs();
    const auto rep = petsc.solve(b, x);
    if (!rep.converged) {
      std::printf("step %d did not converge: %s\n", s, rep.reason.c_str());
      return;
    }
    prob.accept(x);
    const double T = prob.dt() * s / t_scale;
    if (T + 1e-12 >= report_at[next]) {
      report(report_at[next], prob);
      ++next;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "consolidation";
  // an optional trailing realization flag: `stabilized` selects the
  // stabilized_bdm stress star on the same bdm flux pairing
  PoroelasticModel::Layer layer = PoroelasticModel::Layer::bdm;
  for (int a = 1; a < argc; ++a) {
    if (std::string(argv[a]) == "stabilized") layer = PoroelasticModel::Layer::bdm_stabilized;
  }
  const int dim = argc > 2 ? std::atoi(argv[2]) : 3;
  const std::string fam = argc > 3 ? argv[3] : "cart";
  const Family family = fam == "simplex" ? Family::simplex : Family::cartesian;
  if (dim != 2 && dim != 3) {
    std::printf("dimension must be 2 or 3\n");
    return 1;
  }
  const int axis = dim - 1;  // the column axis is the last coordinate

  if (which == "consolidation") {
    // ---- TERZAGHI'S COLUMN (Coussy Sect. 5.2.2) ------------------------
    const int n = argc > 4 ? std::atoi(argv[4]) : 20;
    const double h = 1.0, width = 1.0;
    PoroelasticMaterial mat{1.0, 1.0, 1.0, 0.0, 1.0};  // mu, lam, b, 1/M, k
    const exokal::Mesh m = column(n, dim, family, h, width);
    const graphos::Complex& c = m.topology();
    // the uniaxial clock, which is a property of the material and not of d
    const double c_v =
        mat.mobility / (mat.inverse_biot_modulus + mat.biot * mat.biot / mat.oedometer_modulus());
    const double dt = 1.0e-4 * h * h / c_v;

    PoroelasticModel prob(m, dim, mat, dt, layer);
    std::vector<Index> loaded, confined;
    for (const Index f : boundary_facets(c, dim)) {
      const auto x = exokal::centroid(m, dim - 1, f);
      (std::abs(x[static_cast<std::size_t>(axis)] - h) < 1e-9 ? loaded : confined).push_back(f);
    }
    std::array<double, 9> load{};
    load[static_cast<std::size_t>(axis * 3 + axis)] = -1.0;
    // MECHANICS: the step load on top, free slip everywhere else
    prob.mechanics().emplace<TractionBC>(loaded, load);
    prob.mechanics().emplace<FreeSlipBC>(confined);
    // FLOW: sealed on the confined faces; the drained top needs nothing, a
    // zero pressure there being the homogeneous natural case
    prob.flow().emplace<NormalFluxBC>(confined);
    prob.build();

    std::printf("consolidation %dD %s: %zu cells, %zu dofs, c_v=%.6f\n", dim,
                family == Family::simplex ? "simplex" : "cartesian", prob.n_cells(),
                prob.simulation().n_dofs(), c_v);
    const std::vector<double> factors = {0.001, 0.01, 0.1, 0.25, 0.5, 1.0};
    std::FILE* out = std::fopen(argc > 5 ? argv[5] : "consolidation.txt", "w");
    std::fprintf(out, "# consolidation %dD %s c_v=%g\n# factor zbar pbar\n", dim, fam.c_str(), c_v);
    march(prob, factors, h * h / c_v, [&](double T, const PoroelasticModel& p) {
      // the deepest cell, found rather than guessed at: a triangle's centroid
      // and a quadrilateral's sit at different heights in the same column
      double deep = 0.0, zmin = 1e300;
      for (Index e = 0; e < c.count(dim); ++e) {
        const double z = exokal::centroid(m, dim, e)[static_cast<std::size_t>(axis)];
        std::fprintf(out, "%g %.10g %.10g\n", T, (h - z) / h, p.cell_pressure(e));
        if (z < zmin) {
          zmin = z;
          deep = p.cell_pressure(e);
        }
      }
      std::printf("  T = %-6g  p(base) = %+.6f\n", T, deep);
    });
    std::fclose(out);
    return 0;
  }

  if (which == "borehole") {
    // ---- COUSSY SECT. 5.2.3, plane strain ------------------------------
    const int nr = argc > 4 ? std::atoi(argv[4]) : 60;
    const int nt = argc > 5 ? std::atoi(argv[5]) : 16;
    const double a = 1.0, R = 40.0, hz = 1.0;
    const double p0 = 1.0, p1 = 2.0, w = 3.0, M = 10.0;
    PoroelasticMaterial mat{1.0, 1.0, 0.8, 1.0 / M, 1.0};
    const exokal::Mesh m = annulus(nr, nt, dim, family, a, R, hz);
    const graphos::Complex& c = m.topology();
    const double cf = mat.diffusivity();
    const double dt = (argc > 7 ? std::atof(argv[7]) : 0.002) * a * a / cf;

    PoroelasticModel prob(m, dim, mat, dt, layer);
    std::vector<Index> sym, wall, far;
    const double rmid = 0.5 * (a + R);
    for (const Index f : boundary_facets(c, dim)) {
      const auto x = exokal::centroid(m, dim - 1, f);
      const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      // the symmetry planes, and in three dimensions the plane-strain closure
      const bool on_sym = std::abs(x[0]) < 1e-8 || std::abs(x[1]) < 1e-8 ||
                          (dim == 3 && (std::abs(x[2]) < 1e-8 || std::abs(x[2] - hz) < 1e-8));
      if (on_sym) {
        sym.push_back(f);
      } else if (r < rmid) {
        wall.push_back(f);
      } else {
        far.push_back(f);
      }
    }
    // MECHANICS: free slip on every symmetry plane, hydrostatic tractions on
    // both radial boundaries -- the wall's is ONE form over a curved surface
    prob.mechanics().emplace<FreeSlipBC>(sym);
    prob.mechanics().emplace<TractionBC>(wall, isotropic(-p1, dim));
    prob.mechanics().emplace<TractionBC>(far, isotropic(-w, dim));
    // FLOW: sealed on the symmetry planes, prescribed pressure on the radial
    // ones. The two sets are genuinely different here.
    prob.flow().emplace<NormalFluxBC>(sym);
    prob.flow().emplace<PressureBC>(wall, p1);
    prob.flow().emplace<PressureBC>(far, p0);
    prob.initial_stress(isotropic(-w, dim));
    prob.initial_pressure(p0);
    prob.build();

    std::printf("borehole %dD %s: %lld cells, %zu dofs, c_f=%.6f, wall %zu far %zu sym %zu\n", dim,
                family == Family::simplex ? "simplex" : "cartesian", (long long)c.count(dim),
                prob.simulation().n_dofs(), cf, wall.size(), far.size(), sym.size());

    const std::vector<double> factors = {0.1, 0.4, 0.8};
    std::FILE* out = std::fopen(argc > 6 ? argv[6] : "borehole.txt", "w");
    std::fprintf(out, "# borehole %dD %s a=%g R=%g c_f=%g\n# T r pbar\n", dim, fam.c_str(), a, R,
                 cf);
    march(prob, factors, a * a / cf, [&](double T, const PoroelasticModel& p) {
      double pw = 0.0, rw = 1e300;
      for (Index e = 0; e < c.count(dim); ++e) {
        const auto x = exokal::centroid(m, dim, e);
        const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        std::fprintf(out, "%g %.10g %.10g\n", T, r, (p.cell_pressure(e) - p0) / (p1 - p0));
        if (r < rw) {
          rw = r;
          pw = (p.cell_pressure(e) - p0) / (p1 - p0);
        }
      }
      std::printf("  T = %-6g  pbar(wall) = %+.6f\n", T, pw);
    });
    std::fclose(out);
    return 0;
  }

  std::printf("usage: %s [consolidation|borehole] [2|3] [cart|simplex] ...\n", argv[0]);
  return 1;
}
