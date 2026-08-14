// THE COLUMN'S FIRST STEP, EXPORTED FOR COMPARISON AGAINST THE PYTHON.
//
// The port's claim is that the C++ assembles the same operator as the Python.
// Checking it needs the same mesh, the same material, the same step and the
// same four boundary conditions, and then the two matrices side by side.
//
// One thing does NOT carry across, and it is not a discrepancy: each code
// assigns its own CANONICAL NORMAL to a facet, from that mesh's stored vertex
// cycle. A facet whose two normals disagree has all of its moments signed, and
// its second in-facet tangent t2 = n x t1 signed a second time, so the two
// operators are related by the involution
//
//     P = (+) diag(s_f, s_f, 1),      s_f = sign(n_cpp . n_py)
//
// per facet and per component. That is a congruence by an orthogonal matrix:
// it leaves the spectrum and the assembled solution alone and changes every
// entry. So the geometry goes out alongside the system and the comparison
// forms P for itself rather than guessing at signs.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/io/system_export.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/models/poroelasticity.hpp"

using namespace mimetika;
using graphos::Index;

namespace {

// A column: one cell across, `n` cells tall, height h. The same generator the
// Terzaghi driver uses, so the comparison is on the driver's own mesh.
exokal::Mesh column(int n, double h, double width) {
  std::vector<exokal::Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= 1; ++i) pts.push_back({i * width, j * width, k * h / n});
    }
  }
  static constexpr int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                      {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<std::vector<std::vector<Index>>> cells;
  for (int k = 0; k < n; ++k) {
    std::vector<std::vector<Index>> cell;
    for (const auto& f : faces) {
      std::vector<Index> cyc;
      for (const int l : f) {
        cyc.push_back(static_cast<Index>(((k + ((l >> 2) & 1)) * 2 + ((l >> 1) & 1)) * 2 + (l & 1)));
      }
      cell.push_back(std::move(cyc));
    }
    cells.push_back(std::move(cell));
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 3;
  const double dt = argc > 2 ? std::atof(argv[2]) : 1.0;
  const std::string out = argc > 3 ? argv[3] : "column.bin";

  // the benchmark Column's material, unnormalized: a stiff, low-permeability
  // rock, so the comparison runs on the numbers the closed form is checked at
  const double h = argc > 4 ? std::atof(argv[4]) : 10.0;
  const double width = 1.0;
  // the material, overridable so the same comparison can run at the driver's
  // normalized constants as well as at the benchmark's physical ones
  const auto env = [](const char* k, double d) {
    const char* v = std::getenv(k);
    return v ? std::atof(v) : d;
  };
  const double mu = env("COLMU", 6.0e8);
  const double nu = argc > 5 ? std::atof(argv[5]) : env("COLNU", 0.2);
  const double lam = 2.0 * mu * nu / (1.0 - 2.0 * nu);
  const double alpha = env("COLALPHA", 0.9), inv_M = env("COLINVM", 1.0e-10);
  const double perm = env("COLPERM", 1.0e-13 / 1.0e-3);  // k / mu_f
  const double load = env("COLLOAD", 1.0e7);

  const double inv_mod = (1.0 - 2.0 * nu) / (2.0 * mu * (1.0 - 2.0 * nu + 3.0 * nu));
  const double storage = 3.0 * alpha * alpha * inv_mod + inv_M;

  const exokal::Mesh m = column(n, h, width);
  const graphos::Complex& c = m.topology();

  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, mu, lam);
  const exokal::hodge::FluxHodge hodge = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(perm),
      exokal::hodge::FluxHodge::Realization::derham);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("flux_hodge", hodge);

  physics::ModelOptions o;
  o.mobility = dt;  // backward Euler: the flux over the step is q~ = dt q
  o.storage = storage;
  o.volumetric_compliance = inv_mod;
  o.biot = alpha;
  Simulation sim(physics::Catalogue::instance().build("consolidation", o),
                 {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();

  const auto base = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, h));
  std::vector<Index> confined = base;
  for (const double v : {0.0, width}) {
    for (const int axis : {0, 1}) {
      for (const Index f : FacetSelector::where(m, 3, FacetSelector::at(axis, v))) {
        confined.push_back(f);
      }
    }
  }
  pin_facet_traction(sim.constraints(), sp, "s_0", 3, m, top,
                     {0, 0, 0, 0, 0, 0, 0, 0, -load});
  pin_facet_roller(sim.constraints(), sp, "s_0", 3, m, confined);
  pin_facets(sim.constraints(), sp, "q_0", 3, confined);
  sim.freeze_constraints();

  io::export_system(out, sim);

  // the geometry the comparison needs: which facet is which, and which way its
  // canonical normal points
  const std::string geo = out + ".geo";
  std::FILE* g = std::fopen(geo.c_str(), "w");
  std::fprintf(g, "%lld %lld\n", (long long)c.count(2), (long long)c.count(3));
  for (Index f = 0; f < c.count(2); ++f) {
    const auto x = exokal::centroid(m, 2, f);
    const auto av = exokal::face_area_vector(m, f);
    const double a = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
    std::fprintf(g, "%.17g %.17g %.17g %.17g %.17g %.17g\n", x[0], x[1], x[2], av[0] / a,
                 av[1] / a, av[2] / a);
  }
  for (Index e = 0; e < c.count(3); ++e) {
    const auto x = exokal::centroid(m, 3, e);
    std::fprintf(g, "%.17g %.17g %.17g\n", x[0], x[1], x[2]);
  }
  std::fclose(g);

  std::printf("column %d, dt = %.6e: %zu dofs, %zu pinned -> %s (+ .geo)\n", n, dt, sim.n_dofs(),
              sim.constraints().size(), out.c_str());
  return 0;
}
