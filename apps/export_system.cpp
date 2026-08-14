#include <cstdio>
#include <string>

#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/io/system_export.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"

// Assemble a named model on a small box and write the system out for the
// Python to solve. This is the seam during the port: the C++ assembles, the
// Python does the linear algebra and the comparison.

using namespace mimetika;

static exokal::Mesh box(int n) {
  const auto vid = [n](int i, int j, int k) {
    return static_cast<graphos::Index>((k * (n + 1) + j) * (n + 1) + i);
  };
  static constexpr int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                      {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<exokal::Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        pts.push_back({double(i) / n, double(j) / n, double(k) / n});
      }
    }
  }
  std::vector<std::vector<std::vector<graphos::Index>>> cells;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        std::vector<std::vector<graphos::Index>> cell;
        for (const auto& f : faces) {
          std::vector<graphos::Index> cyc;
          for (const int l : f) {
            cyc.push_back(vid(i + (l & 1), j + ((l >> 1) & 1), k + ((l >> 2) & 1)));
          }
          cell.push_back(std::move(cyc));
        }
        cells.push_back(std::move(cell));
      }
    }
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

int main(int argc, char** argv) {
  const std::string model = argc > 1 ? argv[1] : "linear_elasticity";
  const int n = argc > 2 ? std::atoi(argv[2]) : 2;
  const std::string out = argc > 3 ? argv[3] : "system.bin";

  const exokal::Mesh m = box(n);
  const graphos::Complex& c = m.topology();

  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, 3, 1.0, 1.0);
  const exokal::hodge::FluxHodge hodge = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(1.0),
      exokal::hodge::FluxHodge::Realization::derham);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("flux_hodge", hodge);

  Simulation sim(physics::Catalogue::instance().build(model, {}),
                 {StratumSpec{"ambient", &c, 3, 0}}, ctx);

  // A LOADED COLUMN IN MINIATURE, which is what makes the exported system
  // worth solving: the traction is essential in this form, so the load is a
  // pinned value rather than a right-hand side. The base is held and the top
  // is pushed; the sides are left free, so the answer is uniaxial rather than
  // confined — enough to exercise every block without needing the closed form.
  const auto& sp = sim.epoch().stratum(0).space();
  const auto bottom = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, 1.0));
  if (sp.has("s_0")) {
    impose_traction(sim.constraints(), sp, "s_0", 3, m, bottom, std::array<double, 9>{});
    impose_traction(sim.constraints(), sp, "s_0", 3, m, top,{0,0,0, 0,0,0, 0,0,-1.0});
  }
  if (sp.has("q_0")) {
    impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, bottom);
    impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, top);
  }
  sim.freeze_constraints();

  io::export_system(out, sim);
  std::printf("%s on a %d^3 box: %zu dofs, %zu pinned -> %s\n", model.c_str(), n, sim.n_dofs(),
              sim.constraints().size(), out.c_str());
  return 0;
}
