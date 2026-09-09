#include <string>
#include <vector>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "exokal/hodge/coefficient.hpp"
#include "exokal/forms/epoch.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "mimetika/model/compositions/flow.hpp"

using exokal::hodge::Coefficient;
using exokal::hodge::FluxOperators;
using mimetika::physics::Catalogue;
using mimetika::physics::Composition;
using mimetika::physics::ModelOptions;
using mimetika::physics::Scope;

// "flow" and "compositional_flow" build the same package: they differ by the
// component count, hence by the field count, not by an implementation.
MIMETIKA_TEST(both_flow_models_are_the_same_package) {
  const auto& cat = Catalogue::instance();
  CHECK(cat.has("flow") && cat.has("compositional_flow"));

  const Composition single = cat.build("flow", {});
  ModelOptions o;
  o.components = 2;
  const Composition multi = cat.build("compositional_flow", o);

  CHECK(single.size() == 1 && multi.size() == 1);
  CHECK(single.at(0).name() == "Flow" && multi.at(0).name() == "Flow");

  // the same package, and only the field count differs
  CHECK(single.requirements_of(0, 3).fields.size() == 2);  // q, p
  CHECK(multi.requirements_of(0, 3).fields.size() == 4);   // q, p, z0, z1
}

// The closure slots of flow, by scope. The normal permeability is bound to a
// stratum pair and has nowhere to live in a per-cell field, hence Scope::interface.
MIMETIKA_TEST(flow_declares_its_closures_at_three_scopes) {
  const Composition c = Catalogue::instance().build("flow", {});
  const auto slots = c.slots(3);

  int fluid = 0, rock = 0, iface = 0;
  for (const auto& s : slots) {
    fluid += s.scope == Scope::fluid;
    rock += s.scope == Scope::rock;
    iface += s.scope == Scope::interface;
  }
  CHECK(fluid == 2);  // density, viscosity
  CHECK(rock == 2);   // porosity, permeability
  CHECK(iface == 1);  // normal_permeability
}

// End to end across the repository seam: number the composition's space, attach
// the package's term, assemble through exokal, and check the triplets index
// inside the epoch.
MIMETIKA_TEST(a_composed_flow_model_assembles) {
  const auto m = mimetika_test::hex_grid(3);
  const graphos::Complex& c = m.topology();

  const Composition comp = Catalogue::instance().build("flow", {});
  auto space = comp.space(c, 3);
  CHECK(space.n_fields() == 2);
  // named by codimension: this is the ambient stratum, so q_0 and p_0
  CHECK(space.has("q_0") && space.has("p_0"));
  // the flux sits on the facets, the pressure on the cells: an (n-1, n)
  // pair, which is what makes the second row a discrete divergence
  CHECK(space.map(space.index_of("q_0")).layout().degree() == 2);
  CHECK(space.map(space.index_of("p_0")).layout().degree() == 3);

  // the same package one codimension down names them q_1 and p_1
  const auto sub = comp.space(c, 3, 1);
  CHECK(sub.has("q_1") && sub.has("p_1"));

  // the closures the driver owns and the terms merely read
  // hexahedra are polytopes: derham_bdm enriches the de Rham product to
  // unisolvence there, and carries d moments per facet
  const FluxOperators hodge =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_bdm);
  exokal::forms::TermContext ctx;
  ctx.provide("flux_operators", hodge);

  exokal::forms::StratifiedEpoch se;
  se.add("top", 0, exokal::forms::Epoch(c, std::move(space), 3));

  exokal::forms::Model model;
  model.use(ctx);
  comp.attach(model, ctx);
  CHECK(model.size() == 1);
  CHECK(model.n_evaluations(se) == 1);

  const auto n = static_cast<std::size_t>(se.size());
  const std::vector<double> state(n, 1.0);
  exokal::forms::Workspace ws;
  exokal::forms::TripletSink sink(n);
  model.assemble(se, state, sink, ws);

  CHECK(sink.nnz() > 0);
  for (std::size_t k = 0; k < sink.nnz(); ++k) {
    CHECK(sink.row[k] >= 0 && sink.row[k] < se.size());
    CHECK(sink.col[k] >= 0 && sink.col[k] < se.size());
  }
}

MIMETIKA_TEST_MAIN()
