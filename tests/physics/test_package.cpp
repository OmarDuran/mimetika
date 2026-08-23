#include <memory>
#include <string>

#include "../mesh_fixtures.hpp"
#include "../mimetika_test.hpp"
#include "mimetika/physics/package.hpp"

using mimetika::physics::Composition;
using mimetika::physics::DofLayout;
using mimetika::physics::Package;
using mimetika::physics::Requirements;
using mimetika::physics::Scope;

namespace {

// two packages that know nothing of each other: one supplies a
// displacement, the other needs one
struct Mechanics final : Package {
  std::string name() const override { return "Mechanics"; }
  Requirements requirements(int dim, int) const override {
    Requirements r;
    r.fields.push_back({"u", DofLayout::cell_wise(dim, 1, dim)});
    r.provides = {"displacement"};
    r.slots = {{"stiffness", Scope::rock}};
    return r;
  }
  void attach(exokal::forms::Model&, const exokal::forms::TermContext&) const override {}
};

struct Pressure final : Package {
  std::string name() const override { return "Pressure"; }
  Requirements requirements(int dim, int) const override {
    Requirements r;
    r.fields.push_back({"p", DofLayout::cell_wise(dim)});
    r.provides = {"pressure"};
    r.slots = {{"porosity", Scope::rock}, {"density", Scope::fluid}};
    return r;
  }
  void attach(exokal::forms::Model&, const exokal::forms::TermContext&) const override {}
};

// poromechanics is not a physics: it is the coupling between two that are
struct PoroCoupling final : Package {
  std::string name() const override { return "PoroCoupling"; }
  Requirements requirements(int, int) const override {
    Requirements r;
    r.needs = {"pressure", "displacement"};  // and contributes no field of its own
    r.slots = {{"biot", Scope::rock}, {"normal_permeability", Scope::interface}};
    return r;
  }
  void attach(exokal::forms::Model&, const exokal::forms::TermContext&) const override {}
};

}  // namespace

// A composition is validated. Composing the coupling without the mechanics is
// caught here, by name, rather than by a term reading past the end of a stencil
// during assembly.
MIMETIKA_TEST(a_missing_capability_is_reported_by_name) {
  Composition c;
  c.emplace<Pressure>();
  c.emplace<PoroCoupling>();  // needs a displacement nobody provides

  bool threw = false;
  try {
    c.validate(3);
  } catch (const std::invalid_argument& e) {
    const std::string w = e.what();
    threw = w.find("PoroCoupling") != std::string::npos &&
            w.find("'displacement'") != std::string::npos;
  }
  CHECK(threw);

  Composition full;
  full.emplace<Pressure>();
  full.emplace<Mechanics>();
  full.emplace<PoroCoupling>();
  full.validate(3);  // and now it stands
  CHECK(full.size() == 3);
}

MIMETIKA_TEST(two_packages_may_not_provide_the_same_capability) {
  Composition c;
  c.emplace<Pressure>();
  c.emplace<Pressure>();
  bool threw = false;
  try {
    c.validate(3);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find("'pressure'") != std::string::npos;
  }
  CHECK(threw);
}

// The space is the union of what the packages contribute, in the order they
// were added — and no term depends on that order, because exokal resolves a
// term's fields by name against whatever space it is handed.
MIMETIKA_TEST(the_space_is_the_union_of_the_packages_fields) {
  Composition c;
  c.emplace<Pressure>();
  c.emplace<Mechanics>();
  c.emplace<PoroCoupling>();

  const auto m = mimetika_test::unit_cube();
  const auto s = c.space(m.topology(), 3);
  CHECK(s.n_fields() == 2);  // p and u; the coupling adds none
  CHECK(s.has("p") && s.has("u"));
  CHECK(s.index_of("p") == 0 && s.index_of("u") == 1);

  // the displacement carries one functional per cell with dim components
  CHECK(s.map(s.index_of("u")).layout().degree() == 3);
  CHECK(s.map(s.index_of("u")).layout().components == 3);
}

// The configuration surface is reportable. Every closure every package needs,
// with the scope it binds at — what a driver prints so a user need not read the
// source to configure a model.
MIMETIKA_TEST(the_closure_slots_are_reported_with_their_scope) {
  Composition c;
  c.emplace<Pressure>();
  c.emplace<Mechanics>();
  c.emplace<PoroCoupling>();

  const auto slots = c.slots(3);
  CHECK(slots.size() == 5);  // porosity, density, stiffness, biot, normal_permeability

  int fluid = 0, rock = 0, iface = 0;
  for (const auto& s : slots) {
    fluid += s.scope == Scope::fluid;
    rock += s.scope == Scope::rock;
    iface += s.scope == Scope::interface;
  }
  CHECK(fluid == 1 && rock == 3 && iface == 1);

  // the out-of-plane permeability is bound to a stratum pair, which is why
  // it cannot be a per-cell coefficient
  bool found = false;
  for (const auto& s : slots) {
    if (s.name == "normal_permeability") found = s.scope == Scope::interface;
  }
  CHECK(found);
}

MIMETIKA_TEST_MAIN()
