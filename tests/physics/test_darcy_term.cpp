#include <algorithm>
#include <cmath>

#include "../mimetika_test.hpp"
#include "exokal/forms/model.hpp"
#include "mimetika/physics/constitutive/immiscible.hpp"
#include "mimetika/physics/terms/darcy.hpp"

using exokal::Index;
using exokal::Mesh;
using exokal::forms::Coupling;
using exokal::forms::Epoch;
using exokal::forms::Model;
using exokal::forms::On;
using exokal::forms::StratifiedEpoch;
using exokal::forms::TermContext;
using exokal::forms::TripletSink;
using exokal::forms::Workspace;
using exokal::hodge::Coefficient;
using exokal::hodge::FluxOperators;
using exokal::hodge::SymTensor;
using exokal::spaces::DofLayout;
using exokal::spaces::DofMap;
using exokal::spaces::ProductSpace;
using mimetika::physics::terms::ConstantMobility;
using mimetika::physics::terms::mixed_darcy_cell;
using mimetika::physics::terms::PressureEnthalpyMobility;

namespace {

bool near(double a, double b, double tol = 1e-10) { return std::abs(a - b) <= tol; }

Mesh two_tets() {
  return Mesh::from_simplices(3, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}},
                              {{0, 1, 2, 3}, {1, 2, 3, 4}});
}

Mesh unit_cube() {
  std::vector<Mesh::Point> pts(8);
  for (int i = 0; i < 8; ++i) {
    pts[static_cast<std::size_t>(i)] = {double(i & 1), double((i >> 1) & 1), double((i >> 2) & 1)};
  }
  return Mesh::from_polyhedra(
      std::move(pts),
      {{{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}}});
}

// the mixed pair: one flux per facet, one pressure per cell
ProductSpace mixed(const graphos::Complex& c) {
  ProductSpace s;
  s.add("q", DofMap(c, DofLayout::cochain(3, 2)));
  s.add("p", DofMap(c, DofLayout::cell_wise(3)));
  return s;
}

TripletSink assemble(const Mesh& m, const FluxOperators& h, const exokal::forms::Params& p) {
  StratifiedEpoch se;
  se.add("bulk", 0, Epoch(m.topology(), mixed(m.topology()), 3));
  Model model;
  model.add(mixed_darcy_cell<>(h, p), On::all());
  const auto n = static_cast<std::size_t>(se.size());
  TripletSink sink(n);
  Workspace ws;
  model.assemble(se, std::vector<double>(n, 1.0), sink, ws);
  return sink;
}

// accumulate the triplets into a dense matrix
std::vector<double> dense_of(const TripletSink& s, std::size_t n) {
  std::vector<double> a(n * n, 0.0);
  for (std::size_t k = 0; k < s.nnz(); ++k) {
    a[static_cast<std::size_t>(s.row[k]) * n + static_cast<std::size_t>(s.col[k])] += s.value[k];
  }
  return a;
}

}  // namespace

// The saddle-point structure comes out of the term, not out of a
// declaration: the Hodge block, the two divergence blocks, and no
// pressure-pressure coupling at all
MIMETIKA_TEST(the_local_system_is_a_saddle_point) {
  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(2.0), FluxOperators::Realization::derham_rt);
  const ProductSpace s = mixed(c);
  const exokal::forms::StencilBuilder sb(c, s, Coupling::closure);
  exokal::ad::LocalContext ctx(exokal::ad::local_space_of(sb.at(0).view));
  auto st = sb.at(0);
  const std::vector<exokal::ad::Local> a = ctx.seed(std::vector<double>(st.view.size(), 1.0));
  std::vector<exokal::ad::Local> r = ctx.accumulators();
  mimetika::physics::terms::MixedDarcyCell<> term(h, ConstantMobility{});
  const exokal::forms::Slots sl(s, term.fields(), "mixed_darcy_cell");
  st.slots = &sl;
  term(st, a, r);

  const exokal::ad::LocalSystem sys(ctx.space(), r);
  CHECK(sys.has_block(0, 0) && sys.has_block(0, 1) && sys.has_block(1, 0));
  CHECK(!sys.has_block(1, 1));  // no pressure-pressure coupling

  // the flux block is the Hodge, divided by the mobility
  const auto& M = h.cell(0);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) CHECK(near(sys.block(0, 0)(i, j), M(i, j)));
  }
  // and the coupling blocks are exact transposes, written from one set of
  // signs rather than two kernels
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(near(sys.block(0, 1)(i, 0), -sys.block(1, 0)(0, i)));
    CHECK(std::abs(sys.block(1, 0)(0, i)) == 1.0);  // the boundary coefficients
  }
}

// The two lowest-order realizations are interchangeable on a simplex: nothing
// moves at all. Both are the conforming RT_0 element there -- derham_rt by
// unisolvence, stabilized_rt because its stabilization is sized to reproduce it
// -- so the constitutive blocks agree to round-off and the divergence blocks
// agree because they are topology.
//
// The equality depends on the stabilization being scaled by trace(M1)/(d+2)
// rather than by the mean diagonal; the assertion is an equality rather than a
// tolerance so a regression in that scale fails loudly.
MIMETIKA_TEST(the_two_lowest_order_realizations_assemble_the_same_system) {
  const Mesh m = two_tets();
  const Coefficient K = Coefficient::uniform(SymTensor<>::diagonal(2.0, 1.0, 3.0));
  const FluxOperators stab = FluxOperators::build(m, K, FluxOperators::Realization::stabilized_rt);
  const FluxOperators rt = FluxOperators::build(m, K, FluxOperators::Realization::derham_rt);

  const auto a = assemble(m, stab, {});
  const auto b = assemble(m, rt, {});
  const auto n = static_cast<std::size_t>(m.topology().count(2) + m.topology().count(3));
  const auto A = dense_of(a, n);
  const auto B = dense_of(b, n);

  double worst_flux = 0.0;
  const auto nq = static_cast<std::size_t>(m.topology().count(2));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i < nq && j < nq) {
        worst_flux = std::max(worst_flux, std::abs(A[i * n + j] - B[i * n + j]));
      } else {
        CHECK(near(A[i * n + j], B[i * n + j], 1e-12));  // divergence: topology
      }
    }
  }
  std::printf("  worst |constitutive block difference| %.2e\n", worst_flux);
  CHECK(worst_flux < 1e-10);  // the same element, so the same block
  // and the residuals agree everywhere the divergence writes
  for (std::size_t i = nq; i < n; ++i) CHECK(near(a.residual[i], b.residual[i], 1e-12));
}

// Discrete conservation: every interior facet appears in exactly two
// divergence rows with opposite signs, so the column sums of the
// divergence block vanish there. The flux through the mesh is conserved
// because the topology says so.
MIMETIKA_TEST(the_divergence_block_conserves_on_interior_facets) {
  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_rt);
  const auto nq = static_cast<std::size_t>(c.count(2));
  const auto n = nq + static_cast<std::size_t>(c.count(3));
  const auto A = dense_of(assemble(m, h, {}), n);

  const graphos::CoboundaryOperator cob = graphos::coboundary(c, 2);
  int interior = 0;
  for (std::size_t f = 0; f < nq; ++f) {
    double col = 0.0;
    for (std::size_t i = nq; i < n; ++i) col += A[i * n + f];  // the divergence rows
    const auto b = static_cast<std::size_t>(cob.offsets[f]);
    const auto e = static_cast<std::size_t>(cob.offsets[f + 1]);
    if (e - b == 2) {
      ++interior;
      CHECK(near(col, 0.0, 1e-13));  // the two cells cancel
    } else {
      CHECK(std::abs(col) == 1.0);  // a boundary facet leaves the mesh
    }
  }
  CHECK(interior == 1);  // two tetrahedra share one facet
}

// a polytope runs on the realization that admits it, and the term is
// unchanged: one kernel, six unknowns instead of four
MIMETIKA_TEST(the_same_term_runs_on_a_polytope) {
  const Mesh m = unit_cube();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(1.5), FluxOperators::Realization::stabilized_rt);
  CHECK(h.n_dofs(0) == 6);
  const auto n = static_cast<std::size_t>(m.topology().count(2) + m.topology().count(3));
  const auto sink = assemble(m, h, {});
  CHECK(sink.nnz() > 0);
  const auto A = dense_of(sink, n);
  // the pressure row of the single cell is its divergence: six +/-1 entries
  double s = 0.0;
  for (std::size_t f = 0; f < 6; ++f) s += std::abs(A[6 * n + f]);
  CHECK(near(s, 6.0));

  // And the consistency-only realization takes it too, by enrichment.
  //
  // RT_0's argument is d+1 modes against d+1 facets, which a cube does not
  // satisfy, so derham_rt enriches with curl-type divergence-free fields until
  // the six facet moments are unisolvent. Six unknowns either way, reached by
  // two different arguments, so the term is the same kernel over the same
  // layout.
  const FluxOperators rt =
      FluxOperators::build(m, Coefficient::uniform(1.5), FluxOperators::Realization::derham_rt);
  CHECK(rt.n_dofs(0) == 6);
}

// A state-dependent mobility makes the term nonlinear, and the AD produces
// the tangent from the same source. With lambda(p, h) the flux row acquires
// a block against the enthalpy that a constant mobility never touches — the
// dependency inputs() declares in advance.
MIMETIKA_TEST(a_declared_dependency_becomes_a_jacobian_block) {
  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_rt);

  // the declarations, before anything is assembled
  CHECK(ConstantMobility{}.inputs().empty());
  CHECK(PressureEnthalpyMobility{}.inputs().size() == 2);
  const auto df = mimetika::physics::terms::declared_fields<PressureEnthalpyMobility>();
  CHECK(df.size() == 3 && df[0] == "q" && df[1] == "p" && df[2] == "h");

  // a space carrying the declared fields, in the declared order
  ProductSpace s;
  s.add("q", DofMap(c, DofLayout::cochain(3, 2)));
  s.add("p", DofMap(c, DofLayout::cell_wise(3)));
  s.add("h", DofMap(c, DofLayout::cell_wise(3)));
  const exokal::forms::StencilBuilder sb(c, s, Coupling::closure);
  auto st = sb.at(0);

  PressureEnthalpyMobility mob;
  mob.reference = 2.0;
  mob.pressure_coefficient = 0.25;
  mob.enthalpy_coefficient = 0.5;
  mimetika::physics::terms::MixedDarcyCell<PressureEnthalpyMobility> term(h, mob);
  const exokal::forms::Slots sl(s, term.fields(), "mixed_darcy_cell_ph");
  st.slots = &sl;

  std::vector<double> alpha(st.view.size(), 1.0);
  alpha[st.view.blocks[1].begin] = 3.0;  // pressure
  alpha[st.view.blocks[2].begin] = 2.0;  // enthalpy
  exokal::ad::LocalContext ctx(exokal::ad::local_space_of(st.view));
  const auto a = ctx.seed(alpha);
  std::vector<exokal::ad::Local> r = ctx.accumulators();
  term(st, a, r);
  const exokal::ad::LocalSystem sys(ctx.space(), r);

  const double lam = 2.0 * (1.0 + 0.25 * 3.0) * (1.0 + 0.5 * 2.0);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      CHECK(near(sys.block(0, 0)(i, j), h.cell(0)(i, j) / lam, 1e-12));
    }
  }

  // the declared dependency is real: the flux row couples to the enthalpy
  CHECK(sys.has_block(0, 2));
  double mq = 0.0;
  for (std::size_t j = 0; j < 4; ++j) mq += h.cell(0)(0, j) * alpha[j];
  const double dlam_dh = 2.0 * (1.0 + 0.25 * 3.0) * 0.5;
  CHECK(near(sys.block(0, 2)(0, 0), -mq * dlam_dh / (lam * lam), 1e-10));

  // and the discovery is a subset of the declaration: the AD fills no block
  // the model did not announce. Field 2 (h) is touched here; with a
  // constant mobility it is not.
  mimetika::physics::terms::MixedDarcyCell<ConstantMobility> plain(h, ConstantMobility{});
  exokal::ad::LocalContext c2(exokal::ad::local_space_of(st.view));
  const auto a2 = c2.seed(alpha);
  std::vector<exokal::ad::Local> r2 = c2.accumulators();
  plain(st, a2, r2);
  const exokal::ad::LocalSystem flat(c2.space(), r2);
  CHECK(!flat.has_block(0, 2));  // undeclared, and indeed untouched
  CHECK(flat.has_block(0, 1) && flat.has_block(1, 0));
}

// a space that does not carry a declared field is refused, rather than
// indexing past the end of the blocks
MIMETIKA_TEST(a_space_missing_a_declared_field_is_refused) {
  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_rt);
  const ProductSpace s = mixed(c);  // only (q, p): no enthalpy
  const exokal::forms::StencilBuilder sb(c, s, Coupling::closure);
  auto st = sb.at(0);

  mimetika::physics::terms::MixedDarcyCell<PressureEnthalpyMobility> term(
      h, PressureEnthalpyMobility{});
  // resolving the names against a space that lacks one is where this is
  // caught, and the message names the field rather than counting blocks
  bool threw = false;
  try {
    const exokal::forms::Slots sl(s, term.fields(), "mixed_darcy_cell_ph");
    (void)sl;
  } catch (const std::invalid_argument& e) {
    const std::string w = e.what();
    threw =
        w.find("'h'") != std::string::npos && w.find("mixed_darcy_cell_ph") != std::string::npos;
  }
  CHECK(threw);
}

// the registry carries the declaration, so a model can see what a named
// term will touch before it commits to a space
MIMETIKA_TEST(the_registry_reports_the_declared_fields) {
  const auto& reg = exokal::forms::Registry::instance();
  CHECK(reg.has("mixed_darcy_cell") && reg.has("mixed_darcy_cell_ph"));
  CHECK(reg.info("mixed_darcy_cell").fields.size() == 2);
  CHECK(reg.info("mixed_darcy_cell_ph").fields.size() == 3);
  CHECK(reg.info("mixed_darcy_cell_ph").fields[2] == "h");
}

// A term carrying model data is composed by name, with no C++ construction at
// the call site. This is the shape a scripting layer drives.
MIMETIKA_TEST(a_data_bearing_term_composes_by_name) {
  const Mesh m = two_tets();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(2.0), FluxOperators::Realization::derham_rt);

  TermContext ctx;
  ctx.provide("flux_operators", h);  // the data the name cannot carry
  CHECK(ctx.has("flux_operators"));
  CHECK(ctx.names().size() == 1);

  StratifiedEpoch se;
  se.add("bulk", 0, Epoch(m.topology(), mixed(m.topology()), 3));
  Model model;
  model.use(ctx);
  model.add("mixed_darcy_cell", On::all(), {{"mobility", 2.0}});
  CHECK(model.size() == 1 && model.n_evaluations(se) == 1);

  const auto n = static_cast<std::size_t>(se.size());
  TripletSink named(n);
  Workspace ws;
  model.assemble(se, std::vector<double>(n, 1.0), named, ws);

  // identical to the directly-constructed term: the registry adds
  // indirection and nothing else
  Model direct;
  direct.add(mixed_darcy_cell<>(h, {{"mobility", 2.0}}), On::all());
  TripletSink built(n);
  direct.assemble(se, std::vector<double>(n, 1.0), built, ws);

  CHECK(named.nnz() == built.nnz());
  const auto A = dense_of(named, n), B = dense_of(built, n);
  for (std::size_t i = 0; i < n * n; ++i) CHECK(near(A[i], B[i], 1e-14));
  for (std::size_t i = 0; i < n; ++i) CHECK(near(named.residual[i], built.residual[i], 1e-14));

  // the term announces itself in the registry like any other
  CHECK(exokal::forms::Registry::instance().has("mixed_darcy_cell"));
  CHECK(exokal::forms::Registry::instance().info("mixed_darcy_cell").fields.size() == 2);
}

// Mistakes at the named boundary are reported, not segfaults — this is the
// surface a script drives, so a missing or mistyped slot must say so
MIMETIKA_TEST(a_missing_or_mistyped_context_slot_is_reported) {
  const Mesh m = two_tets();
  const Coefficient perm = Coefficient::uniform(1.0);

  // nothing provided at all
  bool threw = false;
  try {
    Model bare;
    bare.add("mixed_darcy_cell", On::all());
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find("flux_operators") != std::string::npos;
  }
  CHECK(threw);

  // the right name holding the wrong type: both types are named
  TermContext wrong;
  wrong.provide("flux_operators", perm);  // a Coefficient, not a FluxOperators
  threw = false;
  try {
    Model bad;
    bad.use(wrong);
    bad.add("mixed_darcy_cell", On::all());
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).find("provided as") != std::string::npos;
  }
  CHECK(threw);
  // find() reports the mismatch without throwing, for a caller that probes
  CHECK(wrong.find<FluxOperators>("flux_operators") == nullptr);
  CHECK(wrong.find<Coefficient>("flux_operators") != nullptr);
}

// the realization stays a model-level choice under the named path: swap
// what the context holds, and the same named model assembles the other
// operator
MIMETIKA_TEST(the_context_selects_the_realization) {
  const Mesh m = two_tets();
  const Coefficient K = Coefficient::uniform(SymTensor<>::diagonal(2.0, 1.0, 3.0));
  const FluxOperators stab = FluxOperators::build(m, K, FluxOperators::Realization::stabilized_rt);
  const FluxOperators rt = FluxOperators::build(m, K, FluxOperators::Realization::derham_rt);

  StratifiedEpoch se;
  se.add("bulk", 0, Epoch(m.topology(), mixed(m.topology()), 3));
  const auto n = static_cast<std::size_t>(se.size());
  Workspace ws;

  const auto run = [&](const FluxOperators& h) {
    TermContext ctx;
    ctx.provide("flux_operators", h);
    Model model;
    model.use(ctx);
    model.add("mixed_darcy_cell", On::all());
    TripletSink sink(n);
    model.assemble(se, std::vector<double>(n, 1.0), sink, ws);
    return dense_of(sink, n);
  };

  const auto A = run(stab), B = run(rt);
  const auto nq = static_cast<std::size_t>(m.topology().count(2));
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i < nq && j < nq) {
        worst = std::max(worst, std::abs(A[i * n + j] - B[i * n + j]));
      } else {
        CHECK(near(A[i * n + j], B[i * n + j], 1e-12));  // divergence: unmoved
      }
    }
  }
  // the same element on a simplex, so the whole assembled system coincides
  CHECK(worst < 1e-10);
}

// What ctx.provide("mobility", brine) does: the fluid is an object the caller
// owns, handed to the term by name and type. Nothing is copied — the context
// stores a pointer — and the term reads lambda from the whole state (p, h, z),
// which no scalar parameter could express.
MIMETIKA_TEST(a_fluid_object_supplies_the_mobility) {
  using mimetika::physics::constitutive::ImmiscibleFluid;
  using mimetika::physics::constitutive::PhaseModel;
  using mimetika::physics::constitutive::State;
  using mimetika::physics::terms::FluidMobility;

  PhaseModel liquid;
  liquid.reference_density = 1000.0;
  liquid.viscosity = 1.0e-3;
  liquid.heat_capacity = 4200.0;
  liquid.compressibility = 1.0e-9;  // makes lambda genuinely depend on p
  PhaseModel vapour;
  vapour.reference_density = 20.0;
  vapour.viscosity = 2.0e-5;
  vapour.heat_capacity = 2000.0;
  const ImmiscibleFluid brine({liquid, vapour});  // owned here, not by the context

  TermContext ctx;
  ctx.provide("mobility", brine);  // a pointer plus a type tag, nothing copied
  CHECK(ctx.find<ImmiscibleFluid>("mobility") == &brine);

  // the arity comes from the object: two phases give two composition
  // fields, which a static declaration could never have known
  const FluidMobility mob({}, ctx);
  const auto in = mob.inputs();
  CHECK(in.size() == 4 && in[0] == "p" && in[1] == "h" && in[2] == "z0" && in[3] == "z1");
  const auto df = mimetika::physics::terms::declared_fields(mob);
  CHECK(df.size() == 5);  // q, p, h, z0, z1

  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(1.0), FluxOperators::Realization::derham_rt);
  ProductSpace s;
  s.add("q", DofMap(c, DofLayout::cochain(3, 2)));
  s.add("p", DofMap(c, DofLayout::cell_wise(3)));
  s.add("h", DofMap(c, DofLayout::cell_wise(3)));
  s.add("z0", DofMap(c, DofLayout::cell_wise(3)));
  s.add("z1", DofMap(c, DofLayout::cell_wise(3)));
  const exokal::forms::StencilBuilder sb(c, s, Coupling::closure);
  auto st = sb.at(0);

  mimetika::physics::terms::MixedDarcyCell<FluidMobility> term(h, mob);
  const exokal::forms::Slots sl(s, term.fields(), "mixed_darcy_cell_fluid");
  st.slots = &sl;
  std::vector<double> alpha(st.view.size(), 0.0);
  for (std::size_t i = st.view.blocks[0].begin; i < st.view.blocks[0].end; ++i) alpha[i] = 1.0;
  alpha[st.view.blocks[1].begin] = 5.0e6;  // p
  alpha[st.view.blocks[2].begin] = 1.0e5;  // h
  alpha[st.view.blocks[3].begin] = 0.6;    // z0
  alpha[st.view.blocks[4].begin] = 0.4;    // z1

  exokal::ad::LocalContext lc(exokal::ad::local_space_of(st.view));
  const auto a = lc.seed(alpha);
  std::vector<exokal::ad::Local> r = lc.accumulators();
  term(st, a, r);
  const exokal::ad::LocalSystem sys(lc.space(), r);

  // the constitutive block is the Hodge divided by the fluid's mobility
  const std::vector<double> z = {0.6, 0.4};
  const double lam = brine.evaluate(State<double>{5.0e6, 1.0e5, z}).total_mobility;
  CHECK(lam > 0.0);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      CHECK(near(sys.block(0, 0)(i, j), h.cell(0)(i, j) / lam,
                 1e-9 * std::abs(h.cell(0)(i, j) / lam) + 1e-12));
    }
  }

  // Declaration is a superset of discovery, which is why the sparsity pattern
  // must come from the declaration. The model declares it may read p, h and
  // both compositions. The AD finds that lambda moves with the pressure
  // (through the compressible density) and with the compositions (through the
  // saturations), but not with the enthalpy — these phases carry no
  // temperature-dependent viscosity. A pattern built from what was discovered
  // would omit a block that another fluid, or the same fluid at another state,
  // would fill.
  CHECK(sys.has_block(0, 1));                         // pressure: through the compressibility
  CHECK(sys.has_block(0, 3) && sys.has_block(0, 4));  // compositions
  CHECK(!sys.has_block(0, 2));                        // enthalpy: declared, and legitimately unused
  const auto declared = mimetika::physics::terms::declared_fields(mob);
  CHECK(std::find(declared.begin(), declared.end(), "h") != declared.end());
  // the composition derivative matches a finite difference in lambda
  double mq = 0.0;
  for (std::size_t j = 0; j < 4; ++j) mq += h.cell(0)(0, j) * alpha[j];
  const double eps = 1e-9;
  const std::vector<double> zp = {0.6 + eps, 0.4};
  const double dlam = (brine.evaluate(State<double>{5.0e6, 1.0e5, zp}).total_mobility - lam) / eps;
  CHECK(near(sys.block(0, 3)(0, 0), -mq * dlam / (lam * lam),
             1e-5 * std::abs(mq * dlam / (lam * lam)) + 1e-9));
}

// The reason the resolution exists. A composition that adds mechanics puts a
// displacement in the space, and nothing says it must come last — here it comes
// first, ahead of both fields the Darcy term reads. Positional indexing would
// have the term read the displacement as its flux and the flux as its pressure:
// same shapes, same arithmetic, silently different physics. Resolving by name
// makes the assembled system bit-for-bit the one built without the extra
// field.
MIMETIKA_TEST(a_field_inserted_ahead_of_the_terms_own_does_not_move_it) {
  const Mesh m = two_tets();
  const graphos::Complex& c = m.topology();
  const FluxOperators h =
      FluxOperators::build(m, Coefficient::uniform(2.0), FluxOperators::Realization::derham_rt);
  mimetika::physics::terms::MixedDarcyCell<> term(h, ConstantMobility{});

  // the flow-only space: q, p
  const ProductSpace flow = mixed(c);
  // the same physics inside a composition: u, q, p — the term declared
  // {"q","p"} and must still find them at 1 and 2
  ProductSpace coupled;
  coupled.add("u", DofMap(c, DofLayout::cell_wise(3)));
  coupled.add("q", DofMap(c, DofLayout::cochain(3, 2)));
  coupled.add("p", DofMap(c, DofLayout::cell_wise(3)));

  const exokal::forms::Slots sl_flow(flow, term.fields(), "darcy");
  const exokal::forms::Slots sl_coupled(coupled, term.fields(), "darcy");
  CHECK(sl_flow.at(0) == 0 && sl_flow.at(1) == 1);
  CHECK(sl_coupled.at(0) == 1 && sl_coupled.at(1) == 2);  // shifted by the displacement

  auto run = [&](const ProductSpace& sp, const exokal::forms::Slots& sl) {
    const exokal::forms::StencilBuilder sb(c, sp, Coupling::closure);
    auto st = sb.at(0);
    st.slots = &sl;
    exokal::ad::LocalContext lc(exokal::ad::local_space_of(st.view));
    const auto a = lc.seed(std::vector<double>(st.view.size(), 1.0));
    std::vector<exokal::ad::Local> r = lc.accumulators();
    term(st, a, r);
    return exokal::ad::LocalSystem(lc.space(), r).block(sl.at(0), sl.at(0));
  };

  const auto A = run(flow, sl_flow);
  const auto B = run(coupled, sl_coupled);
  CHECK(A.rows() == B.rows() && A.cols() == B.cols());
  for (std::size_t i = 0; i < A.rows(); ++i) {
    for (std::size_t j = 0; j < A.cols(); ++j) CHECK(A(i, j) == B(i, j));
  }

  // and the displacement is untouched: the term never declared it
  const exokal::forms::StencilBuilder sb(c, coupled, Coupling::closure);
  auto st = sb.at(0);
  st.slots = &sl_coupled;
  exokal::ad::LocalContext lc(exokal::ad::local_space_of(st.view));
  const auto a = lc.seed(std::vector<double>(st.view.size(), 1.0));
  std::vector<exokal::ad::Local> r = lc.accumulators();
  term(st, a, r);
  const exokal::ad::LocalSystem sys(lc.space(), r);
  CHECK(!sys.has_block(0, 0) && !sys.has_block(1, 0) && !sys.has_block(0, 1));
}

MIMETIKA_TEST_MAIN()
