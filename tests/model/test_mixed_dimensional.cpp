#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/forms/assemble.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "graphos/core/coboundary.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/mixed_dimensional.hpp"
#include "mimetika/physics/package.hpp"
#include "mimetika/physics/contact_interface.hpp"
#include "mimetika/physics/terms/contact_interface.hpp"
#include "mimetika/physics/terms/contact_trace.hpp"

// The topology a fracture needs, on the smallest thing that has one: two cells
// sharing a facet, and that facet is the fracture.
//
// WHAT HAS TO BE TRUE for the mixed contact formulation to be posable at all.
// Its stress lives in H(div; Omega \ Gamma), not H(div; Omega): the normal
// trace is free to jump across Gamma, so the two sides must hold SEPARATE
// traction unknowns on the shared facet. In a mixed space the facet's moments
// are the traction, so that is a statement about the complex, not about the
// forms: the facet has to be two facets.
//
// exokal::mixed_dimensional_mesh does exactly that and nothing here repeats it.
// What is checked is that what it hands back is what the formulation needs:
// two copies carrying one fracture cell, the sides distinguishable, and a
// Simulation that numbers both strata as one system.

using exokal::spaces::DofLayout;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Composition;

namespace {

struct Cut {
  exokal::Mesh mesh;
  exokal::MixedDimensionalMesh md;
};

// Two cells of the unit box, split by the plane x = 1/2.
Cut two_cells(mimetika::mesh::Family family = mimetika::mesh::Family::cartesian) {
  exokal::Mesh mesh = mimetika::mesh::box({2, 1, 1}, 3, family);
  std::vector<graphos::Index> nodes;
  for (graphos::Index v = 0; v < mesh.topology().count(0); ++v) {
    if (std::fabs(mesh.point(v)[0] - 0.5) < 1e-12) nodes.push_back(v);
  }
  exokal::MixedDimensionalMesh md = exokal::mixed_dimensional_mesh(mesh, {nodes});
  return Cut{std::move(mesh), std::move(md)};
}

// The shapes the linear contact formulation asks for, and no equations: what is
// under test is the numbering. On the ambient stratum the traction moments of a
// face, T_h(f), six of them under the strongly symmetric layout; on the
// fracture, the contact traction t_c and the two-sided trace u_c, both cell-wise
// and broken face by face, which is what makes A_c and J block diagonal.
constexpr int kFaceTraction = 6;

class ContactShapes final : public mimetika::physics::Package {
 public:
  // (moments, components) of a facet's tractions: the wrench layout by default,
  // (dim, dim) for the componentwise one
  explicit ContactShapes(int moments = kFaceTraction, int components = 1)
      : nb_(moments), nc_(components) {}

  std::string name() const override { return "ContactShapes"; }

  mimetika::physics::Requirements requirements(int dim, int codim = 0) const override {
    const auto at = [codim](std::string_view b) { return exokal::forms::field_at(b, codim); };
    const int k = nb_ * nc_;
    mimetika::physics::Requirements r;
    if (codim == 0) {
      r.fields.push_back({at("s"), DofLayout::moments(dim, dim - 1, nb_, nc_)});
      r.fields.push_back({at("u"), DofLayout::cell_wise(dim, 1, dim)});
    } else {
      r.fields.push_back({at("t"), DofLayout::cell_wise(dim, 1, k)});
      r.fields.push_back({at("uc"), DofLayout::cell_wise(dim, 1, 2 * k)});
    }
    return r;
  }

  // the one coupling under test; the model evaluates it on the stratum that
  // carries a fracture and nowhere else
  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    model.add("contact_trace_facet", exokal::forms::On::all(), exokal::forms::Params{});
    model.add("contact_compliance_cell", exokal::forms::On::all(), exokal::forms::Params{});
    model.add("contact_jump_cell", exokal::forms::On::all(), exokal::forms::Params{});
  }

 private:
  int nb_{kFaceTraction}, nc_{1};
};

// Full symmetric blocks, with every entry distinct: the chart that would supply
// the true Gram and compliance is not built yet, and what is under test here is
// that the terms place their matrices where the block system says -- an index
// swapped anywhere shows up against these and would not against a diagonal.
mimetika::physics::ContactInterface one_face_interface() {
  mimetika::physics::ContactInterface ci;
  ci.k = kFaceTraction;
  ci.gram.assign(kFaceTraction * kFaceTraction, 0.0);
  ci.compliance.assign(kFaceTraction * kFaceTraction, 0.0);
  for (std::size_t i = 0; i < kFaceTraction; ++i) {
    for (std::size_t j = 0; j < kFaceTraction; ++j) {
      const double sym = static_cast<double>(1 + i + j) + (i == j ? 10.0 : 0.0);
      ci.gram[i * kFaceTraction + j] = sym;
      ci.compliance[i * kFaceTraction + j] = 0.25 * sym + (i == j ? 3.0 : 0.0);
    }
  }
  ci.validate();
  return ci;
}

}  // namespace

// ---- the cut ---------------------------------------------------------------
MIMETIKA_TEST(the_cut_doubles_the_fracture_facet_and_keeps_the_original) {
  const Cut c = two_cells();
  const graphos::Complex& ambient = c.mesh.topology();
  const graphos::Complex& top = c.md.top.mesh.topology();

  CHECK(ambient.count(3) == 2);
  // The cells are untouched. The shared facet becomes THREE: a copy for each
  // side, wired to that side's cell, and the original, which survives detached
  // and is the fracture stratum's own cell. So the count rises by two.
  CHECK(top.count(3) == ambient.count(3));
  CHECK(top.count(2) == ambient.count(2) + 2);

  const auto fractures = c.md.strata_of_codim(1);
  CHECK(fractures.size() == 1);
  CHECK(fractures[0]->dim == 2);
  CHECK(fractures[0]->mesh.topology().count(2) == 1);  // the one shared facet
  CHECK(fractures[0]->n_components == 1);
  std::printf("  ambient %lld facets -> cut %lld, fracture %lld cell(s)\n",
              static_cast<long long>(ambient.count(2)), static_cast<long long>(top.count(2)),
              static_cast<long long>(fractures[0]->mesh.topology().count(2)));
}

// ---- the two sides ---------------------------------------------------------
//
// The side is topological and no geometric test is needed: after the cut each
// copy has exactly one cofacet, so its incidence sign there is +1 on one side
// and -1 on the other. That sign is what a carrier term reads to know which of
// u_c^+ and u_c^- it is pairing against.
MIMETIKA_TEST(the_two_sides_of_the_fracture_are_told_apart_by_incidence) {
  const Cut c = two_cells();
  const graphos::Complex& top = c.md.top.mesh.topology();
  const auto fractures = c.md.strata_of_codim(1);
  const graphos::CoboundaryOperator cob = graphos::coboundary(top, 2);

  const auto& facet_to_ambient = c.md.top.to_ambient.index[2];
  const auto& cell_to_ambient = fractures[0]->to_ambient.index[2];
  const graphos::Index ambient_facet = cell_to_ambient[0];

  std::vector<graphos::Index> sides, cells;
  std::vector<int> signs;
  std::size_t detached = 0;
  for (graphos::Index f = 0; f < top.count(2); ++f) {
    if (facet_to_ambient[static_cast<std::size_t>(f)] != ambient_facet) continue;
    const auto b = cob.offsets[static_cast<std::size_t>(f)];
    const auto e = cob.offsets[static_cast<std::size_t>(f) + 1];
    if (e == b) {
      ++detached;  // the original, which is the fracture stratum's own cell
      continue;
    }
    CHECK(e - b == 1);  // a side sees one cell: that is what cutting means
    sides.push_back(f);
    cells.push_back(cob.indices[static_cast<std::size_t>(b)]);
    signs.push_back(static_cast<int>(cob.signs[static_cast<std::size_t>(b)]));
  }

  CHECK(detached == 1);
  CHECK(sides.size() == 2);
  if (sides.size() == 2) {
    CHECK(cells[0] != cells[1]);         // one side per bulk cell
    CHECK(signs[0] == -signs[1]);        // and opposite orientations
    CHECK(std::abs(signs[0]) == 1);
    std::printf("  fracture cell carried by facets %lld(cell %lld, %+d) and %lld(cell %lld, %+d)\n",
                static_cast<long long>(sides[0]), static_cast<long long>(cells[0]), signs[0],
                static_cast<long long>(sides[1]), static_cast<long long>(cells[1]), signs[1]);
  }
}

// ---- the two strata as one system ------------------------------------------
//
// The fields are declared with the shapes the formulation asks for and with no
// equations behind them: what is under test is the numbering.
MIMETIKA_TEST(a_simulation_numbers_the_carried_stratum_before_its_carrier) {
  const Cut c = two_cells();
  exokal::forms::TermContext ctx;
  const mimetika::physics::ContactInterface ci = one_face_interface();
  ctx.provide("contact_interface", ci);
  Composition comp;
  comp.emplace<ContactShapes>();
  const std::vector<StratumSpec> specs = mimetika::strata_of(c.md, c.mesh);

  CHECK(specs.size() == 2);
  CHECK(specs[0].carries == 1);  // the ambient carries the fracture
  CHECK(specs[1].carries == -1);

  Simulation sim(comp, specs, ctx);
  const auto& epoch = sim.epoch();
  CHECK(epoch.n_strata() == 2);

  const std::size_t ambient = epoch.index_of("ambient");
  const std::size_t fracture = epoch.index_of("fracture");
  CHECK(epoch.entry(ambient).codim == 0);
  CHECK(epoch.entry(fracture).codim == 1);

  // THE CARRIED STRATUM COMES FIRST. Its offset is needed to complete the link
  // the ambient borrows, so it is added before it; the ordering is a
  // consequence of that and is asserted so a change to it is deliberate.
  CHECK(epoch.offset(fracture) == 0);
  CHECK(epoch.offset(ambient) == epoch.stratum(fracture).n_dofs());
  CHECK(sim.n_dofs() == static_cast<std::size_t>(epoch.stratum(ambient).n_dofs() +
                                                 epoch.stratum(fracture).n_dofs()));

  // and the coupling the interface forms are evaluated on exists on the
  // carrier, and only there
  CHECK(epoch.stratum(ambient).carries_substratum());
  CHECK(epoch.stratum(ambient).supports(exokal::forms::Coupling::carrier));
  CHECK(!epoch.stratum(fracture).carries_substratum());

  // THE JUMP IS REPRESENTABLE, which is what the cut was for: the stress
  // carries its face moments on every facet, and the fracture facet is two, so
  // the two sides hold independent tractions. Against the uncut complex that is
  // one face's worth of extra unknowns.
  const auto& space = epoch.stratum(ambient).space();
  const exokal::spaces::DofMap& stress = space.map(space.index_of("s_0"));
  CHECK(stress.size() == kFaceTraction * c.md.top.mesh.topology().count(2));
  CHECK(stress.size() == kFaceTraction * (c.mesh.topology().count(2) + 2));

  // and the fracture's own unknowns are the contact traction and the two traces
  const auto& fspace = epoch.stratum(fracture).space();
  CHECK(fspace.map(fspace.index_of("t_1")).size() == kFaceTraction);
  CHECK(fspace.map(fspace.index_of("uc_1")).size() == 2 * kFaceTraction);

  std::printf("  dofs: fracture %lld + ambient %lld = %zu\n",
              static_cast<long long>(epoch.stratum(fracture).n_dofs()),
              static_cast<long long>(epoch.stratum(ambient).n_dofs()), sim.n_dofs());
}

// ---- the trace coupling ----------------------------------------------------
//
// t(tau, v_c) assembled, and what it must be: an identity block per side,
// signed by that side's incidence, appearing twice so that T sits in one row
// and T^T in the other. Nothing else, and nothing at all on the detached copy
// of the fracture facet.
MIMETIKA_TEST(the_trace_coupling_is_the_signed_identity_on_each_side) {
  const Cut c = two_cells();
  exokal::forms::TermContext ctx;
  const mimetika::physics::ContactInterface ci = one_face_interface();
  ctx.provide("contact_interface", ci);
  Composition comp;
  comp.emplace<ContactShapes>();
  Simulation sim(comp, mimetika::strata_of(c.md, c.mesh), ctx);

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  std::map<std::pair<graphos::Index, graphos::Index>, double> entry;
  for (std::size_t i = 0; i < jac.nnz(); ++i) {
    if (jac.value[i] != 0.0) entry[{jac.row[i], jac.col[i]}] += jac.value[i];
  }

  const auto& epoch = sim.epoch();
  const std::size_t ambient = epoch.index_of("ambient");
  const std::size_t fracture = epoch.index_of("fracture");
  const auto& aspace = epoch.stratum(ambient).space();
  const auto& fspace = epoch.stratum(fracture).space();
  const exokal::spaces::DofMap& smap = aspace.map(aspace.index_of("s_0"));
  const exokal::spaces::DofMap& umap = fspace.map(fspace.index_of("uc_1"));
  const graphos::Index s_base = epoch.offset(ambient) + aspace.offset(aspace.index_of("s_0"));
  const graphos::Index u_base = epoch.offset(fracture) + fspace.offset(fspace.index_of("uc_1"));

  // the sides, as the previous test found them
  const graphos::Complex& top = c.md.top.mesh.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(top, 2);
  const auto& facet_to_ambient = c.md.top.to_ambient.index[2];
  const graphos::Index ambient_facet = c.md.strata_of_codim(1)[0]->to_ambient.index[2][0];

  std::size_t checked = 0, detached_entries = 0;
  for (graphos::Index f = 0; f < top.count(2); ++f) {
    if (facet_to_ambient[static_cast<std::size_t>(f)] != ambient_facet) continue;
    const auto b = cob.offsets[static_cast<std::size_t>(f)];
    const bool free_standing = cob.offsets[static_cast<std::size_t>(f) + 1] == b;
    const double side = free_standing ? 0.0 : static_cast<double>(cob.signs[static_cast<std::size_t>(b)]);
    const std::size_t half = free_standing ? 0 : (side > 0.0 ? 0 : kFaceTraction);

    for (int j = 0; j < kFaceTraction; ++j) {
      const graphos::Index row = s_base + smap.global(2, f, j, 0);
      if (free_standing) {
        // it carries the fracture cell but belongs to no bulk cell: no traction
        // to pair, so the term must leave it alone
        for (int m = 0; m < 2 * kFaceTraction; ++m) {
          const graphos::Index col = u_base + umap.global(2, 0, 0, m);
          if (entry.count({row, col}) != 0) ++detached_entries;
        }
        continue;
      }
      const graphos::Index col = u_base + umap.global(2, 0, 0, static_cast<int>(half) + j);
      const auto forward = entry.find({row, col});
      const auto backward = entry.find({col, row});
      CHECK(forward != entry.end());
      CHECK(backward != entry.end());
      if (forward != entry.end() && backward != entry.end()) {
        CHECK(std::fabs(forward->second + side) < 1e-14);   // -incidence, exactly
        CHECK(std::fabs(backward->second + side) < 1e-14);  // and its transpose
        ++checked;
      }
      // the other half of the trace belongs to the other side
      const graphos::Index other =
          u_base + umap.global(2, 0, 0, static_cast<int>(kFaceTraction - half) + j);
      CHECK(entry.count({row, other}) == 0);
    }
  }

  CHECK(detached_entries == 0);
  CHECK(checked == 2 * kFaceTraction);  // both sides, every traction moment

  // and the coupling is exactly the two signed identities and their
  // transposes: counted over the stress-to-trace entries alone, since the
  // interface rows are assembled here too
  const graphos::Index s_end = s_base + smap.size();
  const graphos::Index u_end = u_base + umap.size();
  const auto in = [](graphos::Index i, graphos::Index lo, graphos::Index hi) {
    return i >= lo && i < hi;
  };
  std::size_t coupling = 0;
  for (const auto& [rc, v] : entry) {
    (void)v;
    if ((in(rc.first, s_base, s_end) && in(rc.second, u_base, u_end)) ||
        (in(rc.first, u_base, u_end) && in(rc.second, s_base, s_end))) {
      ++coupling;
    }
  }
  CHECK(coupling == static_cast<std::size_t>(4 * kFaceTraction));
  std::printf("  %zu stress-to-trace entries of %zu, %zu pairs checked, %zu on the detached copy\n",
              coupling, entry.size(), checked, detached_entries);
}

// ---- the interface rows ----------------------------------------------------
//
// A_c and J, assembled, and the property that ties all three interface terms
// together: each is a bilinear form appearing in two rows, so what they add is
// symmetric. With no bulk terms in this composition the assembled operator IS
// the interface, and its symmetry is exact rather than approximate.
MIMETIKA_TEST(the_interface_rows_are_the_compliance_and_the_signed_jump) {
  const Cut c = two_cells();
  const mimetika::physics::ContactInterface ci = one_face_interface();
  exokal::forms::TermContext ctx;
  ctx.provide("contact_interface", ci);
  Composition comp;
  comp.emplace<ContactShapes>();
  Simulation sim(comp, mimetika::strata_of(c.md, c.mesh), ctx);

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  std::map<std::pair<graphos::Index, graphos::Index>, double> entry;
  for (std::size_t i = 0; i < jac.nnz(); ++i) {
    if (jac.value[i] != 0.0) entry[{jac.row[i], jac.col[i]}] += jac.value[i];
  }

  const auto& epoch = sim.epoch();
  const std::size_t fracture = epoch.index_of("fracture");
  const auto& fspace = epoch.stratum(fracture).space();
  const exokal::spaces::DofMap& tmap = fspace.map(fspace.index_of("t_1"));
  const exokal::spaces::DofMap& umap = fspace.map(fspace.index_of("uc_1"));
  const graphos::Index t_base = epoch.offset(fracture) + fspace.offset(fspace.index_of("t_1"));
  const graphos::Index u_base = epoch.offset(fracture) + fspace.offset(fspace.index_of("uc_1"));

  const auto at = [&](graphos::Index r, graphos::Index col) {
    const auto it = entry.find({r, col});
    return it == entry.end() ? 0.0 : it->second;
  };

  std::size_t compliance_checked = 0, jump_checked = 0;
  for (int i = 0; i < kFaceTraction; ++i) {
    const graphos::Index ti = t_base + tmap.global(2, 0, 0, i);
    for (int j = 0; j < kFaceTraction; ++j) {
      const graphos::Index tj = t_base + tmap.global(2, 0, 0, j);
      const graphos::Index up = u_base + umap.global(2, 0, 0, j);
      const graphos::Index um = u_base + umap.global(2, 0, 0, kFaceTraction + j);
      const double g = ci.gram[static_cast<std::size_t>(i) * kFaceTraction + j];
      const double A = ci.compliance[static_cast<std::size_t>(i) * kFaceTraction + j];

      CHECK(std::fabs(at(ti, tj) - A) < 1e-14);  // A_c, in the compliance row
      ++compliance_checked;

      // J = -G[I, -I]: the jump opposes the traction on the + side and
      // supports it on the -, in the compliance row and its transpose alike
      CHECK(std::fabs(at(ti, up) + g) < 1e-14);
      CHECK(std::fabs(at(ti, um) - g) < 1e-14);
      CHECK(std::fabs(at(up, ti) + g) < 1e-14);
      CHECK(std::fabs(at(um, ti) - g) < 1e-14);
      ++jump_checked;
    }
  }
  CHECK(compliance_checked == kFaceTraction * kFaceTraction);
  CHECK(jump_checked == kFaceTraction * kFaceTraction);

  // EVERY INTERFACE TERM IS A FORM IN TWO ROWS, so the whole of what this
  // composition assembles is symmetric. A sign taken once instead of twice, or
  // a transpose written the same way round as its partner, breaks this.
  double worst = 0.0, scale = 0.0;
  for (const auto& [rc, v] : entry) {
    worst = std::max(worst, std::fabs(v - at(rc.second, rc.first)));
    scale = std::max(scale, std::fabs(v));
  }
  CHECK(scale > 0.0);
  CHECK(worst == 0.0);
  std::printf("  %zu entries, asymmetry %.1e of %.1e\n", entry.size(), worst, scale);
}

// ---- the chart, from the ambient facet -------------------------------------
//
// G and A_c built rather than supplied, from the operators of the cut mesh, and
// checked against what the law means. The facet is the unit square at x = 1/2,
// so |f| = 1 and every number below is the compliance itself.
namespace {

struct Built {
  exokal::hodge::StressOperators ops;
  exokal::spaces::ProductSpace space;
};

Built operators_for(const Cut& c, int moments, int components,
                    exokal::hodge::StressOperators::Realization how,
                    exokal::hodge::StressOperators::Formulation form) {
  Composition comp;
  comp.emplace<ContactShapes>(moments, components);
  return Built{exokal::hodge::StressOperators::build(c.md.top.mesh, 3, 1.0, 1.0, how, form),
               comp.space(c.md.top.mesh.topology(), 3, 0)};
}

}  // namespace

MIMETIKA_TEST(the_wrench_chart_gives_a_diagonal_compliance) {
  const Cut c = two_cells();
  const Built b = operators_for(c, kFaceTraction, 1,
                                exokal::hodge::StressOperators::Realization::stabilized_vem,
                                exokal::hodge::StressOperators::Formulation::strong_symmetry);
  const auto chart = mimetika::physics::interface_chart(c.md, b.space, "s_0", b.ops, 3);
  CHECK(chart.k == kFaceTraction);

  const double cn = 4.0, ct = 25.0;
  const auto ci = mimetika::physics::linear_contact(chart, cn, ct);

  // the modes are orthonormal up to |f|, and each is purely normal or purely
  // tangential, so the compliance is diagonal: 1/c_T on the three tangential
  // modes and 1/c_n on the three normal ones
  double worst_g = 0.0, worst_a = 0.0;
  for (std::size_t i = 0; i < chart.k; ++i) {
    for (std::size_t j = 0; j < chart.k; ++j) {
      const double g = ci.gram[i * chart.k + j];
      const double a = ci.compliance[i * chart.k + j];
      const double g_want = i == j ? 1.0 : 0.0;                      // |f| = 1
      const double a_want = i != j ? 0.0 : (i < 3 ? 1.0 / ct : 1.0 / cn);
      worst_g = std::max(worst_g, std::fabs(g - g_want));
      worst_a = std::max(worst_a, std::fabs(a - a_want));
    }
  }
  std::printf("  wrench: |G - |f|I| %.1e, |A_c - diag(1/cT, 1/cn)| %.1e\n", worst_g, worst_a);
  CHECK(worst_g < 1e-12);
  CHECK(worst_a < 1e-12);
}

MIMETIKA_TEST(the_componentwise_chart_gives_the_compliance_of_a_constant_traction) {
  const Cut c = two_cells();
  const Built b = operators_for(c, 3, 3, exokal::hodge::StressOperators::Realization::derham_bdm,
                                exokal::hodge::StressOperators::Formulation::weak_symmetry);
  const auto chart = mimetika::physics::interface_chart(c.md, b.space, "s_0", b.ops, 3);
  CHECK(chart.k == 9);

  const double cn = 4.0, ct = 25.0;
  const auto ci = mimetika::physics::linear_contact(chart, cn, ct);

  // A CONSTANT TRACTION, which is the constant chart mode alone: chi_0 = 1, so
  // its coefficients are the vector itself. The quadratic form is then the
  // compliance of that direction times |f|, whatever basis was used -- which is
  // the law, stated without reference to the chart.
  const auto quadratic = [&](const std::vector<double>& m, const double a[3]) {
    std::vector<double> x(chart.k, 0.0);
    for (int k = 0; k < 3; ++k) x[static_cast<std::size_t>(k)] = a[k];  // b = 0, component k
    double q = 0.0;
    for (std::size_t i = 0; i < chart.k; ++i) {
      for (std::size_t j = 0; j < chart.k; ++j) q += x[i] * m[i * chart.k + j] * x[j];
    }
    return q;
  };
  const double normal[3] = {1.0, 0.0, 0.0};     // the facet at x = 1/2
  const double tangent[3] = {0.0, 1.0, 0.0};
  std::printf("  componentwise: normal %.6f (want %.6f), tangential %.6f (want %.6f), G %.6f\n",
              quadratic(ci.compliance, normal), 1.0 / cn, quadratic(ci.compliance, tangent),
              1.0 / ct, quadratic(ci.gram, normal));
  CHECK(std::fabs(quadratic(ci.compliance, normal) - 1.0 / cn) < 1e-12);
  CHECK(std::fabs(quadratic(ci.compliance, tangent) - 1.0 / ct) < 1e-12);
  CHECK(std::fabs(quadratic(ci.gram, normal) - 1.0) < 1e-12);  // |f| = 1
}

// ---- and the compliance is checked, not trusted -----------------------------
//
// A_c is a compliance: symmetric, positive semi-definite, ZERO ADMITTED. The
// zero case is the bonded interface, which is the configuration the patch test
// in tests/model/test_contact_patch.cpp runs on, so a definiteness test here
// would reject exactly what validates the formulation. Everything else must be
// positive, and the check is at construction rather than left to a solver --
// an indefinite A_c produces an answer, just not to the question asked.
MIMETIKA_TEST(the_compliance_must_be_positive_and_zero_is_the_bonded_case) {
  const Cut c = two_cells();
  const Built b = operators_for(c, kFaceTraction, 1,
                                exokal::hodge::StressOperators::Realization::stabilized_vem,
                                exokal::hodge::StressOperators::Formulation::strong_symmetry);
  const auto chart = mimetika::physics::interface_chart(c.md, b.space, "s_0", b.ops, 3);

  // the two admissible ends of the family
  mimetika::physics::linear_contact(chart, 4.0, 25.0).validate();
  const auto bonded = mimetika::physics::rigid_contact(chart);
  bonded.validate();
  double worst = 0.0;
  for (const double a : bonded.compliance) worst = std::max(worst, std::fabs(a));
  CHECK(worst == 0.0);  // bonded IS A_c = 0, not a large stiffness standing in for it

  const auto ok = mimetika::physics::linear_contact(chart, 4.0, 25.0);
  const auto broken = [&](auto mutate) {
    mimetika::physics::ContactInterface bad = ok;
    mutate(bad);
    return bad;
  };
  // a sign error on the whole law: compression would open the fracture
  CHECK_THROWS(broken([](auto& bad) {
                 for (double& a : bad.compliance) a = -a;
               }).validate());
  // one negative direction, the rest untouched: the normal mode alone
  CHECK_THROWS(broken([](auto& bad) {
                 bad.compliance[3 * kFaceTraction + 3] *= -3.0;
               }).validate());
  // not symmetric, so not the matrix of any inner product
  CHECK_THROWS(broken([](auto& bad) {
                 bad.compliance[1 * kFaceTraction + 2] += 1.0;
               }).validate());
  // and a degenerate chart, where J u_c = 0 would no longer mean u_c^+ = u_c^-
  CHECK_THROWS(broken([](auto& bad) {
                 for (std::size_t j = 0; j < kFaceTraction; ++j) {
                   bad.gram[2 * kFaceTraction + j] = 0.0;
                   bad.gram[j * kFaceTraction + 2] = 0.0;
                 }
               }).validate());
  std::printf("  bonded A_c is exactly zero; four indefinite or degenerate blocks refused\n");
}

MIMETIKA_TEST_MAIN()
