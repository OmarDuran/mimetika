#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"
#include "mimetika/solver/linear.hpp"

// SINGLE-PHASE FLOW, STATED AS DATA -- the poroelastic problem with one physics
// instead of two.
//
//     div q = 0,   q = -(K/mu) grad p
//
// It exists for the same reason PoroelasticModel does, and it earns its
// keep twice over. It is the configuration a flow-only benchmark needs, and it
// is the SMALLEST problem that exercises the flux space, the natural pressure
// datum and the strong flux condition without any mechanics in the way. When a
// poroelastic answer is wrong, this is what says whether the flow half is.
//
// Both of its closed forms are one line and hold in any dimension:
//
//     a column   p linear between the two ends          the Laplace solution
//     an annulus p = p_a + (p_b - p_a) ln(r/a)/ln(b/a)  Dupuit
//
// so a discretization that claims a cell type can be asked to prove it.

namespace mimetika {

class SinglePhaseModel {
 public:
  // WHICH DISCRETE HODGE, chosen here rather than fixed. exokal offers two de
  // Rham realizations of the flux star and one stabilized polytopal product,
  // and they are different discretizations of the same equations:
  //
  //   derham      d moments per facet. On a simplex this IS BDM_1 -- 3 edges
  //               x 2 in the plane, 4 facets x 3 in space -- unisolvent with
  //               no enrichment. On a polytope the moments are completed with
  //               div-free curl modes. The space a coupled poroelastic model
  //               needs, because the stress space it pairs with carries d^2.
  //   derham_rt   one flux per facet: RT_0, the minimal de Rham pair, whose
  //               radial mode x - x_E is what lets div reach P_0 at all.
  //               SIMPLICES IN SPACE ONLY today; it refuses anything else
  //               rather than stabilizing it.
  //   stabilized  one flux per facet on any polytope, consistency plus a
  //               stabilization -- not a de Rham construction.
  //
  // The realization decides the SPACE as well as the star, and the two must
  // agree on moments per facet or every index after the first facet is wrong.
  // So both are derived from this one field and neither is stated twice.
  using Realization = exokal::hodge::FluxOperators::Realization;

  SinglePhaseModel(const exokal::Mesh& mesh, int cell_dim, double mobility = 1.0,
                   Realization how = Realization::derham_bdm)
      : mesh_(&mesh), dim_(cell_dim), mobility_(mobility), how_(how) {}

  Realization realization() const { return how_; }
  const char* realization_name() const { return exokal::hodge::FluxOperators::name(how_); }
  int moments_per_facet() const {
    return exokal::hodge::FluxOperators::moments_per_facet(how_, dim_);
  }

  FlowBoundary& flow() { return flow_; }
  const FlowBoundary& flow() const { return flow_; }

  int dim() const { return dim_; }
  const Simulation& simulation() const { return *sim_; }
  const solver::SparseSystem& system() const { return system_; }
  std::size_t n_cells() const { return n_cells_; }

  void build() {
    const graphos::Complex& c = mesh_->topology();
    // the K-independent mode selection, and only the P_1 realization needs it
    if (how_ == Realization::derham_bdm) {
      geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    }
    flux_ = exokal::hodge::FluxOperators::build(
        *mesh_, dim_, exokal::hodge::Coefficient::uniform(mobility_), how_,
        how_ == Realization::derham_bdm ? &geometry_ : nullptr);
    pressure_data_ = BoundaryData(static_cast<std::size_t>(c.count(dim_ - 1)));
    const bool any_pressure = flow_.fill_pressure(pressure_data_, *mesh_, dim_);

    ctx_.provide("flux_operators", flux_);
    ctx_.provide("boundary_pressure", pressure_data_);

    physics::ModelOptions o;
    o.flux_moments = moments_per_facet();
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("single_phase_flow", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    if (any_pressure) sim_->model().add("prescribed_pressure", exokal::forms::On::all(), {});

    const auto& sp = sim_->epoch().stratum(0).space();
    flow_.resolve(*mesh_, dim_, sp);
    flow_.impose(sim_->constraints());
    sim_->freeze_constraints();

    exokal::forms::TripletSink jac(sim_->n_dofs());
    sim_->jacobian(jac);
    system_ = solver::SparseSystem::from(jac);

    // THE STEADY RIGHT-HAND SIDE. Everything the terms contribute that does not
    // depend on the unknowns is a load, and the residual at the zero state is
    // exactly minus that load -- so it is read off rather than re-derived. The
    // strongly imposed rows carry their own datum, scaled as their equation is.
    std::vector<double> r;
    sim_->state().assign(sim_->n_dofs(), 0.0);
    sim_->residual(r);
    rhs_.assign(sim_->n_dofs(), 0.0);
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      rhs_[i] = sim_->constraints().pinned(i)
                    ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i)
                    : -r[i];
    }
    p_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  const std::vector<double>& rhs() const { return rhs_; }

  void accept(std::vector<double> x) { state_ = std::move(x); }
  double cell_pressure(Index e) const { return state_[p_offset_ + static_cast<std::size_t>(e)]; }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  double mobility_;
  Realization how_;
  FlowBoundary flow_;

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::FluxOperators flux_;
  BoundaryData pressure_data_{0};
  exokal::forms::TermContext ctx_;
  std::unique_ptr<Simulation> sim_;
  solver::SparseSystem system_;
  std::vector<double> rhs_, state_;
  std::size_t p_offset_{0}, n_cells_{0};
};

}  // namespace mimetika
