#pragma once

#include <array>
#include <functional>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/partition.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"
#include "mimetika/linear_solver/linear.hpp"

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
  const exokal::Mesh& mesh() const { return *mesh_; }
  bool built() const { return sim_ != nullptr; }

  // Dereferencing an unbuilt model is a null read, and from Python that is an
  // abort rather than an exception -- so it is refused by name instead.
  // NON-CONST, for the one caller that configures the simulation rather than
  // reading it: the partition, which tells it which sites to assemble.
  Simulation& simulation() {
    if (!sim_) {
      throw std::logic_error("SinglePhaseModel: not built yet; call build() or solve() first");
    }
    return *sim_;
  }

  const Simulation& simulation() const {
    if (sim_ == nullptr) {
      throw std::logic_error("SinglePhaseModel: not built yet; call build() or solve() first");
    }
    return *sim_;
  }
  const solver::SparseSystem& system() const { return system_; }
  std::size_t n_cells() const { return n_cells_; }

  // SHARE THIS MODEL OUT over `n_ranks` processes. Nothing happens until
  // build(): the partition needs the space, and the space is built there.
  // `reduce` sums a vector across the processes, and is used once per
  // assembly, on the scales of the constrained rows alone.
  void distribute_over(int n_ranks, int rank, std::function<void(std::vector<double>&)> reduce) {
    n_ranks_ = n_ranks;
    rank_ = rank;
    reduce_ = std::move(reduce);
  }

  const Distribution& distribution() const { return distribution_; }

  void build() {
    const graphos::Complex& c = mesh_->topology();
    // THE PARTITION COMES FIRST, because the products below are per cell and
    // are the bulk of the work: a process builds its own and no others.
    if (n_ranks_ > 1) distribution_ = partition_cells(*mesh_, dim_, n_ranks_, rank_);
    const std::vector<char>* only =
        distribution_.assembled_cells.empty() ? nullptr : &distribution_.assembled_cells;
    // the K-independent mode selection, and only the P_1 realization needs it
    if (how_ == Realization::derham_bdm) {
      geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    }
    flux_ = exokal::hodge::FluxOperators::build(
        *mesh_, dim_, exokal::hodge::Coefficient::uniform(mobility_), how_,
        how_ == Realization::derham_bdm ? &geometry_ : nullptr,
        exokal::hodge::default_enrichment_degree,
        exokal::hodge::default_max_facets, only);
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

    // THE PARTITION, APPLIED WHERE THE SPACE EXISTS. Asked for before the
    // build, because the assembly below is the thing it divides.
    if (n_ranks_ > 1) {
      add_dof_ownership(distribution_, *mesh_, dim_, sim_->epoch(), n_ranks_, rank_);
      // ASSEMBLE the halo as well, WRITE only what this process owns: the
      // rows it owns are then complete without a single message.
      sim_->distribute_over(distribution_.assembled_cells, distribution_.assembled_facets,
                            distribution_.owned_dofs, reduce_);
    }

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
      // -r is a CONTRIBUTION and is summed across the processes; a constrained
      // row is a REPLACEMENT and is written by whichever process owns it, so
      // the others leave it at zero rather than adding a copy of it.
      const bool mine = sim_->owned_dofs().empty() || sim_->owned_dofs()[i] != 0;
      rhs_[i] = sim_->constraints().pinned(i)
                    ? (mine ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i) : 0.0)
                    : -r[i];
    }
    p_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
    q_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("q_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  const std::vector<double>& rhs() const { return rhs_; }

  void accept(std::vector<double> x) { state_ = std::move(x); }
  double cell_pressure(Index e) const { return state_[p_offset_ + static_cast<std::size_t>(e)]; }

  // THE CELL'S FLUX VECTOR, FROM THE FACET FLUXES THAT ARE THE UNKNOWNS.
  //
  // The unknown on a facet is the moment int_f q.n against the CANONICAL
  // normal, so a cell reads it through its own incidence to get an outward
  // flux, and
  //
  //     q_E = |E|^-1 sum_f (q_f n_f) (x_f - x_E)
  //
  // is the cell average of any field whose divergence is constant -- the
  // divergence theorem applied to q (x - x_E)^T, whose divergence is
  // q + (div q)(x - x_E). It is the flow's cell_stress: the same lever arm
  // against the same leading moment, with a scalar in place of a traction.
  //
  // The FACETS COME FROM THE TOPOLOGY rather than from the product, because
  // FluxOperators::cell is the dense inner product alone -- it carries no face
  // list, where StressOperators::Cell does.
  std::array<double, 3> cell_flux(Index e) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mq = sp.map(sp.index_of("q_0"));
    const exokal::Point xE = exokal::centroid(*mesh_, dim_, e);
    const double volume = exokal::measure(*mesh_, dim_, e);
    const graphos::BoundaryOperator& d = mesh_->topology().boundary(dim_);
    std::array<double, 3> out{};
    for (Index k = d.offsets[static_cast<std::size_t>(e)];
         k < d.offsets[static_cast<std::size_t>(e) + 1]; ++k) {
      const Index f = d.indices[static_cast<std::size_t>(k)];
      const FacetFrame fr = FacetFrame::of(*mesh_, dim_, e, f);
      const exokal::Point xf = exokal::centroid(*mesh_, dim_ - 1, f);
      const double q =
          fr.incidence *
          state_[q_offset_ + static_cast<std::size_t>(mq.global(dim_ - 1, f, 0, 0))];
      for (int j = 0; j < dim_; ++j) {
        out[static_cast<std::size_t>(j)] += q * (xf[static_cast<std::size_t>(j)] -
                                                 xE[static_cast<std::size_t>(j)]);
      }
    }
    for (int j = 0; j < dim_; ++j) out[static_cast<std::size_t>(j)] /= volume;
    return out;
  }

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
  // the partition, requested before the build and applied inside it
  int n_ranks_{1}, rank_{0};
  std::function<void(std::vector<double>&)> reduce_;
  Distribution distribution_;
  std::vector<double> rhs_, state_;
  std::size_t p_offset_{0}, q_offset_{0}, n_cells_{0};
};

}  // namespace mimetika
