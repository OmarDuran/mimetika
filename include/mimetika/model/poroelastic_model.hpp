#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"
#include "mimetika/linear_solver/linear.hpp"

// A poroelastic problem, stated as data.
//
// Terzaghi's column and Coussy's borehole are one model -- Biot poroelasticity,
// the mimetic-AFW-BDM de Rham discretization -- given two configurations:
//
//                    consolidation              borehole
//   domain           a column, 2D or 3D         a quarter annulus, plane strain
//   material         drained, alpha, 1/M        the same five numbers
//   initial state    unloaded                   sigma = -w I, p = p0
//   boundary         traction on the loaded     traction on the wall and the
//                    face, free slip and        far field, free slip and sealed
//                    sealed elsewhere,          on the symmetry planes,
//                    drained top (natural)      prescribed pressure on both
//                                               radial boundaries (natural)
//
// Everything that differs is in this struct; nothing that differs is in a
// term, a package or a model. That is the claim the catalogue makes.
//
// The conditions are forms, so a curved wall is not a special case. The
// borehole's inner boundary has a different normal on every facet, and
// "the radial stress is -p1 there" is one statement about all of them.

namespace mimetika {

// The five numbers a Biot medium needs, plus what the discretization derives
// from them. All are d-dependent through the compliance, so the dimension is
// asked for rather than assumed -- plane strain is not three dimensions with a
// direction ignored, it is d = 2 with its own compliance coefficient.
struct PoroelasticMaterial {
  double shear{1.0};                 // mu
  double lame{1.0};                  // lambda
  double biot{1.0};                  // b (or alpha)
  double inverse_biot_modulus{0.0};  // 1/M
  double mobility{1.0};              // k / mu_f

  double poisson() const { return lame / (2.0 * (lame + shear)); }

  // tr(C^{-1} T)/tr(T): finite at nu = 1/2, where it is zero
  double volumetric_compliance(int d) const {
    const double nu = poisson();
    return (1.0 - 2.0 * nu) / (2.0 * shear * (1.0 - 2.0 * nu + d * nu));
  }

  // the sigma-form storage the mass balance carries: S = d b^2/(dK) + 1/M
  double storage(int d) const {
    return d * biot * biot * volumetric_compliance(d) + inverse_biot_modulus;
  }

  double oedometer_modulus() const { return lame + 2.0 * shear; }

  // Coussy's diffusivity, Eq. (5.22): c_f = k M (K + 4mu/3)/(K_u + 4mu/3),
  // with K_u = K + b^2 M the undrained bulk modulus. Reported rather than
  // used -- it sets the clock a benchmark is read against.
  double diffusivity() const {
    if (!(inverse_biot_modulus > 0.0)) {
      // an incompressible fluid: the uniaxial form is what applies
      return mobility / (biot * biot / oedometer_modulus());
    }
    const double M = 1.0 / inverse_biot_modulus;
    const double K = lame + 2.0 * shear / 3.0;
    const double Ku = K + biot * biot * M;
    return mobility * M * (K + 4.0 * shear / 3.0) / (Ku + 4.0 * shear / 3.0);
  }
};

// The two physics carry their own conditions. See boundary_conditions.hpp:
// a face may be traction-loaded and sealed, or roller-supported and drained,
// and the mechanical and hydraulic sets need not coincide.

class PoroelasticModel {
 public:
  // Which scalar de Rham layer the whole model stands on -- one choice, not
  // two.
  //
  // A stress space is d copies of a flux space plus a rank-one trace, and the
  // Biot coupling pairs the discrete trace of the stress against the pressure
  // of the same cell. Copying BDM for the stress while giving the flux RT would
  // pair a traction that varies over a facet with a flux that does not, on
  // either side of every interior facet: not de Rham compatible. So the layer
  // is chosen once and both spaces follow it.
  //
  //   bdm             d moments per facet: flux d per facet, stress d^2, both
  //                   de Rham -- derham_bdm under derham_bdm. The
  //                   mimetic-AFW-BDM formulation. Any cell type, either
  //                   dimension.
  //   bdm_stabilized  the same layout and the same flux, with the stress star
  //                   realized as stabilized_bdm: the full linear tensor
  //                   reconstruction, stabilized on ker(N^T) where a polytope
  //                   leaves one. On a simplex mesh it coincides with `bdm`
  //                   cell by cell, since the stabilization vanishes and both
  //                   reduce to the conforming AFW element.
  //   rt              one per facet: flux 1, stress d. d copies of the minimal
  //                   de Rham pair; sound as a product, and not an element --
  //                   its weak-symmetry inf-sup degenerates (see
  //                   test_dimensions).
  enum class Layer { bdm, bdm_stabilized, rt };

  PoroelasticModel(const exokal::Mesh& mesh, int cell_dim, PoroelasticMaterial material, double dt,
                   Layer layer = Layer::bdm)
      : mesh_(&mesh), dim_(cell_dim), material_(material), dt_(dt), layer_(layer) {}

  Layer layer() const { return layer_; }
  const char* layer_name() const {
    switch (layer_) {
      case Layer::bdm: return "bdm";
      case Layer::bdm_stabilized: return "bdm_stabilized";
      case Layer::rt: return "rt";
    }
    return "?";
  }
  int moments_per_facet() const { return layer_ == Layer::rt ? 1 : dim_; }

  MechanicsBoundary& mechanics() { return mechanics_; }
  FlowBoundary& flow() { return flow_; }
  const MechanicsBoundary& mechanics() const { return mechanics_; }
  const FlowBoundary& flow() const { return flow_; }

  void initial_stress(const std::array<double, 9>& s) { sigma0_ = s; }
  void initial_pressure(double p) { p0_ = p; }

  const PoroelasticMaterial& material() const { return material_; }
  int dim() const { return dim_; }
  double dt() const { return dt_; }
  const Simulation& simulation() const { return *sim_; }
  Simulation& simulation() { return *sim_; }
  const std::vector<double>& state() const { return state_; }

  // ---- assemble the problem this configuration describes ----------------
  void build() {
    const graphos::Complex& c = mesh_->topology();
    // one de Rham selection for both products; it does not know the material
    const bool bdm = layer_ != Layer::rt;  // both bdm layers share the layout and the flux
    // the K-independent mode selection, shared by the de Rham flux and the
    // derham_bdm stress; the stabilized stress builds without it
    if (bdm) geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    const auto stress_how =
        layer_ == Layer::bdm ? exokal::hodge::StressOperators::Realization::derham_bdm
                             : (layer_ == Layer::bdm_stabilized
                                    ? exokal::hodge::StressOperators::Realization::stabilized_bdm
                                    : exokal::hodge::StressOperators::Realization::derham_rt);
    stress_ = exokal::hodge::StressOperators::build(
        *mesh_, dim_, material_.shear, material_.lame, stress_how,
        exokal::hodge::StressOperators::Formulation::weak_symmetry,
        layer_ == Layer::bdm ? &geometry_ : nullptr);
    // backward Euler without stepping machinery: the flux over a step is
    // q~ = dt q, which is the Darcy mobility set to dt
    flux_ = exokal::hodge::FluxOperators::build(
        *mesh_, dim_, exokal::hodge::Coefficient::uniform(material_.mobility),
        bdm ? exokal::hodge::FluxOperators::Realization::derham_bdm
            : exokal::hodge::FluxOperators::Realization::derham_rt,
        bdm ? &geometry_ : nullptr);
    pressure_data_ = BoundaryData(static_cast<std::size_t>(c.count(dim_ - 1)));
    const bool any_pressure = flow_.fill_pressure(pressure_data_, *mesh_, dim_);

    ctx_.provide("stress_operators", stress_);
    ctx_.provide("flux_operators", flux_);
    ctx_.provide("boundary_pressure", pressure_data_);

    physics::ModelOptions o;
    o.mobility = dt_;
    o.storage = material_.storage(dim_);
    o.volumetric_compliance = material_.volumetric_compliance(dim_);
    o.biot = material_.biot;
    // one layer, both layouts: the space and the star cannot disagree because
    // neither is stated independently of the other
    o.flux_moments = moments_per_facet();
    o.traction_moments = moments_per_facet();
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("consolidation", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    // the natural pressure datum is a boundary term of the same model, and it
    // is attached only when the flow set actually prescribes one -- the
    // column's drained face is the homogeneous natural case and needs nothing
    if (any_pressure) sim_->model().add("prescribed_pressure", exokal::forms::On::all(), {});

    // each physics resolves its own conditions against the space, which gives
    // every condition its dofs; the strong ones then hand their forms to the
    // constraint set. The natural ones are already data.
    const auto& sp = sim_->epoch().stratum(0).space();
    mechanics_.resolve(*mesh_, dim_, sp);
    flow_.resolve(*mesh_, dim_, sp);
    mechanics_.impose(sim_->constraints());
    flow_.impose(sim_->constraints());
    sim_->freeze_constraints();

    // ---- the initial state, interpolated onto the unknowns --------------
    // The stress is a facet traction moment and the pressure a cell average,
    // so a uniform initial field is exactly representable and needs no
    // projection: the constant moment of sigma0 n over each facet, and p0 on
    // each cell.
    state_.assign(sim_->n_dofs(), 0.0);
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto s_off = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
    for (Index f = 0; f < c.count(dim_ - 1); ++f) {
      const Index cell = first_cofacet(c, dim_, f);
      const auto av = exokal::facet_normal_vector(*mesh_, dim_, cell, f);
      for (int k = 0; k < dim_; ++k) {
        double t = 0.0;
        for (int j = 0; j < dim_; ++j) {
          t += sigma0_[static_cast<std::size_t>(k * 3 + j)] * av[static_cast<std::size_t>(j)];
        }
        state_[s_off + static_cast<std::size_t>(ms.global(dim_ - 1, f, 0, k))] = t;
      }
    }
    const auto p_off = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
    for (Index e = 0; e < c.count(dim_); ++e) state_[p_off + static_cast<std::size_t>(e)] = p0_;

    // the tangent is constant: the model is linear and dt does not move
    jac_ = exokal::forms::TripletSink(sim_->n_dofs());
    sim_->jacobian(jac_);
    system_ = solver::SparseSystem::from(jac_);

    // the accumulation blocks: the pressure row against the stress and against
    // itself. Taking them from the assembled operator rather than re-deriving
    // them keeps the step consistent with the system it steps.
    const auto p_end = p_off + static_cast<std::size_t>(sp.map(sp.index_of("p_0")).size());
    const auto s_end = s_off + static_cast<std::size_t>(sp.map(sp.index_of("s_0")).size());
    for (std::size_t k = 0; k < system_.nnz(); ++k) {
      const auto r = static_cast<std::size_t>(system_.row[k]);
      const auto cc = static_cast<std::size_t>(system_.col[k]);
      if (r >= p_off && r < p_end && ((cc >= s_off && cc < s_end) || (cc >= p_off && cc < p_end))) {
        accumulation_.push_back(k);
      }
    }
    p_offset_ = p_off;
    p_count_ = static_cast<std::size_t>(c.count(dim_));
  }

  // ---- one backward-Euler step -----------------------------------------
  const solver::SparseSystem& system() const { return system_; }

  // the right-hand side of the step from the current state
  std::vector<double> step_rhs() const {
    std::vector<double> b(sim_->n_dofs(), 0.0);
    for (const std::size_t k : accumulation_) {
      b[static_cast<std::size_t>(system_.row[k])] +=
          system_.value[k] * state_[static_cast<std::size_t>(system_.col[k])];
    }
    // the natural pressure datum is a constant load, so it belongs on the
    // right-hand side of every step, not only the first
    std::vector<double> r;
    const_cast<Simulation&>(*sim_).state().assign(sim_->n_dofs(), 0.0);
    const_cast<Simulation&>(*sim_).residual(r);
    for (std::size_t i = 0; i < b.size(); ++i) {
      if (!sim_->constraints().pinned(i)) b[i] -= r[i];
    }
    for (std::size_t d = 0; d < sim_->n_dofs(); ++d) {
      if (sim_->constraints().pinned(d)) {
        b[d] = sim_->constraints().scale_at(d) * sim_->constraints().rhs_at(d);
      }
    }
    return b;
  }

  void accept(std::vector<double> x) { state_ = std::move(x); }

  double cell_pressure(Index e) const { return state_[p_offset_ + static_cast<std::size_t>(e)]; }
  std::size_t n_cells() const { return p_count_; }

 private:
  static Index first_cofacet(const graphos::Complex& c, int cell_dim, Index facet) {
    const graphos::CoboundaryOperator cob = graphos::coboundary(c, cell_dim - 1);
    return cob.indices[static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(facet)])];
  }

  const exokal::Mesh* mesh_;
  int dim_;
  PoroelasticMaterial material_;
  double dt_;
  Layer layer_{Layer::bdm};
  MechanicsBoundary mechanics_;
  FlowBoundary flow_;
  std::array<double, 9> sigma0_{};
  double p0_{0.0};

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::StressOperators stress_;
  exokal::hodge::FluxOperators flux_;
  BoundaryData pressure_data_{0};
  exokal::forms::TermContext ctx_;
  std::unique_ptr<Simulation> sim_;
  exokal::forms::TripletSink jac_{0};
  solver::SparseSystem system_;
  std::vector<std::size_t> accumulation_;
  std::vector<double> state_;
  std::size_t p_offset_{0}, p_count_{0};
};

}  // namespace mimetika
