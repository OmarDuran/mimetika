#pragma once

#include <functional>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/geometry/degeneracy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/linear_solver/linear.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/elasticity.hpp"
#include "mimetika/model/partition.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"

// CAUCHY ELASTICITY, STATED AS DATA -- the poroelastic model with the flow
// taken out, and the smallest problem that exercises the stress space alone.
//
// Weakly-symmetric mixed form (Hellinger-Reissner), THREE fields:
//
//     r_sigma = M sigma - D^T u - A^T gamma      the constitutive relation
//     r_u     = + D sigma                        momentum balance
//     r_gamma = + A sigma                        symmetry, imposed weakly
//
// gamma is not decoration. The space carries no symmetry constraint -- that is
// what makes it usable -- so gamma is the multiplier enforcing sigma = sigma^T
// against the rigid rotations. Dropping it does not simplify the method, it
// changes it.
//
// It earns its keep the way SinglePhaseModel does for the flux: when a
// poroelastic answer is wrong, this is what says whether the mechanics half is.
// Both of its closed forms hold in any dimension:
//
//     a column    uniaxial extension: constant stress, LINEAR displacement,
//                 which every one of these spaces contains exactly
//     an annulus  Lame's thick-walled tube: sigma_rr, sigma_tt carry a 1/r^2
//                 and u_r a 1/r, which none of them contains
//
// so the first is a statement about the space and the second about resolution.

namespace mimetika {

// Isotropic linear elasticity, in the two parameters the operators want. The
// engineering constants are derived rather than stored, so a caller can state
// whichever pair it has.
struct ElasticMaterial {
  double shear{1.0};  // mu
  double lame{1.0};   // lambda

  static ElasticMaterial from_young_poisson(double E, double nu) {
    if (!(nu > -1.0 && nu < 0.5)) {
      throw std::invalid_argument(
          "ElasticMaterial: Poisson's ratio must lie in (-1, 1/2); at 1/2 exactly the material "
          "is incompressible and lambda is not finite -- state mu and lambda directly");
    }
    return {E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))};
  }

  double poisson() const { return lame / (2.0 * (lame + shear)); }
  double young() const { return shear * (3.0 * lame + 2.0 * shear) / (lame + shear); }
  // the oedometer modulus: lam + 2mu in any dimension, since confinement
  // removes every lateral strain whatever d is
  double oedometer() const { return lame + 2.0 * shear; }
};

class CauchyElasticityModel {
 public:
  // WHICH DISCRETE STRESS HODGE, and only the two that are elements.
  //
  //   derham          d copies of the mimetic-BDM plus a rank-one volumetric
  //                   fold-back. Consistency-only: the scalar layer is
  //                   unisolvent, so N is square and nothing is stabilized.
  //                   Any cell type, either dimension.
  //   stabilized_bdm  the same d^2 degrees of freedom reconstructed on the FULL
  //                   linear tensor space [P_1]^{dxd}, m = d^2(d+1) modes. On a
  //                   simplex D = m and the stabilization VANISHES -- there it
  //                   is the conforming AFW/BDM_1 element, checked by congruence
  //                   in exokal's hodge.test_afw_equivalence. On a polytope a
  //                   stabilization remains: 4 on a quadrilateral, 18 on a hex.
  //
  // The third realization exokal offers, derham_rt, is deliberately NOT here. It
  // is a sound inner product -- unisolvent, positive definite, exact on the
  // compliance energy -- and it is not an element: one constant traction vector
  // per facet cannot control the rigid rotations across a mesh, the inf-sup for
  // gamma degenerates, and the saddle point comes out singular. Offering it
  // would be offering a model that does not solve. See
  // tests/model/test_dimensions.cpp, which pins exactly that.
  using Realization = exokal::hodge::StressOperators::Realization;
  using Formulation = exokal::hodge::StressOperators::Formulation;

  CauchyElasticityModel(const exokal::Mesh& mesh, int cell_dim, ElasticMaterial material,
                        Realization how = Realization::derham_bdm,
                        Formulation form = Formulation::weak_symmetry)
      : mesh_(&mesh), dim_(cell_dim), material_(material), how_(how), form_(form) {
    // derham_rt is unisolvent and still refused: d per facet cannot control the
    // rigid rotations across a mesh, so the weak-symmetry inf-sup degenerates
    // and the saddle point comes out singular.
    //
    // diagonal_tpsa carries the same d per facet and is NOT refused, because it
    // is a different scheme rather than a coarser space: its face rotation is a
    // convention rather than an inf-sup, and exokal admits it in four fields
    // only, which is where its M is diagonal.
    if (how == Realization::derham_rt) {
      throw std::invalid_argument(
          "CauchyElasticityModel: derham_rt is unisolvent but its weak-symmetry inf-sup "
          "degenerates; use derham_bdm, stabilized_bdm, or diagonal_tpsa");
    }
    if (how == Realization::diagonal_tpsa && form != Formulation::weak_symmetry_total) {
      throw std::invalid_argument(
          "CauchyElasticityModel: diagonal_tpsa needs the total-pressure formulation -- the "
          "three-field compliance couples traction components through the trace");
    }
    // THE SYMMETRY AXIS IS ONE DECISION: a strongly-symmetric realization
    // under a weak formulation (or the reverse) is refused here as exokal
    // refuses it at build, so the error arrives where the choice was made.
    if (exokal::hodge::StressOperators::strongly_symmetric(how) !=
        exokal::hodge::StressOperators::strongly_symmetric(form)) {
      throw std::invalid_argument(
          "CauchyElasticityModel: the realization and the formulation must agree on where the "
          "symmetry lives -- the vem family builds strong_symmetry(_total), the rest the weak "
          "pair");
    }
    // the rigid-motion ansatz is a three-dimensional construction: six
    // traction moments per facet and the six rigid motions per cell
    if (exokal::hodge::StressOperators::strongly_symmetric(how) && cell_dim != 3) {
      throw std::invalid_argument(
          "CauchyElasticityModel: the strongly-symmetric vem family is a 3D construction");
    }
    if ((how == Realization::diagonal_vem || how == Realization::adaptive_vem) &&
        form != Formulation::strong_symmetry_total) {
      throw std::invalid_argument(
          std::string("CauchyElasticityModel: ") + exokal::hodge::StressOperators::name(how) +
          " needs strong_symmetry_total -- the plain compliance couples traction components "
          "through the trace and cannot be diagonal");
    }
  }

  // THE adaptive_vem THRESHOLD: scan for metrically degenerate cells at this
  // percentage of the node-star mean and give them the diagonal star, eta = 0.
  // The same contract adaptive_rt carries for the flux: eta is DERIVED --
  // ones, with the flagged cells zeroed -- never given as a field. Unset,
  // exokal scans at its own default_degeneracy_percent.
  void set_degeneracy_percent(double percent) { degeneracy_percent_ = percent; }

  // The selection AS BUILT, one value per cell: 1 is the stabilized vem
  // product, 0 the diagonal star. Empty unless the realization is
  // adaptive_vem; the field to write next to the solution.
  const std::vector<double>& eta() const {
    if (sim_ == nullptr) {
      throw std::logic_error("CauchyElasticityModel: not built yet; call build() or solve() first");
    }
    return stress_.eta();
  }

  bool strongly_symmetric() const {
    return exokal::hodge::StressOperators::strongly_symmetric(how_);
  }

  // THE VALIDITY GATE OF THE DIAGONAL STAR, which exokal states and leaves to
  // the consumer: a facet the cell centroid does not see squarely carries a
  // NON-POSITIVE weight -- delta = (x_f - x_E).n <= 0 -- and the assembled M
  // is then not positive definite. Condensation divides by those entries and
  // the solve collapses to a near-zero field that still reports CONVERGED:
  // the one failure mode worse than a wrong answer. So the count is exposed,
  // per cell with any offending facet, for the driver to refuse or report.
  std::size_t n_invalid_star() const {
    if (sim_ == nullptr) {
      throw std::logic_error("CauchyElasticityModel: not built yet; call build() or solve() first");
    }
    std::size_t n = 0;
    for (Index e = 0; e < static_cast<Index>(n_cells_); ++e) {
      const auto& d = stress_.compact(e).diag;
      for (const double v : d) {
        if (!(v > 0.0)) {
          ++n;
          break;
        }
      }
    }
    return n;
  }

  Formulation formulation() const { return form_; }

  // THE TOTAL PRESSURE p = lambda div u, one scalar per cell, and a FIELD in
  // the four-field formulation rather than a post-processing of the stress.
  // Asking for it in three fields asks for something that was never solved for.
  double total_pressure(Index cell) const {
    if (form_ != Formulation::weak_symmetry_total && form_ != Formulation::strong_symmetry_total) {
      throw std::logic_error(
          "CauchyElasticityModel::total_pressure: this formulation has no total pressure; build "
          "with weak_symmetry_total or strong_symmetry_total");
    }
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mp = sp.map(sp.index_of("p_0"));
    return state_[static_cast<std::size_t>(sp.offset(sp.index_of("p_0"))) +
                  static_cast<std::size_t>(mp.global(dim_, cell, 0, 0))];
  }

  MechanicsBoundary& mechanics() { return mechanics_; }
  const MechanicsBoundary& mechanics() const { return mechanics_; }

  int dim() const { return dim_; }
  const exokal::Mesh& mesh() const { return *mesh_; }
  Realization realization() const { return how_; }
  const char* realization_name() const { return exokal::hodge::StressOperators::name(how_); }
  const ElasticMaterial& material() const { return material_; }
  // NON-CONST, for the one caller that configures the simulation rather than
  // reading it: the partition, which tells it which sites to assemble.
  Simulation& simulation() { return *sim_; }
  const Simulation& simulation() const { return *sim_; }
  const solver::SparseSystem& system() const { return system_; }
  const std::vector<double>& rhs() const { return rhs_; }
  std::size_t n_cells() const { return n_cells_; }
  // how many cells needed a stabilization: zero on a simplex mesh for either
  // realization, and that is the construction rather than a coincidence
  std::size_t n_stabilized() const { return stress_.n_stabilized(); }
  const exokal::hodge::StressOperators& stress_operators() const { return stress_; }

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
    // the K-independent mode selection, which only the de Rham product has
    const bool derham = how_ == Realization::derham_bdm;
    if (derham) geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    // THE adaptive_vem SELECTION, DERIVED RATHER THAN GIVEN: ones, with 0 on
    // the cells the scan flags at the caller's threshold. exokal forces its
    // own default-threshold zeros on top either way, so a caller can only
    // widen the set -- the same contract adaptive_rt carries for the flux.
    if (degeneracy_percent_ >= 0.0 && how_ != Realization::adaptive_vem) {
      throw std::invalid_argument(
          "CauchyElasticityModel: the degeneracy threshold is adaptive_vem's cell selection");
    }
    std::vector<double> eta;
    if (how_ == Realization::adaptive_vem && degeneracy_percent_ >= 0.0) {
      eta.assign(static_cast<std::size_t>(c.count(dim_)), 1.0);
      for (const Index e : exokal::degenerate_cell_ids(*mesh_, dim_, degeneracy_percent_)) {
        eta[static_cast<std::size_t>(e)] = 0.0;
      }
    }
    stress_ = exokal::hodge::StressOperators::build(*mesh_, dim_, material_.shear, material_.lame,
                                                    how_, form_, derham ? &geometry_ : nullptr,
                                                    only, eta.empty() ? nullptr : &eta);
    ctx_.provide("stress_operators", stress_);

    displacement_data_ = BoundaryVectorData(static_cast<std::size_t>(c.count(dim_ - 1)));
    for (const auto& d : displacement_facets_) {
      displacement_data_.set_affine(d.facets, d.constant, d.gradient);
    }
    ctx_.provide("boundary_displacement", displacement_data_);

    // THE STRONG DATUM, EXPANDED AT BUILD: the six moments of the affine
    // u_D = a + B (x - x_E) against the facet basis, divided by the |f| Gram.
    // The operators do not carry the chart and second moments per cell, and
    // the datum is affine, so a quadrature that is exact for these integrals
    // runs once here and the boundary term reads numbers -- the affine datum
    // stays EXACT, as it is in the weak family.
    if (strongly_symmetric() && !displacement_facets_.empty()) {
      strong_displacement_ =
          StrongDisplacementCoefficients(static_cast<std::size_t>(c.count(dim_ - 1)));
      // THE COBOUNDARY IS BUILT ONCE. cofacet_of rebuilds it per call, which
      // is O(mesh) each time -- 5k boundary facets on a 22k-cell mesh then
      // spend minutes recomputing the same operator that takes milliseconds
      // to build once. The same lesson the python example already carries.
      const graphos::CoboundaryOperator cob = graphos::coboundary(c, dim_ - 1);
      for (const auto& d : displacement_facets_) {
        for (const Index f : d.facets) {
          const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
          const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
          if (e - b != 1) {
            throw std::invalid_argument(
                "prescribe_displacement: facet " + std::to_string(f) + " is interior");
          }
          strong_displacement_.set(
              f, strong_datum_coefficients(f, cob.indices[b], d.constant, d.gradient));
        }
      }
      ctx_.provide("strong_boundary_displacement", strong_displacement_);
    }

    reservoir_data_ = CellData(static_cast<std::size_t>(c.count(dim_)));
    for (const auto& r : reservoir_) reservoir_data_.set(r.cells, r.pressure);
    ctx_.provide("reservoir_pressure", reservoir_data_);

    // THE SPACE FOLLOWS THE STAR: d^2 traction moments per facet for both of
    // these, read off the operators rather than restated, so the layout and the
    // product cannot drift apart.
    physics::ModelOptions o;
    o.traction_moments = stress_.moments_per_facet();
    // AND THE FIELD COUNT FOLLOWS THE FORMULATION, read off the operators for
    // the same reason: the field roster is a property of the product that was
    // built, not a second choice made here.
    o.total_pressure = stress_.formulation() == Formulation::weak_symmetry_total ||
                       stress_.formulation() == Formulation::strong_symmetry_total;
    o.strong_symmetry = strongly_symmetric();
    // the four-field term keeps (2 mu)^-1 as its compliance, so mu travels with
    // the composition rather than being looked up from a slot
    o.shear_modulus = material_.shear;
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("linear_elasticity", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    // the natural displacement datum, attached only when one is actually given
    if (!displacement_facets_.empty()) {
      sim_->model().add(
          strongly_symmetric() ? "strong_prescribed_displacement" : "prescribed_displacement",
          exokal::forms::On::all(), {});
    }
    if (!reservoir_.empty()) {
      exokal::forms::Params rp;
      rp.set("biot", biot_);
      rp.set("volumetric_compliance", volumetric_compliance_);
      sim_->model().add("reservoir_pressurization", exokal::forms::On::all(), rp);
    }

    // the conditions resolve against the space, which is what gives each of
    // them its dofs, and the strong ones then hand their forms to the
    // constraint set
    const auto& sp = sim_->epoch().stratum(0).space();
    mechanics_.resolve(*mesh_, dim_, sp);
    mechanics_.impose(sim_->constraints());

    // the prescribed fracture traction, registered with a placeholder value:
    // the STRUCTURE is what freeze_constraints needs, and the values move later
    {
      const auto& ms = sp.map(sp.index_of("s_0"));
      const auto s_base = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
      // the strong family carries its six moments on ONE scalar layout, the
      // weak one `moments` per each of d components
      const int nb = stress_.moments_per_facet();
      const int nk = strongly_symmetric() ? 1 : dim_;
      for (const Index f : prescribed_) {
        for (int b = 0; b < nb; ++b) {
          for (int k = 0; k < nk; ++k) {
            const auto d =
                static_cast<Index>(s_base + static_cast<std::size_t>(ms.global(dim_ - 1, f, b, k)));
            prescribed_forms_.push_back(sim_->constraints().size());
            prescribed_dofs_.push_back(static_cast<std::size_t>(d));
            sim_->constraints().pin(d, 0.0);
          }
        }
      }
    }
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

    // RESERVE BEFORE ASSEMBLING, and hand the storage over afterwards.
    //
    // The triplets are the largest object this model ever holds: every cell
    // emits a dense block over its own unknowns, so the count is the sum of
    // their squares -- of order 10^8 on a mesh of tens of thousands of
    // polyhedra. Growing three vectors to that reallocates about twenty-five
    // times, and each reallocation copies everything already written; the
    // transient peak is several times the final size. The count is known here
    // in closed form, so it is asked for once.
    exokal::forms::TripletSink jac(sim_->n_dofs());
    // n_cells_ is not set until later in this function, so the count comes from
    // the mesh: reading it too early reserved nothing at all, and the three
    // vectors then grew to 10^8 entries by doubling.
    std::size_t nnz = 0;
    const Index n_cells = mesh_->topology().count(dim_);
    for (Index e = 0; e < n_cells; ++e) {
      // counted on the COMPACT cell: reading M here would materialize the
      // dense zeros of every diagonal star just to know their size, and a
      // diagonal block emits its diagonal alone
      const auto& op = stress_.compact(e);
      const std::size_t nb = op.diag.empty() ? op.M.rows() : op.diag.size();
      const std::size_t d = nb + op.Dv.rows() + op.As.rows();
      nnz += op.diag.empty() ? d * d : d * d - nb * (nb - 1);
    }
    jac.reserve(nnz);
    sim_->jacobian(jac);
    system_ = solver::SparseSystem::from(std::move(jac));

    // THE LOAD. Everything the terms contribute that does not depend on the
    // unknowns is a load, and the residual at the zero state is exactly minus
    // that -- so it is read off rather than re-derived. The strongly imposed
    // rows carry their own datum, scaled as their equation is.
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

    // THE FACTORIZATION FOR S IS DEFERRED, because most problems never ask for
    // it. S exists for prescribed TRACTIONS -- the contact iteration reuses one
    // factorization across many right-hand sides -- and a model that prescribes
    // none never evaluates it. Taking it here made every build pay a full
    // direct solve of the whole saddle point: on a mesh of tens of thousands of
    // polyhedra that is minutes and gigabytes of fill, spent inside "assembly",
    // for a matrix nothing goes on to use.
    factorized_ = false;
    load_ready_ = true;
    work_.assign(sim_->n_dofs(), 0.0);
    s_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
    u_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
    // the strong family has no rotation field: symmetry lives in the space
    g_offset_ = strongly_symmetric() ? 0 : static_cast<std::size_t>(sp.offset(sp.index_of("g_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  // MOVE THE LOAD WITHOUT REFACTORIZING.
  //
  // THE SYSTEM MATRIX NEVER DEPENDS ON THE PRESSURE. A depletion is a load: it
  // enters the stress row as alpha T^T p and touches the right-hand side alone,
  // while the matrix is fixed by the mesh, the moduli and which facets are
  // prescribed. So a sweep over depletion levels -- which is what benchmarks 2
  // and 3 are -- needs ONE factorization, not one per level.
  //
  // Rebuilding the model per level is not merely slow. It re-runs the whole
  // construction and a direct factorization of a large system on every step of
  // an outer iteration, which for a slip-weakening branch tracker is hundreds
  // of times per benchmark. The Python reference caches exactly this and says
  // so: "the system matrix never depends on the pressure; on a cache hit only
  // the right-hand side is rebuilt".
  void set_depletion(double pressure) {
    if (!load_ready_) throw std::logic_error("set_depletion: call build() first");
    for (Reservoir& r : reservoir_) r.pressure = pressure;
    reservoir_data_ = CellData(n_cells_);
    for (const auto& r : reservoir_) reservoir_data_.set(r.cells, r.pressure);
    ctx_.provide("reservoir_pressure", reservoir_data_);

    // the residual at the zero state IS minus the load, and only it moved
    std::vector<double> r(sim_->n_dofs(), 0.0);
    sim_->state().assign(sim_->n_dofs(), 0.0);
    sim_->residual(r);
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      // -r is a CONTRIBUTION and is summed across the processes; a constrained
      // row is a REPLACEMENT and is written by whichever process owns it, so
      // the others leave it at zero rather than adding a copy of it.
      const bool mine = sim_->owned_dofs().empty() || sim_->owned_dofs()[i] != 0;
      rhs_[i] = sim_->constraints().pinned(i)
                    ? (mine ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i) : 0.0)
                    : -r[i];
    }
  }

  // THE TRACE OPERATOR: the displacement jump across an INTERIOR facet, as the
  // adjoint of the divergence and the asymmetry.
  //
  //     [| D^T u + A^T gamma |]_f = (D^T u + A^T gamma)_f^+ - (...)_f^-
  //
  // and the right-hand side is not two evaluations and a subtraction. D maps
  // facet tractions to cell vectors, so its adjoint maps cell displacements
  // back onto facets carrying each cell's OUTWARD incidence on the facet; the
  // two cofaces of an interior facet therefore enter with opposite signs and the
  // ASSEMBLED ROW IS THE DIFFERENCE. The jump falls out of the adjoint for free,
  // and the same holds for A and the rotation, which supplies the rigid-rotation
  // part of the displacement field the facet sees.
  //
  // What is returned is the FULL constitutive row of the UNFRACTURED system,
  //
  //     jump_f = -( M sigma - D^T u - A^T gamma )_f ,
  //
  // in the ambient traction-moment components of facet f. The M sigma term is
  // not optional and the bonded case is the proof: on an interior facet with no
  // fracture the displacement is continuous, the two cofaces' boundary terms
  // cancel, and this residual vanishes -- which is exactly the statement
  // [[u]] = 0. Dropping M sigma leaves 4e-2 there on a unit column instead of
  // round-off, so the adjoint terms alone are a jump only in the sense of
  // naming which operators carry the sign structure, not as a formula.
  //
  // Three properties, each load bearing:
  //
  //   * IT IS LINEAR in the state, so it applies even where that row has been
  //     REPLACED by a contact constraint. The equation is gone; the functional
  //     it expressed is not.
  //   * IT MUST BE THE UNFRACTURED ROW. Where a constraint replaced it, the
  //     fractured residual is zero at the solution, whereas THIS residual is
  //     precisely the jump it exists to extract.
  //   * NO Gram^{-1}. A traction degree of freedom IS a moment
  //     m = int_f (sigma n) b, so recovering its pointwise values needs
  //     Gram^{-1} m. The jump term int_f [[u]].(tau n) is paired AGAINST that
  //     moment -- writing (tau n) = sum_b phi_b b_b gives m = Gram phi -- so the
  //     pairing already carries a Gram^{-1} and what emerges are the expansion
  //     COEFFICIENTS of the jump. A second inversion divides by |f| and the slip
  //     then grows like 1/h under refinement: a mesh-dependent answer.
  //
  // It is a property of the DISCRETIZATION and not of contact, which is why it
  // lives here: any consumer wanting relative motion across a facet -- a
  // fracture, a material interface, a post-processing -- wants this functional.
  // PoroelasticModel carries the same method with the Biot term added.
  std::vector<double> trace(Index facet, const std::vector<double>& z) const {
    // the contact machinery reads the weak family's (moment, component) facet
    // blocks; the strong family's six-slot basis is not wired through it yet
    if (strongly_symmetric()) {
      throw std::logic_error(
          "CauchyElasticityModel::trace: not implemented for the strongly-symmetric vem family");
    }
    const graphos::Complex& c = mesh_->topology();
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto& mu = sp.map(sp.index_of("u_0"));
    const auto& mg = sp.map(sp.index_of("g_0"));
    const auto u_off = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
    const auto g_off = static_cast<std::size_t>(sp.offset(sp.index_of("g_0")));

    const int nb = stress_.moments_per_facet();
    const std::size_t ndf = static_cast<std::size_t>(dim_) * static_cast<std::size_t>(nb);
    std::vector<double> row(ndf, 0.0);

    // BOTH COFACES, each contributing through its own local operators. The
    // outward incidence is already inside Dv and As -- StressOperators put it
    // there when it converted to the canonical basis -- so summing the two
    // sides IS the jump, with no sign applied here.
    const graphos::CoboundaryOperator cob = graphos::coboundary(c, dim_ - 1);
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(facet)]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(facet) + 1]);
    for (std::size_t m = b; m < e; ++m) {
      const Index cell = cob.indices[m];
      const auto& op = stress_.cell(cell);
      // where this facet's block sits within the cell's stress operators
      std::size_t slot = 0;
      bool found = false;
      for (std::size_t i = 0; i < op.faces.size(); ++i) {
        if (op.faces[i] == facet) {
          slot = i;
          found = true;
          break;
        }
      }
      if (!found) throw std::runtime_error("trace: the facet is not a face of its coface");

      const std::size_t nu = op.Dv.rows(), ng = op.As.rows();
      const std::size_t D = op.M.rows();
      for (std::size_t k = 0; k < ndf; ++k) {
        // the local stress index of (this facet, component/basis k), in the
        // ProductSpace order the operators were permuted into
        const std::size_t local = slot * ndf + k;
        double acc = 0.0;
        // M sigma, over EVERY facet of this cell: the compliance couples the
        // fracture facet to the cell's other faces
        for (std::size_t j = 0; j < D; ++j) {
          // within a facet block the ProductSpace orders the COMPONENT fastest
          // -- b * d + k -- which is the order StressOperators permuted its
          // operators into, so the moment and the component split out of the
          // local index rather than being passed whole
          const std::size_t jf = j / ndf, jk = j % ndf;
          const int moment = static_cast<int>(jk) / dim_;
          const int comp = static_cast<int>(jk) % dim_;
          acc += op.M(local, j) * z[s_offset_ + static_cast<std::size_t>(ms.global(
                                                    dim_ - 1, op.faces[jf], moment, comp))];
        }
        for (std::size_t j = 0; j < nu; ++j) {
          acc -= op.Dv(j, local) *
                 z[u_off + static_cast<std::size_t>(mu.global(dim_, cell, 0, static_cast<int>(j)))];
        }
        for (std::size_t j = 0; j < ng; ++j) {
          acc -= op.As(j, local) *
                 z[g_off + static_cast<std::size_t>(mg.global(dim_, cell, 0, static_cast<int>(j)))];
        }
        row[k] += acc;
      }
    }
    // THE SIGN IS FIXED BY THE COMPRESSED CASE, not by the bonded one. The
    // residual vanishes wherever the material is continuous, and zero has no
    // sign, so the bonded identity cannot distinguish g from -g. What does:
    // prescribe a ZERO traction on a fault under compressive boundary
    // displacement and the two halves must OVERLAP, so the gap has to come out
    // NEGATIVE there. With the opposite convention the solver reads that as an
    // open fault, settles in one iteration carrying no load, and reports a
    // converged answer satisfying g >= 0, t <= 0 and g t = 0 -- formally
    // Signorini, and the wrong branch of it.
    //
    // The same choice is what makes the outer iteration contract. Measured on a
    // compressed column, dg/dt = -0.16 in this convention, and the Uzawa
    // multiplier |1 + r dg/dt| is below one exactly when that slope is
    // negative; with the sign the other way no augmentation converges. The
    // physics and the contraction agree, which is the check that they are both
    // right.
    for (double& v : row) v = -v;
    return row;
  }

  // ---- the affine solution operator ------------------------------------
  //
  // PRESCRIBE THE TRACTION on a set of facets, as an essential condition: in
  // the mixed form sigma|_f IS a degree of freedom, so "the fracture carries
  // this traction" is a Dirichlet condition on the unknown and not a penalty.
  // Registered BEFORE build(), because it changes which equations the system
  // has; the VALUES may then move freely, which is what the outer iteration
  // needs.
  void prescribe_traction(std::vector<Index> facets) { prescribed_ = std::move(facets); }

  // RESERVOIR PRESSURIZATION: a pore-pressure change on a set of cells, entering
  // the mechanics as a LOAD.
  //
  // The benchmarks do not solve the flow -- the pressure is data and only the
  // mechanical response is computed -- so the Biot coupling contributes
  // alpha T^T p to the right-hand side and there is no pressure unknown. The
  // coefficient and the operator are the ones the coupled solver uses, so the
  // two agree exactly where both are posed.
  void pressurize(const std::vector<Index>& cells, double pressure, double biot = 1.0,
                  double volumetric_compliance = 1.0) {
    reservoir_.push_back({cells, pressure});
    biot_ = biot;
    volumetric_compliance_ = volumetric_compliance;
  }

  // PRESCRIBE A DISPLACEMENT on a set of boundary facets: u = a + B (x - x_E).
  //
  // NATURAL, not essential. In the Hellinger-Reissner form the interior
  // displacement enters the stress row as -D^T u, so on a boundary facet a
  // prescribed displacement takes its place in the right-hand side -- there is
  // no displacement degree of freedom on a facet to constrain. The affine datum
  // is EXACT: both integrals it needs are already in the stress operators'
  // facet moments, so no boundary quadrature enters and a linear displacement
  // is reproduced rather than approximated.
  //
  // It is also what makes a fracture spanning the whole domain well posed. Under
  // pure traction data each side of such a fracture carries a rigid-body null
  // mode and equilibrium alone fixes the fault traction; with the displacement
  // prescribed there is no null mode and the traction is genuinely an unknown
  // for a contact law to determine.
  void prescribe_displacement(const std::vector<Index>& facets,
                              const std::array<double, 3>& constant,
                              const std::array<double, 9>& gradient = {}) {
    displacement_facets_.push_back({facets, constant, gradient});
  }
  const std::vector<Index>& prescribed_traction() const { return prescribed_; }

  // S : m -> z(m), the solution of the mixed problem on the affine subspace
  // { z : sigma|_F = m }.
  //
  //     A_CC z_C = b_C - A_CF m ,   z_F = m .
  //
  // A_CC DOES NOT DEPEND ON m -- prescribing removes the same rows and columns
  // whatever the values are -- so the operator is factorized once and every
  // later evaluation is a back-substitution against a moved right-hand side.
  // That is the whole reason contact can iterate without touching the global
  // system more than once.
  //
  // `moments` runs facet-major over prescribed_traction(), d * nb entries each,
  // in the ProductSpace order (component fastest).
  const std::vector<double>& solution_operator(const std::vector<double>& moments) const {
    const std::size_t ndf =
        strongly_symmetric()
            ? std::size_t{6}
            : static_cast<std::size_t>(dim_) * static_cast<std::size_t>(stress_.moments_per_facet());
    if (moments.size() != prescribed_.size() * ndf) {
      throw std::invalid_argument("solution_operator: one traction moment block per facet");
    }
    auto& constraints = const_cast<Simulation&>(*sim_).constraints();
    for (std::size_t i = 0; i < prescribed_.size(); ++i) {
      for (std::size_t k = 0; k < ndf; ++k) {
        constraints.set_value(prescribed_forms_[i * ndf + k], moments[i * ndf + k]);
      }
    }
    // only the prescribed rows move: the load is unchanged, so the residual
    // never has to be reassembled
    for (std::size_t i = 0; i < prescribed_.size(); ++i) {
      for (std::size_t k = 0; k < ndf; ++k) {
        const auto d = prescribed_dofs_[i * ndf + k];
        rhs_[d] = constraints.scale_at(d) * moments[i * ndf + k];
      }
    }
    // one factorization, taken the first time S is actually evaluated
    if (!factorized_) {
      solver_.factorize(system_);
      factorized_ = true;
    }
    solver_.solve(rhs_, work_);
    return work_;
  }

  void accept(std::vector<double> x) { state_ = std::move(x); }
  const std::vector<double>& state() const { return state_; }

  // THE CELL DISPLACEMENT, in the sign and the scale a caller means by it.
  //
  // Two conversions, and both belong here rather than at every read site. The
  // unknown is the MOMENT of u over the cell, not a nodal value, so the mean is
  // that divided by the measure. And it carries the opposite SIGN to the
  // physical displacement: the mixed form is written [M, -B^T; +B, 0], so the
  // multiplier standing in the constitutive row is -u. That convention is not a
  // free choice once several physics land in one system -- two of them meeting
  // there would give a matrix neither symmetric nor antisymmetric -- so the
  // place to undo it is the accessor, once, and not in whatever measures a
  // result. Reading the raw dof gives an answer that is exactly twice the
  // solution away from it, which looks like a discretization error and is not.
  double displacement(Index cell, int axis) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mu = sp.map(sp.index_of("u_0"));
    return -state_[u_offset_ + static_cast<std::size_t>(mu.global(dim_, cell, 0, axis))] /
           exokal::measure(*mesh_, dim_, cell);
  }

  // THE CELL ROTATION, the multiplier of the weak symmetry constraint.
  //
  // Same two conversions as the displacement, and for the same reasons: the
  // unknown is the MOMENT over the cell and it stands in the constitutive row
  // with the opposite sign. `p` indexes the generators of skew(d) in (i < j)
  // order -- one in two dimensions, three in three -- so gamma(e, 0) is the
  // rotation of the plane in 2D and the yz, xz, xy components follow in 3D.
  //
  // For a displacement field u, the rotation it is measuring is skew(grad u):
  // an exactly reproduced linear field therefore has an exactly reproduced
  // rotation, which is what makes it worth reporting next to the stress.
  double rotation(Index cell, int p) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    // STRONG SYMMETRY HAS NO MULTIPLIER: the rotation lives in the last three
    // of the displacement's six rigid-motion coefficients. Two conversions on
    // top of the measure and multiplier sign:
    //
    //   * the ORDER. exokal's generators run BY AXIS (e_x∧r, e_y∧r, e_z∧r);
    //     the weak multiplier runs by (i < j) PLANE. Pair p is the rotation
    //     about axis 2-p, with the alternating sign of ε_ijk: W_ij = -ω_axis
    //     for the xy and yz planes, +ω for xz.
    //   * the STRAIN COUPLING. The coefficients are the L²-RM projection of
    //     u, and on a cell whose centred second moment M₂ is anisotropic that
    //     projection sees the strain: ω_proj = ω + J⁻¹ q(ε), with
    //     J = tr(M₂)I - M₂ and q_a(ε) = ε_{lan} ε_lm (M₂)_nm. The strain is
    //     C⁻¹σ from the cell stress -- exact for affine fields -- so
    //     ω = ω_proj - J⁻¹ q(ε) restores skw(grad u) on ANY cell. On a cube
    //     q vanishes, which is how reporting ω_proj raw passes every
    //     structured test and fails a sheared polyhedron by O(shear).
    if (strongly_symmetric()) {
      const auto& mu = sp.map(sp.index_of("u_0"));
      const double volume = exokal::measure(*mesh_, dim_, cell);
      std::array<double, 3> w_proj{};
      for (std::size_t k = 0; k < 3; ++k) {
        w_proj[k] = -state_[u_offset_ + static_cast<std::size_t>(
                                            mu.global(dim_, cell, 0, 3 + static_cast<int>(k)))] /
                    volume;
      }

      // the cell-average strain from the full stress: eps = (sigma - a tr I)/2mu
      const std::array<double, 9> s = cell_stress(cell);
      const double a = material_.lame / (2.0 * material_.shear + 3.0 * material_.lame);
      const double trs = s[0] + s[4] + s[8];
      std::array<double, 9> eps{};
      for (std::size_t i = 0; i < 9; ++i) {
        eps[i] = (s[i] - (i % 4 == 0 ? a * trs : 0.0)) / (2.0 * material_.shear);
      }

      // the centred second moments, by a rule exact for them
      const exokal::Point xE = exokal::centroid(*mesh_, dim_, cell);
      const exokal::QuadratureRule qr = exokal::cell_quadrature(*mesh_, dim_, cell, 2);
      std::array<double, 9> m2{};
      for (std::size_t pt = 0; pt < qr.weights.size(); ++pt) {
        const exokal::Point& x = qr.points[pt];
        const std::array<double, 3> r{x[0] - xE[0], x[1] - xE[1], x[2] - xE[2]};
        for (std::size_t i = 0; i < 3; ++i) {
          for (std::size_t j = 0; j < 3; ++j) m2[i * 3 + j] += qr.weights[pt] * r[i] * r[j];
        }
      }

      // q_a = eps_{lan} E_lm M2_nm, J_ab = tr(M2) delta_ab - M2_ab
      const auto lc = [](std::size_t i, std::size_t j, std::size_t k) -> double {
        if (i == j || j == k || i == k) return 0.0;
        return (j == (i + 1) % 3) ? 1.0 : -1.0;
      };
      std::array<double, 3> q{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t l = 0; l < 3; ++l) {
          for (std::size_t n = 0; n < 3; ++n) {
            const double e = lc(l, axis, n);
            if (e == 0.0) continue;
            for (std::size_t mcol = 0; mcol < 3; ++mcol) {
              q[axis] += e * eps[l * 3 + mcol] * m2[n * 3 + mcol];
            }
          }
        }
      }
      const double trm = m2[0] + m2[4] + m2[8];
      std::array<double, 9> J{};
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
          J[i * 3 + j] = (i == j ? trm : 0.0) - m2[i * 3 + j];
        }
      }
      // solve J c = q, 3x3, by Cramer: J is SPD on a non-degenerate cell
      const auto det3 = [](const std::array<double, 9>& A) {
        return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
               A[2] * (A[3] * A[7] - A[4] * A[6]);
      };
      const double dJ = det3(J);
      std::array<double, 3> cvec{};
      for (std::size_t col = 0; col < 3; ++col) {
        std::array<double, 9> Jc = J;
        for (std::size_t row = 0; row < 3; ++row) Jc[row * 3 + col] = q[row];
        cvec[col] = det3(Jc) / dJ;
      }
      const std::array<double, 3> w{w_proj[0] - cvec[0], w_proj[1] - cvec[1],
                                    w_proj[2] - cvec[2]};
      // pair p spans plane (i, j) and is the rotation about axis 2-p
      const double by_axis = w[static_cast<std::size_t>(2 - p)];
      return p == 1 ? by_axis : -by_axis;
    }
    const auto& mg = sp.map(sp.index_of("g_0"));
    return -state_[g_offset_ + static_cast<std::size_t>(mg.global(dim_, cell, 0, p))] /
           exokal::measure(*mesh_, dim_, cell);
  }

  int n_rotations() const { return dim_ * (dim_ - 1) / 2; }

  // THE CELL-AVERAGE STRESS TENSOR, reconstructed from the facet tractions the
  // space already carries.
  //
  //     |E| sigma_ij = int_{dE} (sigma n_out)_i (x - x_E)_j
  //
  // which is the divergence theorem applied to sigma_ik d_k (x - x_E)_j, and it
  // is EXACT for a constant stress: the leading facet moment is int_f (sigma n)
  // because the chart has chi_0 = 1, and the remaining factor comes out of the
  // integral. So a mixed method whose stress is piecewise constant reproduces it
  // to round-off, and one whose stress varies is sampled at the facet centroids
  // -- second order, not first, because the moment is the facet MEAN.
  //
  // EXPANDING ABOUT THE CELL CENTROID IS NOT COSMETIC. The alternative, using x
  // itself, differs by x_{E,j} times the net force on the cell -- zero only when
  // the cell is in equilibrium with no body load. Under a reservoir
  // pressurization it is not, and the difference is the whole depletion signal.
  //
  // The result is symmetrized. Symmetry of the stress is imposed WEAKLY in this
  // formulation -- that is what the rotation multiplier gamma is for -- so the
  // raw reconstruction carries an antisymmetric part of the size of the
  // discretization error, and reporting it as stress would be reporting that
  // error as physics.
  std::array<double, 9> cell_stress(Index cell) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto& op = stress_.compact(cell);  // faces and orientation: no dense M
    const exokal::Point xE = exokal::centroid(*mesh_, dim_, cell);
    const double volume = exokal::measure(*mesh_, dim_, cell);

    std::array<double, 9> raw{};
    for (const Index f : op.faces) {
      const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cell, f);
      const exokal::Point xf = exokal::centroid(*mesh_, dim_ - 1, f);
      // int_f (sigma n) against the CANONICAL normal: the leading moment per
      // component for the weak family, and the three mean slots read through
      // the facet frame for the strong one. The incidence turns it outward.
      const std::array<double, 3> force =
          strongly_symmetric() ? strong_facet_force(f, fr)
                               : std::array<double, 3>{
                                     state_[s_offset_ + static_cast<std::size_t>(
                                                            ms.global(dim_ - 1, f, 0, 0))],
                                     state_[s_offset_ + static_cast<std::size_t>(
                                                            ms.global(dim_ - 1, f, 0, 1))],
                                     dim_ > 2 ? state_[s_offset_ + static_cast<std::size_t>(
                                                                       ms.global(dim_ - 1, f, 0, 2))]
                                              : 0.0};
      for (int i = 0; i < dim_; ++i) {
        const double t = fr.incidence * force[static_cast<std::size_t>(i)];
        for (int j = 0; j < dim_; ++j) {
          raw[static_cast<std::size_t>(i * 3 + j)] +=
              t * (xf[static_cast<std::size_t>(j)] - xE[static_cast<std::size_t>(j)]);
        }
      }
    }
    std::array<double, 9> out{};
    for (int i = 0; i < dim_; ++i) {
      for (int j = 0; j < dim_; ++j) {
        out[static_cast<std::size_t>(i * 3 + j)] =
            0.5 *
            (raw[static_cast<std::size_t>(i * 3 + j)] + raw[static_cast<std::size_t>(j * 3 + i)]) /
            volume;
      }
    }
    return out;
  }

  // THE TRACTION ON ANY FACET, interior or boundary, in ambient components and
  // against the facet's CANONICAL normal.
  //
  // In Hellinger-Reissner the facet traction moments ARE primary unknowns, so
  // this is the value on the plane itself rather than a cell-centred stress
  // sampled half a cell away -- which matters most exactly where it is read from
  // a fault, since no amount of refinement along the plane fixes an error
  // across it.
  //
  // It takes no coface, and that is the difference from `normal_traction`: the
  // orientation asked for is the canonical one, which a facet owns by itself,
  // so an INTERIOR facet is a legitimate argument. Going through a coface to
  // find a frame is a boundary accessor and fails on a fault.
  std::array<double, 3> facet_traction(Index facet) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const double area = exokal::measure(*mesh_, dim_ - 1, facet);
    if (strongly_symmetric()) {
      // interior-safe: any coface serves, since a volume facet's canonical
      // frame does not read the cell
      const graphos::CoboundaryOperator cob = graphos::coboundary(mesh_->topology(), dim_ - 1);
      const Index cell = cob.indices[static_cast<std::size_t>(
          cob.offsets[static_cast<std::size_t>(facet)])];
      std::array<double, 3> t = strong_facet_force(facet, FacetFrame::of(*mesh_, dim_, cell, facet));
      for (double& v : t) v /= area;
      return t;
    }
    std::array<double, 3> t{};
    for (int k = 0; k < dim_; ++k) {
      t[static_cast<std::size_t>(k)] =
          state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))] / area;
    }
    return t;
  }

  // THE NORMAL TRACTION on a facet, read through the same form a condition
  // would impose there: n . (sigma n), against the facet's CANONICAL normal and
  // divided by the measure it was integrated against.
  double normal_traction(Index facet) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cofacet_of(*mesh_, dim_, facet), facet);
    if (strongly_symmetric()) {
      // n . int_f (sigma n) is slot 3 alone: chi_0 = 1 and the tangential
      // slots are orthogonal to the normal
      return state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 3, 0))] /
             fr.measure;
    }
    double t = 0.0;
    for (int k = 0; k < dim_; ++k) {
      t += fr.normal[static_cast<std::size_t>(k)] *
           state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))];
    }
    return t / fr.measure;
  }

  // and the traction along an arbitrary direction, which a curved boundary
  // needs: e . (sigma n) with n canonical
  double traction_along(Index facet, const std::array<double, 3>& e) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cofacet_of(*mesh_, dim_, facet), facet);
    if (strongly_symmetric()) {
      const std::array<double, 3> force = strong_facet_force(facet, fr);
      return (e[0] * force[0] + e[1] * force[1] + e[2] * force[2]) / fr.measure;
    }
    double t = 0.0;
    for (int k = 0; k < dim_; ++k) {
      t += e[static_cast<std::size_t>(k)] *
           state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))];
    }
    return t / fr.measure;
  }

  // int_f u_D . basis_b / |f| for the six strong-symmetry facet functionals,
  // with u_D = a + B (x - x_E) about the facet's one cofacet. The basis is
  // {t1, t2, n^(x-x_f)/rho, n chi_0, n chi_1, n chi_2} in the facet's
  // CANONICAL frame -- the one FacetFrame::of shares with the discrete dofs --
  // and the chart comes from the facet's own centred second moments, computed
  // by the same quadrature that evaluates the moments. Degree 4 is exact for
  // every integrand here, so the affine datum is reproduced rather than
  // approximated.
  std::array<double, 6> strong_datum_coefficients(Index f, Index cell,
                                                  const std::array<double, 3>& a,
                                                  const std::array<double, 9>& B) const {
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cell, f);
    const exokal::Point xE = exokal::centroid(*mesh_, dim_, cell);
    const exokal::Point xf = exokal::centroid(*mesh_, dim_ - 1, f);
    const exokal::QuadratureRule qr = exokal::facet_quadrature(*mesh_, dim_, f, 4);

    const auto xi = [&](const exokal::Point& x, int t) {
      double s = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        s += fr.tangent[static_cast<std::size_t>(t)][k] * (x[k] - xf[k]);
      }
      return s;
    };
    double area = 0.0, m11 = 0.0, m12 = 0.0, m22 = 0.0;
    for (std::size_t p = 0; p < qr.weights.size(); ++p) {
      const double w = qr.weights[p];
      const double x1 = xi(qr.points[p], 0), x2 = xi(qr.points[p], 1);
      area += w;
      m11 += w * x1 * x1;
      m12 += w * x1 * x2;
      m22 += w * x2 * x2;
    }
    const exokal::hodge::FacetChart chart = exokal::hodge::facet_chart(area, m11, m12, m22, dim_);
    const double rho = std::sqrt((m11 + m22) / area);

    std::array<double, 6> v{};
    for (std::size_t p = 0; p < qr.weights.size(); ++p) {
      const double w = qr.weights[p];
      const exokal::Point& x = qr.points[p];
      std::array<double, 3> u{};
      for (std::size_t i = 0; i < 3; ++i) {
        u[i] = a[i];
        for (std::size_t j = 0; j < 3; ++j) u[i] += B[i * 3 + j] * (x[j] - xE[j]);
      }
      const std::array<double, 3> dx{x[0] - xf[0], x[1] - xf[1], x[2] - xf[2]};
      const std::array<double, 3> nxdx{fr.normal[1] * dx[2] - fr.normal[2] * dx[1],
                                       fr.normal[2] * dx[0] - fr.normal[0] * dx[2],
                                       fr.normal[0] * dx[1] - fr.normal[1] * dx[0]};
      double ut1 = 0.0, ut2 = 0.0, un = 0.0, urot = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        ut1 += fr.tangent[0][k] * u[k];
        ut2 += fr.tangent[1][k] * u[k];
        un += fr.normal[k] * u[k];
        urot += nxdx[k] * u[k];
      }
      const double x1 = xi(x, 0), x2 = xi(x, 1);
      const double chi1 = chart.a11 * x1;
      const double chi2 = chart.a21 * x1 + chart.a22 * x2;
      v[0] += w * ut1;
      v[1] += w * ut2;
      v[2] += w * urot / rho;
      v[3] += w * un;
      v[4] += w * un * chi1;
      v[5] += w * un * chi2;
    }
    for (double& e : v) e /= area;
    return v;
  }

  // int_f (sigma n) in ambient components against the CANONICAL normal, from
  // the six strong-symmetry slots. The rotation and first-moment slots
  // integrate to zero against a constant -- ∫chi_a = 0 for a >= 1 and the
  // rotation basis is centred -- so the mean is carried by slots 0, 1 and 3
  // in the facet's canonical frame, which FacetFrame::of shares with the
  // discrete basis.
  std::array<double, 3> strong_facet_force(Index facet, const FacetFrame& fr) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto slot = [&](int b) {
      return state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, b, 0))];
    };
    const double s0 = slot(0), s1 = slot(1), s3 = slot(3);
    std::array<double, 3> force{};
    for (std::size_t k = 0; k < 3; ++k) {
      force[k] = s0 * fr.tangent[0][k] + s1 * fr.tangent[1][k] + s3 * fr.normal[k];
    }
    return force;
  }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  ElasticMaterial material_;
  Realization how_;
  Formulation form_{Formulation::weak_symmetry};
  double degeneracy_percent_{-1.0};  // adaptive_vem's scan threshold; negative is unset
  MechanicsBoundary mechanics_;

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::StressOperators stress_;
  StrongDisplacementCoefficients strong_displacement_;
  exokal::forms::TermContext ctx_;
  // the partition, requested before the build and applied inside it
  int n_ranks_{1}, rank_{0};
  std::function<void(std::vector<double>&)> reduce_;
  Distribution distribution_;
  std::unique_ptr<Simulation> sim_;
  solver::SparseSystem system_;
  mutable std::vector<double> rhs_;
  std::vector<double> state_;
  mutable std::vector<double> work_;
  mutable solver::PetscSolver solver_;
  struct DisplacementDatum {
    std::vector<Index> facets;
    std::array<double, 3> constant{};
    std::array<double, 9> gradient{};
  };
  struct Reservoir {
    std::vector<Index> cells;
    double pressure{0.0};
  };
  std::vector<Reservoir> reservoir_;
  CellData reservoir_data_{0};
  double biot_{1.0}, volumetric_compliance_{1.0};
  std::vector<DisplacementDatum> displacement_facets_;
  BoundaryVectorData displacement_data_{0};
  std::vector<Index> prescribed_;
  std::vector<std::size_t> prescribed_forms_, prescribed_dofs_;
  std::size_t s_offset_{0}, u_offset_{0}, g_offset_{0}, n_cells_{0};
  bool load_ready_{false};
  mutable bool factorized_{false};
};

}  // namespace mimetika
