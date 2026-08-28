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
#include "exokal/hodge/hybrid_stress.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/hybrid_interface.hpp"
#include "mimetika/model/compositions/cauchy_mechanics.hpp"
#include "mimetika/model/conditioning.hpp"
#include "mimetika/model/partition.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"

// Cauchy elasticity, stated as data -- the poroelastic model with the flow
// taken out, and the smallest problem that exercises the stress space alone.
//
// Weakly-symmetric mixed form (Hellinger-Reissner), three fields:
//
//     r_sigma = M sigma - D^T u - A^T gamma      the constitutive relation
//     r_u     = + D sigma                        momentum balance
//     r_gamma = + A sigma                        symmetry, imposed weakly
//
// The space carries no symmetry constraint -- that is what makes it usable --
// so gamma is the multiplier enforcing sigma = sigma^T against the rigid
// rotations. Dropping it changes the method rather than simplifying it.
//
// When a poroelastic answer is wrong, this is what says whether the mechanics
// half is. Both of its closed forms hold in any dimension:
//
//     a column    uniaxial extension: constant stress, linear displacement,
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

class CauchyMechanicsModel {
 public:
  // Which discrete stress Hodge, and only the two that are elements.
  //
  //   derham          d copies of the mimetic-BDM plus a rank-one volumetric
  //                   fold-back. Consistency-only: the scalar layer is
  //                   unisolvent, so N is square and nothing is stabilized.
  //                   Any cell type, either dimension.
  //   stabilized_bdm  the same d^2 degrees of freedom reconstructed on the full
  //                   linear tensor space [P_1]^{dxd}, m = d^2(d+1) modes. On a
  //                   simplex D = m and the stabilization vanishes -- there it
  //                   is the conforming AFW/BDM_1 element, checked by congruence
  //                   in exokal's hodge.test_afw_equivalence. On a polytope a
  //                   stabilization remains: 4 on a quadrilateral, 18 on a hex.
  //
  // The third realization exokal offers, derham_rt, is deliberately not here.
  // It is a sound inner product -- unisolvent, positive definite, exact on the
  // compliance energy -- and it is not an element: one constant traction vector
  // per facet cannot control the rigid rotations across a mesh, the inf-sup for
  // gamma degenerates, and the saddle point comes out singular. See
  // tests/model/test_dimensions.cpp, which pins exactly that.
  using Realization = exokal::hodge::StressOperators::Realization;
  using Formulation = exokal::hodge::StressOperators::Formulation;

  CauchyMechanicsModel(const exokal::Mesh& mesh, int cell_dim, ElasticMaterial material,
                        Realization how = Realization::derham_bdm,
                        Formulation form = Formulation::weak_symmetry)
      : mesh_(&mesh), dim_(cell_dim), material_(material), how_(how), form_(form) {
    // derham_rt is unisolvent and still refused; see the realization list above.
    //
    // diagonal_afw carries the same d per facet and is not refused, because it
    // is a different scheme rather than a coarser space: its face rotation is a
    // convention rather than an inf-sup, and exokal admits it in four fields
    // only, which is where its M is diagonal.
    if (how == Realization::derham_rt) {
      throw std::invalid_argument(
          "CauchyMechanicsModel: derham_rt is unisolvent but its weak-symmetry inf-sup "
          "degenerates; use derham_bdm, stabilized_bdm, or diagonal_afw");
    }
    // the blend inherits the demand of its diagonal member
    if ((how == Realization::diagonal_afw || how == Realization::adaptive_afw) &&
        form != Formulation::weak_symmetry_total) {
      throw std::invalid_argument(
          "CauchyMechanicsModel: diagonal_afw needs the total-pressure formulation -- the "
          "three-field compliance couples traction components through the trace");
    }
    // The symmetry axis is one decision: a strongly-symmetric realization
    // under a weak formulation (or the reverse) is refused here as exokal
    // refuses it at build, so the error arrives where the choice was made.
    if (exokal::hodge::StressOperators::strongly_symmetric(how) !=
        exokal::hodge::StressOperators::strongly_symmetric(form)) {
      throw std::invalid_argument(
          "CauchyMechanicsModel: the realization and the formulation must agree on where the "
          "symmetry lives -- the vem family builds strong_symmetry(_total), the rest the weak "
          "pair");
    }
    // the rigid-motion ansatz is a three-dimensional construction: six
    // traction moments per facet and the six rigid motions per cell
    if (exokal::hodge::StressOperators::strongly_symmetric(how) && cell_dim != 3) {
      throw std::invalid_argument(
          "CauchyMechanicsModel: the strongly-symmetric vem family is a 3D construction");
    }
    if ((how == Realization::diagonal_vem || how == Realization::adaptive_vem) &&
        form != Formulation::strong_symmetry_total) {
      throw std::invalid_argument(
          std::string("CauchyMechanicsModel: ") + exokal::hodge::StressOperators::name(how) +
          " needs strong_symmetry_total -- the plain compliance couples traction components "
          "through the trace and cannot be diagonal");
    }
  }

  // The adaptive_vem threshold: scan for metrically degenerate cells at this
  // percentage of the node-star mean and give them the diagonal star, eta = 0.
  // The same contract adaptive_rt carries for the flux: eta is derived --
  // ones, with the flagged cells zeroed -- never given as a field. Unset,
  // exokal scans at its own default_degeneracy_percent.
  void set_degeneracy_percent(double percent) { degeneracy_percent_ = percent; }

  // The second selector, by conditioning: a cell whose stabilized vem block
  // has lambda_max / lambda_min above this takes the diagonal star as well.
  // Composes with the scan -- either flag zeroes eta -- at the cost of one
  // probe build of the stabilized member. Same contract as adaptive_rt.
  void set_cond_threshold(double cond) { cond_threshold_ = cond; }

  // The facet-jump stabilization of diagonal_afw's rotation multiplier, c in
  //
  //     g_{E,f} = c mu |f| delta_{E,f} ,   J_ij = -(g_{E1,f} + g_{E2,f}) ,
  //
  // assembled into the rotation row as -J gamma by the rotation_jump_facet
  // term. J annihilates constant rotations, so the linear patch test is
  // unchanged; the wrench star is stable at c = 0 and the term buys
  // definiteness in the multiplier block for a constant the physics does not
  // supply. Off by default. Not available under --hybrid: a cofacet coupling
  // does not eliminate cell by cell.
  void set_rotation_jump(double c) {
    if (c < 0.0) {
      throw std::invalid_argument("CauchyMechanicsModel: the rotation jump constant is nonnegative");
    }
    if (c > 0.0 && how_ != Realization::diagonal_afw) {
      throw std::invalid_argument(
          std::string("CauchyMechanicsModel: ") + exokal::hodge::StressOperators::name(how_) +
          " carries no rotation jump -- the stabilization belongs to diagonal_afw's rotation "
          "multiplier");
    }
    rotation_jump_ = c;
  }

  // the constant the operators were built at
  double rotation_jump() const { return rotation_jump_; }

  // how many cells the conditioning selector switched, as built
  std::size_t n_ill_conditioned() const { return n_ill_conditioned_; }

  // The selection as built, one value per cell: 1 is the stabilized vem
  // product, 0 the diagonal star. Empty unless the realization is
  // adaptive_vem; the field to write next to the solution.
  const std::vector<double>& eta() const {
    if (sim_ == nullptr) {
      throw std::logic_error("CauchyMechanicsModel: not built yet; call build() or solve() first");
    }
    return stress_.eta();
  }

  // The layout is not the symmetry, and exokal separates them: the wrench
  // layout -- one d(d+1)/2 rigid-motion moment vector per facet, rather than d
  // copies of a scalar layout -- is what the strong family always uses and what
  // diagonal_afw brings to the weak axis. Anything counting unknowns per facet
  // asks this; anything asking whether a rotation field exists asks the other.
  bool wrench_layout() const {
    return exokal::hodge::StressOperators::wrench_layout(how_);
  }
  int facet_dofs() const {
    return exokal::hodge::StressOperators::facet_dofs(how_, dim_);
  }
  bool strongly_symmetric() const {
    return exokal::hodge::StressOperators::strongly_symmetric(how_);
  }

  // The validity gate of the diagonal star, which exokal states and leaves to
  // the consumer: a facet the cell centroid does not see squarely carries a
  // non-positive weight -- delta = (x_f - x_E).n <= 0 -- and the assembled M
  // is then not positive definite. Condensation divides by those entries and
  // the solve collapses to a near-zero field that still reports CONVERGED:
  // the one failure mode worse than a wrong answer. So the count is exposed,
  // per cell with any offending facet, for the driver to refuse or report.
  std::size_t n_invalid_star() const {
    if (sim_ == nullptr) {
      throw std::logic_error("CauchyMechanicsModel: not built yet; call build() or solve() first");
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

  // The total pressure p = lambda div u, one scalar per cell, and a field in
  // the four-field formulation rather than a post-processing of the stress.
  // Asking for it in three fields asks for something that was never solved for.
  double total_pressure(Index cell) const {
    if (form_ != Formulation::weak_symmetry_total && form_ != Formulation::strong_symmetry_total) {
      throw std::logic_error(
          "CauchyMechanicsModel::total_pressure: this formulation has no total pressure; build "
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

  // The rank-one term that makes the THREE-FIELD stress norm lambda-free.
  //
  //     (C^-1 sigma, sigma) = (1/2mu) |sigma|^2 - (a/2mu) (tr sigma)^2 ,
  //     a = lambda / (2 mu + d lambda) ,
  //
  // so the compliance stays BOUNDED as lambda grows -- a -> 1/d -- and becomes
  // singular on the trace. A norm built from it loses the volumetric direction
  // the operator still has, and the map then stops relating to the residual it
  // is preconditioning: measured, GMRES reports CONVERGED_RTOL at lambda = 1e8
  // while the answer is 8e-2 away from the factorization.
  //
  // Adding (a/2mu)(tr sigma)^2 back leaves (1/2mu)|sigma|^2, the plain L^2 mass,
  // which does not see lambda at all. T is the cell's trace functional, so the
  // correction is rank one per cell, weight a / (2 mu |E|).
  //
  // Empty for the total forms: there p carries the trace and the norm already
  // has c_p^-1 ||(2 mu)^-1 tr sigma||^2 through the pressure row.
  struct NormTraceTerm {
    std::vector<Index> dofs;
    std::vector<double> row;
    double weight{0.0};
  };
  std::vector<NormTraceTerm> norm_trace_terms() const {
    std::vector<NormTraceTerm> out;
    if (form_ == Formulation::weak_symmetry_total ||
        form_ == Formulation::strong_symmetry_total) {
      return out;
    }
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto s_base = static_cast<Index>(sp.offset(sp.index_of("s_0")));
    const int nb = stress_.moments_per_facet();
    const int nc = wrench_layout() ? 1 : dim_;
    const double mu = material_.shear;
    const auto d = static_cast<double>(dim_);
    out.reserve(static_cast<std::size_t>(mesh_->topology().count(dim_)));
    for (Index e = 0; e < mesh_->topology().count(dim_); ++e) {
      const auto& c = stress_.compact(e);
      if (c.T.rows() == 0 || c.faces.empty() || !(c.volume > 0.0)) continue;
      const double lam = lame_per_cell_.empty()
                             ? material_.lame
                             : lame_per_cell_[static_cast<std::size_t>(e)];
      const double a = lam / (2.0 * mu + d * lam);
      if (!(a > 0.0)) continue;  // lambda = 0: the compliance already IS the mass
      NormTraceTerm t;
      t.weight = a / (2.0 * mu * c.volume);
      for (std::size_t slot = 0; slot < c.faces.size(); ++slot) {
        for (int b = 0; b < nb; ++b) {
          for (int k = 0; k < nc; ++k) {
            const auto local = slot * static_cast<std::size_t>(nb * nc) +
                               static_cast<std::size_t>(b * nc + k);
            if (local >= c.T.cols()) continue;
            const double v = c.T(0, local);
            if (v == 0.0) continue;
            t.dofs.push_back(s_base + ms.global(dim_ - 1, c.faces[slot], b, k));
            t.row.push_back(v);
          }
        }
      }
      if (!t.dofs.empty()) out.push_back(std::move(t));
    }
    return out;
  }

  // Piecewise-constant Lame parameters, in place of the uniform pair.
  //
  // The material enters star_C and nothing else: d is incidence and does not
  // see it, so a jump in mu or lambda is a jump in the metric of the stress
  // (d-1)-cochains alone.
  //
  // LAMBDA ONLY, for now. The four-field term carries the trace coupling as a
  // single (2 mu)^-1 read from the composition, so a per-cell MU would be
  // applied by the star and not by that row, and the two would disagree. The
  // volumetric contrast a per-cell lambda gives is the one the total-pressure
  // form exists to absorb, and it is the one this admits.
  void set_lame_per_cell(std::vector<double> lam) {
    if (!lam.empty() && lam.size() != static_cast<std::size_t>(mesh_->topology().count(dim_))) {
      throw std::invalid_argument("CauchyMechanicsModel::set_lame_per_cell: one value per cell");
    }
    for (const double v : lam) {
      if (!(v > 0.0)) {
        throw std::invalid_argument(
            "CauchyMechanicsModel::set_lame_per_cell: lambda must be positive");
      }
    }
    lame_per_cell_ = std::move(lam);
  }
  const std::vector<double>& lame_per_cell() const { return lame_per_cell_; }
  // Non-const, for the one caller that configures the simulation rather than
  // reading it: the partition, which tells it which sites to assemble.
  Simulation& simulation() { return *sim_; }
  const Simulation& simulation() const { return *sim_; }
  const solver::SparseSystem& system() const { return system_; }
  const std::vector<double>& rhs() const { return rhs_; }
  std::size_t n_cells() const { return n_cells_; }
  // how many cells needed a stabilization: zero on a simplex mesh for either
  // realization, by construction
  // A BODY FORCE PER CELL, d components cell-major: div sigma + b = 0.
  void set_body_force(std::vector<double> b) { body_force_ = std::move(b); }
  const std::vector<double>& body_force() const { return body_force_; }

  std::size_t n_stabilized() const { return stress_.n_stabilized(); }
  const exokal::hodge::StressOperators& stress_operators() const { return stress_; }

  // Share this model out over `n_ranks` processes. Nothing happens until
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
    // The partition comes first, because the products below are per cell and
    // are the bulk of the work: a process builds its own and no others.
    if (n_ranks_ > 1) distribution_ = partition_cells(*mesh_, dim_, n_ranks_, rank_);
    const std::vector<char>* only =
        distribution_.assembled_cells.empty() ? nullptr : &distribution_.assembled_cells;
    // the K-independent mode selection, which only the de Rham product has
    const bool derham = how_ == Realization::derham_bdm;
    if (derham) geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    // The blend's selection, derived rather than given: ones, with 0 on the
    // cells the scan flags at the caller's threshold. exokal forces its own
    // default-threshold zeros on top either way, so a caller can only widen
    // the set. Both blends carry one -- adaptive_vem on the strong axis,
    // adaptive_afw on the weak -- which is the predicate exokal uses too.
    const bool adaptive =
        how_ == Realization::adaptive_vem || how_ == Realization::adaptive_afw;
    if ((degeneracy_percent_ >= 0.0 || cond_threshold_ >= 0.0) && !adaptive) {
      throw std::invalid_argument(
          "CauchyMechanicsModel: the degeneracy and conditioning thresholds are the "
          "cell selection of adaptive_vem and adaptive_afw");
    }
    std::vector<double> eta;
    if (adaptive && degeneracy_percent_ >= 0.0) {
      eta.assign(static_cast<std::size_t>(c.count(dim_)), 1.0);
      for (const Index e : exokal::degenerate_cell_ids(*mesh_, dim_, degeneracy_percent_)) {
        eta[static_cast<std::size_t>(e)] = 0.0;
      }
    }
    n_ill_conditioned_ = 0;
    if (adaptive && cond_threshold_ >= 0.0) {
      // the probe: the stabilized vem member on every cell; a cell exokal
      // already put on the diagonal star is compact and is not judged again
      if (eta.empty()) eta.assign(static_cast<std::size_t>(c.count(dim_)), 1.0);
      const std::vector<double> ones(eta.size(), 1.0);
      const exokal::hodge::StressOperators probe = exokal::hodge::StressOperators::build(
          *mesh_, dim_, exokal::hodge::Coefficient::uniform(material_.shear), lame_coefficient(),
          how_, form_, nullptr, only, &ones);
      static const exokal::numerics::Dense empty;
      for (std::size_t e = 0; e < eta.size(); ++e) {
        if (!probe.eta().empty() && probe.eta()[e] == 0.0) eta[e] = 0.0;
      }
      n_ill_conditioned_ = cond_selection(
          eta, cond_threshold_,
          [&](std::size_t e) -> const exokal::numerics::Dense& {
            const auto& cell = probe.compact(static_cast<Index>(e));
            return cell.diag.empty() ? cell.M : empty;
          });
    }
    stress_ = exokal::hodge::StressOperators::build(
        *mesh_, dim_, exokal::hodge::Coefficient::uniform(material_.shear), lame_coefficient(),
        how_, form_,
        derham ? &geometry_ : nullptr, only, eta.empty() ? nullptr : &eta, rotation_jump_);
    ctx_.provide("stress_operators", stress_);

    displacement_data_ = BoundaryVectorData(static_cast<std::size_t>(c.count(dim_ - 1)));
    for (const auto& d : displacement_facets_) {
      displacement_data_.set_affine(d.facets, d.constant, d.gradient);
    }
    ctx_.provide("boundary_displacement", displacement_data_);

    // The strong datum, expanded at build: the six moments of the affine
    // u_D = a + B (x - x_E) against the facet basis, divided by the |f| Gram.
    // The operators do not carry the chart and second moments per cell, and
    // the datum is affine, so a quadrature that is exact for these integrals
    // runs once here and the boundary term reads numbers -- the affine datum
    // stays exact, as it is in the weak family.
    // The wrench layout decides this, not the symmetry. The coefficients are
    // (1/|f|) int_f u_D . basis_b against the facet's rigid-motion basis, which
    // is what a wrench-layout space carries whether its symmetry is strong
    // (stabilized_vem, diagonal_vem) or weak (diagonal_afw). Keying it off the
    // symmetry instead writes the componentwise datum into a space that has no
    // components, and the rhs comes out identically zero.
    if (wrench_layout() && !displacement_facets_.empty()) {
      strong_displacement_ = StrongDisplacementCoefficients(
          static_cast<std::size_t>(c.count(dim_ - 1)), static_cast<std::size_t>(facet_dofs()));
      // The coboundary is built once. cofacet_of rebuilds it per call, which
      // is O(mesh) each time -- 5k boundary facets on a 22k-cell mesh then
      // spend minutes recomputing the same operator that takes milliseconds
      // to build once.
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
    // THE BODY FORCE AS THE ROW WANTS IT. The momentum row is Dv sigma, the
    // divergence divided by the measure, so the MEAN force over the cell is the
    // load -- no measure appears here, unlike the flux balance, which is an
    // integral and takes f|E|.
    body_force_data_ = CellVectorData(static_cast<std::size_t>(c.count(dim_)),
                                      static_cast<std::size_t>(dim_));
    bool any_force = false;
    const auto dd = static_cast<std::size_t>(dim_);
    for (std::size_t e = 0; (e + 1) * dd <= body_force_.size(); ++e) {
      for (std::size_t k = 0; k < dd; ++k) {
        const double b = body_force_[e * dd + k];
        if (b != 0.0) any_force = true;
        body_force_data_.set(static_cast<Index>(e), k, b);
      }
    }
    ctx_.provide("body_force", body_force_data_);

    // The space follows the star: d^2 traction moments per facet for both of
    // these, read off the operators rather than restated, so the layout and the
    // product cannot drift apart.
    physics::ModelOptions o;
    // The wrench layout is d(d+1)/2 moments on one component: a facet carries
    // the rigid-motion moment vector whole, which the strong branch of the
    // package already hardcodes as moments(dim, dim-1, 6, 1). The weak member
    // of that layout -- diagonal_afw -- has to be told the same shape, since
    // its branch reads these options.
    o.traction_moments = stress_.moments_per_facet();
    o.traction_components = wrench_layout() ? 1 : dim_;
    // And the field count follows the formulation, read off the operators for
    // the same reason: the field roster is a property of the product that was
    // built.
    o.total_pressure = stress_.formulation() == Formulation::weak_symmetry_total ||
                       stress_.formulation() == Formulation::strong_symmetry_total;
    o.strong_symmetry = strongly_symmetric();
    // the half weights are on the operators; this attaches the facet term that
    // reads them
    o.rotation_jump = rotation_jump_ > 0.0;
    // the four-field term keeps (2 mu)^-1 as its compliance, so mu travels with
    // the composition rather than being looked up from a slot
    o.shear_modulus = material_.shear;
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("linear_elasticity", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    // the natural displacement datum, attached only when one is actually given
    if (!displacement_facets_.empty()) {
      sim_->model().add(
          wrench_layout() ? "strong_prescribed_displacement" : "prescribed_displacement",
          exokal::forms::On::all(), {});
    }
    if (!reservoir_.empty()) {
      exokal::forms::Params rp;
      rp.set("biot", biot_);
      rp.set("volumetric_compliance", volumetric_compliance_);
      sim_->model().add("reservoir_pressurization", exokal::forms::On::all(), rp);
    }
    if (any_force) sim_->model().add("body_force", exokal::forms::On::all(), {});

    // the conditions resolve against the space, which is what gives each of
    // them its dofs, and the strong ones then hand their forms to the
    // constraint set
    const auto& sp = sim_->epoch().stratum(0).space();
    mechanics_.resolve(*mesh_, dim_, sp);
    mechanics_.impose(sim_->constraints());

    // the prescribed fracture traction, registered with a placeholder value:
    // the structure is what freeze_constraints needs, and the values move later
    {
      const auto& ms = sp.map(sp.index_of("s_0"));
      const auto s_base = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
      // the strong family carries its six moments on one scalar layout, the
      // weak one `moments` per each of d components
      const int nb = stress_.moments_per_facet();
      const int nk = wrench_layout() ? 1 : dim_;
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

    // diagonal_afw's patch contract.
    //
    // For a linear displacement the interpolant satisfies every row but the
    // linear slots of the boundary facets: on an interior facet the two
    // cofacets' first moments cancel, and on a boundary facet the datum's P1
    // variation lands on a slot no diagonal weight closes. Pinned to zero the
    // datum enters through the facet means alone, the lowest-order imposition,
    // and the interpolant is the discrete solution exactly on every
    // face-orthogonal mesh, unequal boxes included. Left free the error is
    // O(1) -- 2.5e-01 in u on a box of unequal hexahedra.
    //
    // The mask is exokal's, not a second copy of the rule: afw_boundary_pins
    // indexes f*d*d + b*d + k, the layout ms.global reads.
    // The blend carries the contract on the cells it selected, and only those:
    // a boundary facet whose cofacet kept the stabilized product needs no pin
    // and is harmed by one, its linear slots being where that product carries
    // the datum. eta as BUILT is the selection, exokal's forced zeros included.
    if (how_ == Realization::diagonal_afw || how_ == Realization::adaptive_afw) {
      const std::vector<char> mask = exokal::hodge::afw_boundary_pins(*mesh_, dim_);
      const std::vector<double>& selection = stress_.eta();  // empty: every cell
      const graphos::CoboundaryOperator up = graphos::coboundary(c, dim_ - 1);
      const auto& ms = sp.map(sp.index_of("s_0"));
      const auto s_base = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
      const auto dd = static_cast<std::size_t>(dim_);
      const std::vector<std::size_t> already(prescribed_dofs_.begin(), prescribed_dofs_.end());
      for (Index f = 0; f < c.count(dim_ - 1); ++f) {
        for (std::size_t b = 0; b < dd; ++b) {
          for (std::size_t k = 0; k < dd; ++k) {
            if (mask[static_cast<std::size_t>(f) * dd * dd + b * dd + k] == 0) continue;
            if (!selection.empty()) {
              const auto at = static_cast<std::size_t>(up.offsets[static_cast<std::size_t>(f)]);
              const Index cell = up.indices[at];  // a pinned facet has one cofacet
              if (selection[static_cast<std::size_t>(cell)] != 0.0) continue;
            }
            const auto d = static_cast<Index>(
                s_base + static_cast<std::size_t>(
                             ms.global(dim_ - 1, f, static_cast<int>(b), static_cast<int>(k))));
            // a facet carrying a prescribed traction is already pinned there
            if (std::find(already.begin(), already.end(), static_cast<std::size_t>(d)) !=
                already.end()) {
              continue;
            }
            sim_->constraints().pin(d, 0.0);
          }
        }
      }
    }
    sim_->freeze_constraints();

    // The partition, applied where the space exists. Asked for before the
    // build, because the assembly below is the thing it divides.
    if (n_ranks_ > 1) {
      add_dof_ownership(distribution_, *mesh_, dim_, sim_->epoch(), n_ranks_, rank_);
      // assemble the halo as well, write only what this process owns: the
      // rows it owns are then complete without a single message.
      sim_->distribute_over(distribution_.assembled_cells, distribution_.assembled_facets,
                            distribution_.owned_dofs, reduce_);
    }

    // Reserve before assembling, and hand the storage over afterwards.
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
    // the mesh: reading it here reserves nothing at all.
    std::size_t nnz = 0;
    const Index n_cells = mesh_->topology().count(dim_);
    for (Index e = 0; e < n_cells; ++e) {
      // counted on the compact cell: reading M here would materialize the
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

    // The load. Everything the terms contribute that does not depend on the
    // unknowns is a load, and the residual at the zero state is exactly minus
    // that -- so it is read off rather than re-derived. The strongly imposed
    // rows carry their own datum, scaled as their equation is.
    std::vector<double> r;
    sim_->state().assign(sim_->n_dofs(), 0.0);
    sim_->residual(r);
    rhs_.assign(sim_->n_dofs(), 0.0);
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      // -r is a contribution and is summed across the processes; a constrained
      // row is a replacement and is written by whichever process owns it, so
      // the others leave it at zero rather than adding a copy of it.
      const bool mine = sim_->owned_dofs().empty() || sim_->owned_dofs()[i] != 0;
      rhs_[i] = sim_->constraints().pinned(i)
                    ? (mine ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i) : 0.0)
                    : -r[i];
    }

    // The factorization for S is deferred, because most problems never ask for
    // it. S exists for prescribed tractions -- the contact iteration reuses one
    // factorization across many right-hand sides -- and a model that prescribes
    // none never evaluates it. Taking it here makes every build pay a full
    // direct solve of the whole saddle point: on a mesh of tens of thousands of
    // polyhedra that is minutes and gigabytes of fill.
    factorized_ = false;
    load_ready_ = true;
    work_.assign(sim_->n_dofs(), 0.0);
    s_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
    u_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
    // the strong family has no rotation field: symmetry lives in the space
    g_offset_ = strongly_symmetric() ? 0 : static_cast<std::size_t>(sp.offset(sp.index_of("g_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  // Move the load without refactorizing.
  //
  // The system matrix never depends on the pressure. A depletion is a load: it
  // enters the stress row as alpha T^T p and touches the right-hand side alone,
  // while the matrix is fixed by the mesh, the moduli and which facets are
  // prescribed. So a sweep over depletion levels -- which is what benchmarks 2
  // and 3 are -- needs one factorization, not one per level.
  //
  // Rebuilding the model per level re-runs the whole construction and a direct
  // factorization of a large system on every step of an outer iteration, which
  // for a slip-weakening branch tracker is hundreds of times per benchmark.
  // The Python reference caches exactly this.
  void set_depletion(double pressure) {
    if (!load_ready_) throw std::logic_error("set_depletion: call build() first");
    for (Reservoir& r : reservoir_) r.pressure = pressure;
    reservoir_data_ = CellData(n_cells_);
    for (const auto& r : reservoir_) reservoir_data_.set(r.cells, r.pressure);
    ctx_.provide("reservoir_pressure", reservoir_data_);

    // the residual at the zero state is minus the load, and only it moved
    std::vector<double> r(sim_->n_dofs(), 0.0);
    sim_->state().assign(sim_->n_dofs(), 0.0);
    sim_->residual(r);
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      // as in build(): -r is summed across the processes, a constrained row is
      // written by its owner alone.
      const bool mine = sim_->owned_dofs().empty() || sim_->owned_dofs()[i] != 0;
      rhs_[i] = sim_->constraints().pinned(i)
                    ? (mine ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i) : 0.0)
                    : -r[i];
    }
  }

  // The trace operator: the displacement jump across an interior facet, as the
  // adjoint of the divergence and the asymmetry.
  //
  //     [| D^T u + A^T gamma |]_f = (D^T u + A^T gamma)_f^+ - (...)_f^-
  //
  // and the right-hand side is not two evaluations and a subtraction. D maps
  // facet tractions to cell vectors, so its adjoint maps cell displacements
  // back onto facets carrying each cell's outward incidence on the facet; the
  // two cofaces of an interior facet therefore enter with opposite signs and the
  // assembled row is the difference. The same holds for A and the rotation,
  // which supplies the rigid-rotation part of the displacement field the facet
  // sees.
  //
  // What is returned is the full constitutive row of the unfractured system,
  //
  //     jump_f = -( M sigma - D^T u - A^T gamma )_f ,
  //
  // in the ambient traction-moment components of facet f. The M sigma term is
  // not optional and the bonded case is the proof: on an interior facet with no
  // fracture the displacement is continuous, the two cofaces' boundary terms
  // cancel, and this residual vanishes -- which is exactly the statement
  // [[u]] = 0. Dropping M sigma leaves 4e-2 there on a unit column instead of
  // round-off.
  //
  // Three properties, each load bearing:
  //
  //   * It is linear in the state, so it applies even where that row has been
  //     replaced by a contact constraint. The equation is gone; the functional
  //     it expressed is not.
  //   * It must be the unfractured row. Where a constraint replaced it, the
  //     fractured residual is zero at the solution, whereas this residual is
  //     precisely the jump it exists to extract.
  //   * No Gram^{-1}. A traction degree of freedom is a moment
  //     m = int_f (sigma n) b, so recovering its pointwise values needs
  //     Gram^{-1} m. The jump term int_f [[u]].(tau n) is paired against that
  //     moment -- writing (tau n) = sum_b phi_b b_b gives m = Gram phi -- so the
  //     pairing already carries a Gram^{-1} and what emerges are the expansion
  //     coefficients of the jump. A second inversion divides by |f| and the slip
  //     then grows like 1/h under refinement: a mesh-dependent answer.
  //
  // It is a property of the discretization and not of contact, which is why it
  // lives here: any consumer wanting relative motion across a facet -- a
  // fracture, a material interface, a post-processing -- wants this functional.
  // PoroelasticModel carries the same method with the Biot term added.
  std::vector<double> trace(Index facet, const std::vector<double>& z) const {
    // the contact machinery reads the componentwise (moment, component) facet
    // blocks; the wrench basis -- the strong family's and diagonal_afw's -- is
    // not wired through it yet
    if (wrench_layout()) {
      throw std::logic_error(
          "CauchyMechanicsModel::trace: not implemented for the wrench-layout realizations");
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

    // Both cofaces, each contributing through its own local operators. The
    // outward incidence is already inside Dv and As -- StressOperators put it
    // there when it converted to the canonical basis -- so summing the two
    // sides is the jump, with no sign applied here.
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
        // M sigma, over every facet of this cell: the compliance couples the
        // fracture facet to the cell's other faces
        for (std::size_t j = 0; j < D; ++j) {
          // within a facet block the ProductSpace orders the component fastest
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
    // The sign is fixed by the compressed case, not by the bonded one. The
    // residual vanishes wherever the material is continuous, and zero has no
    // sign, so the bonded identity cannot distinguish g from -g. What does:
    // prescribe a zero traction on a fault under compressive boundary
    // displacement and the two halves must overlap, so the gap has to come out
    // negative there. With the opposite convention the solver reads that as an
    // open fault, settles in one iteration carrying no load, and reports a
    // converged answer satisfying g >= 0, t <= 0 and g t = 0 -- formally
    // Signorini, and the wrong branch of it.
    //
    // The same choice is what makes the outer iteration contract. Measured on a
    // compressed column, dg/dt = -0.16 in this convention, and the Uzawa
    // multiplier |1 + r dg/dt| is below one exactly when that slope is
    // negative; with the sign the other way no augmentation converges.
    for (double& v : row) v = -v;
    return row;
  }

  // ---- the affine solution operator ------------------------------------
  //
  // Prescribe the traction on a set of facets, as an essential condition: in
  // the mixed form sigma|_f is a degree of freedom, so "the fracture carries
  // this traction" is a Dirichlet condition on the unknown and not a penalty.
  // Registered before build(), because it changes which equations the system
  // has; the values may then move freely, which is what the outer iteration
  // needs.
  void prescribe_traction(std::vector<Index> facets) { prescribed_ = std::move(facets); }

  // Reservoir pressurization: a pore-pressure change on a set of cells,
  // entering the mechanics as a load.
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

  // Prescribe a displacement on a set of boundary facets: u = a + B (x - x_E).
  //
  // Natural, not essential. In the Hellinger-Reissner form the interior
  // displacement enters the stress row as -D^T u, so on a boundary facet a
  // prescribed displacement takes its place in the right-hand side -- there is
  // no displacement degree of freedom on a facet to constrain. The affine datum
  // is exact: both integrals it needs are already in the stress operators'
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
  // A_CC does not depend on m -- prescribing removes the same rows and columns
  // whatever the values are -- so the operator is factorized once and every
  // later evaluation is a back-substitution against a moved right-hand side;
  // contact iterates without touching the global system again.
  //
  // `moments` runs facet-major over prescribed_traction(), d * nb entries each,
  // in the ProductSpace order (component fastest).
  const std::vector<double>& solution_operator(const std::vector<double>& moments) const {
    const std::size_t ndf = static_cast<std::size_t>(facet_dofs());
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

  // ---- the hybridized route ------------------------------------------------
  //
  // A second elimination, and a different system. The mixed form condenses its
  // stress only when the star is diagonal; hybridization takes any of them.
  // Each cell keeps its own facet stress, traction continuity moves to a
  // multiplier on the facets, and what a solver sees is the interface system
  // alone -- SPD once a facet is pinned, so a conjugate gradient applies where
  // the condensed mixed system wanted MINRES and, on a hybrid mesh, converged
  // under nothing.
  //
  // The boundary roles swap: the multiplier is the facet displacement, so a
  // Dirichlet facet is a pinned multiplier and traction data enters the free
  // rows naturally -- the opposite of the mixed form. This first cut pins
  // every boundary facet to zero, which is the homogeneous reference exokal's
  // assemblers are written against; an inhomogeneous datum needs its pinned
  // block moved to the load and is not done here.
  //
  // The recovery is exokal's and cell-local, and it reports `jump`: the worst
  // disagreement between the two cofacet recoveries of a shared facet. It is
  // the continuity the multiplier enforces -- a solve that left the interface
  // unconverged shows there.
  struct HybridReport {
    solver::SolveReport solve;
    std::size_t multipliers{0};
    double jump{0.0};
  };

  HybridReport hybridized(solver::LinearSolver& linear) {
    // A NATURAL TRACTION HAS NOWHERE TO GO. The interface load is built from
    // the cell rows and the pinned multipliers alone -- exokal's
    // hybrid_interface_load takes fu, fp and lambda_data, and no sigma-row
    // term -- so a prescribed traction is silently dropped and the answer
    // comes back zero where it should not be. Refused rather than shipped:
    // measured, a traction-driven column returns 0 with the interface solve
    // reporting 0 iterations.
    for (std::size_t i = 0; i < mechanics_.size(); ++i) {
      if (dynamic_cast<const TractionBC*>(&mechanics_.at(i)) != nullptr) {
        throw std::invalid_argument(
            "CauchyMechanicsModel::hybridized: a prescribed traction is a sigma-row "
            "datum and the hybridized interface load carries no sigma-row term -- it "
            "would be dropped; solve this one monolithically");
      }
    }
    if (state_.empty()) state_.assign(sim_->n_dofs(), 0.0);
    const exokal::hodge::HybridStressOperators hops =
        exokal::hodge::HybridStressOperators::build(*mesh_, dim_, stress_, material_.shear);
    // The free mask is the boundary condition. A facet whose displacement is
    // prescribed carries a pinned multiplier -- that datum is the essential
    // condition here -- and every other facet is free, interior and traction
    // alike, because a traction is natural in the hybridized form.
    const graphos::Complex& topo = mesh_->topology();
    const auto n_facets = static_cast<std::size_t>(topo.count(dim_ - 1));
    std::vector<char> free(n_facets, 1);
    std::vector<double> pinned(n_facets * hops.facet_dofs(), 0.0);
    for (const auto& d : displacement_facets_) {
      for (const Index f : d.facets) {
        free[static_cast<std::size_t>(f)] = 0;
        if (!wrench_layout()) {
          // THE COMPONENTWISE DATUM: d moments on each of d components, so the
          // multiplier is a full vector P_1 field on the facet -- 9 entries in
          // space against the wrench's 6, because the multiplier spans the
          // NORMAL TRACE space and a componentwise traction's trace is a linear
          // vector field, not a rigid motion. Its entry is the datum's
          // expansion on {chi_b e_k}, Gram^-1 int_f u_D chi_b, which is exactly
          // what the monolithic term places in the stress row and in the same
          // order, component fastest.
          const Index cell = cofacet_of(*mesh_, dim_, f);
          const auto& cc = stress_.compact(cell);
          std::size_t slot = cc.faces.size();
          for (std::size_t j = 0; j < cc.faces.size(); ++j) {
            if (cc.faces[j] == f) { slot = j; break; }
          }
          if (slot == cc.faces.size() || slot >= cc.moment.size()) continue;
          const exokal::numerics::Dense& mom = cc.moment[slot];
          const exokal::numerics::Dense& gram = cc.facet_gram[slot];
          const std::size_t nb = mom.rows();
          const std::size_t nc = mom.cols() - 1;
          for (std::size_t b = 0; b < nb; ++b) {
            for (std::size_t k = 0; k < nc; ++k) {
              double moment = displacement_data_.constant_at(f, k) * mom(b, 0);
              for (std::size_t q = 0; q < nc; ++q) {
                moment += displacement_data_.gradient_at(f, k, q) * cc.scale * mom(b, q + 1);
              }
              pinned[static_cast<std::size_t>(f) * hops.facet_dofs() + b * nc + k] =
                  -moment / gram(b, b);
            }
          }
          continue;
        }
        for (std::size_t b = 0; b < hops.facet_dofs() && b < 6; ++b) {
          // The sign is the model's, not exokal's. exokal's saddle puts a
          // pinned datum on the sigma-row as -s times the moment vector, and
          // its multiplier is then the displacement itself; this model's
          // boundary term places the same datum as +s (boundary_terms.hpp:
          // r -= s coeff), the convention every monolithic patch test is
          // exact under. Handing exokal the coefficients unchanged solves the
          // patch u = x to u = -x, stress and traction with it, a ratio of
          // exactly -1 on every facet; the multiplier that reproduces the
          // monolithic answer is the negated coefficient vector.
          pinned[static_cast<std::size_t>(f) * hops.facet_dofs() + b] =
              strong_displacement_.applies(f) ? -strong_displacement_.at(f, b) : 0.0;
        }
      }
    }

    // the cell loads, read off the assembled residual: the kinematic rows in
    // the order the local saddle holds them, then the hydrostatic row
    const std::size_t nk = hops.cell(0).n_fields - hops.cell(0).n_p;
    const std::size_t n_p = hops.cell(0).n_p;
    // the hydrostatic field's offset, read off the space rather than cached:
    // it exists only under a total formulation
    const auto& sp = sim_->epoch().stratum(0).space();
    const std::size_t p_offset =
        n_p != 0 ? static_cast<std::size_t>(sp.offset(sp.index_of("p_0"))) : 0;
    const auto cells = static_cast<std::size_t>(n_cells());
    // The kinematic load is not one contiguous run. The local saddle wants the
    // divergence rows then the asymmetry rows, and the model keeps those as
    // separate fields -- u_0 with d per cell, then g_0 with d(d-1)/2 -- laid
    // out cell-major within each. Reading nk in one stride from u_0 walks off
    // the end of it into g_0 and, on the last cells, off the array: that is a
    // segfault.
    // Read the widths off the space. The wrench family's u is the six-component
    // displacement screw and carries no rotation field at all; the
    // componentwise family's u is d wide with the rotation beside it. Assuming
    // either shape asks for a field that does not exist.
    const std::size_t n_u = sp.map(sp.index_of("u_0")).size() / cells;
    const std::size_t n_g = nk - n_u;  // zero where the symmetry is strong
    const std::size_t g_offset =
        n_g != 0 ? static_cast<std::size_t>(sp.offset(sp.index_of("g_0"))) : 0;
    std::vector<double> fu(cells * nk, 0.0), fp(n_p != 0 ? cells : 0, 0.0);
    for (std::size_t e = 0; e < cells; ++e) {
      for (std::size_t r = 0; r < n_u; ++r) fu[e * nk + r] = rhs_[u_offset_ + e * n_u + r];
      for (std::size_t r = 0; r < n_g; ++r) fu[e * nk + n_u + r] = rhs_[g_offset + e * n_g + r];
      if (n_p != 0) fp[e] = rhs_[p_offset + e];
    }

    const solver::SparseSystem S = hybrid_interface_sparse(*mesh_, dim_, hops, free);
    const std::vector<double> b =
        exokal::hodge::hybrid_interface_load(*mesh_, dim_, hops, fu, fp, pinned, &free);

    HybridReport out;
    out.multipliers = S.n;
    std::vector<double> lambda;
    out.solve = linear.solve(S, b, lambda);
    if (!out.solve.converged) return out;

    // the multiplier on every facet: solved where free, the datum where pinned
    std::vector<double> all = pinned;
    {
      std::size_t at = 0;
      for (std::size_t f = 0; f < free.size(); ++f) {
        if (free[f] == 0) continue;
        for (std::size_t a = 0; a < hops.facet_dofs(); ++a) all[f * hops.facet_dofs() + a] = lambda[at++];
      }
    }
    const exokal::hodge::HybridStressState st =
        exokal::hodge::hybrid_recovery(*mesh_, dim_, hops, all, fu, fp);
    out.jump = st.jump;
    // back into the model's own state, so every accessor and every write of a
    // .vtu reads the hybrid answer exactly as it reads the monolithic one
    // Sigma comes back as itself: the recovered stress agrees with the
    // monolithic one to 1.7e-11 on both axes, and the two cofacet recoveries
    // of every shared facet agree to 1e-13, which is `jump`. It reads
    // straight into the slots the space numbers -- the local saddle holds the
    // facet stress in that order already, unlike the kinematic fields below.
    for (std::size_t i = 0; i < st.sigma.size() && s_offset_ + i < state_.size(); ++i) {
      state_[s_offset_ + i] = st.sigma[i];
    }
    // No sign here. The local saddle couples the way the mixed assembly does
    // -- sigma row -Dv^T, field row +Dv -- so the two routes solve the same
    // system and every field comes back as itself.
    // DE-INTERLEAVED, the way the load was built. The local saddle carries the
    // kinematic unknowns as ONE vector of width nk per cell -- the divergence
    // rows then the asymmetry rows -- while the model keeps them as SEPARATE
    // fields, u_0 of width n_u and g_0 of width n_g laid out cell-major. A
    // straight copy is right only where n_g = 0, which is the wrench layout;
    // on the componentwise families it wrote the rotation into the
    // displacement's slots and the answer came back as though the datum had
    // never been applied. The load assembly above interleaves them; this
    // undoes it.
    for (std::size_t e = 0; e < cells; ++e) {
      for (std::size_t r = 0; r < n_u; ++r) {
        const std::size_t at = e * nk + r;
        if (at < st.u.size() && u_offset_ + e * n_u + r < state_.size()) {
          state_[u_offset_ + e * n_u + r] = st.u[at];
        }
      }
      for (std::size_t r = 0; r < n_g; ++r) {
        const std::size_t at = e * nk + n_u + r;
        if (at < st.u.size() && g_offset + e * n_g + r < state_.size()) {
          state_[g_offset + e * n_g + r] = st.u[at];
        }
      }
    }
    for (std::size_t i = 0; i < st.p.size() && p_offset + i < state_.size(); ++i) {
      state_[p_offset + i] = st.p[i];
    }
    return out;
  }
  const std::vector<double>& state() const { return state_; }

  // The cell displacement, in the sign and the scale a caller means by it.
  //
  // Two conversions, and both belong here rather than at every read site. The
  // unknown is the moment of u over the cell, not a nodal value, so the mean is
  // that divided by the measure. And it carries the opposite sign to the
  // physical displacement: the mixed form is written [M, -B^T; +B, 0], so the
  // multiplier standing in the constitutive row is -u. That convention is not a
  // free choice once several physics land in one system -- two of them meeting
  // there would give a matrix neither symmetric nor antisymmetric -- so the
  // place to undo it is the accessor, once. Reading the raw dof gives an answer
  // that is exactly twice the solution away from it, which looks like a
  // discretization error and is not.
  double displacement(Index cell, int axis) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mu = sp.map(sp.index_of("u_0"));
    return -state_[u_offset_ + static_cast<std::size_t>(mu.global(dim_, cell, 0, axis))] /
           exokal::measure(*mesh_, dim_, cell);
  }

  // The cell rotation, the multiplier of the weak symmetry constraint.
  //
  // Same two conversions as the displacement, and for the same reasons: the
  // unknown is the moment over the cell and it stands in the constitutive row
  // with the opposite sign. `p` indexes the generators of skew(d) in (i < j)
  // order -- one in two dimensions, three in three -- so gamma(e, 0) is the
  // rotation of the plane in 2D and the yz, xz, xy components follow in 3D.
  //
  // For a displacement field u, the rotation it is measuring is skew(grad u):
  // an exactly reproduced linear field therefore has an exactly reproduced
  // rotation.
  double rotation(Index cell, int p) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    // Strong symmetry has no multiplier: the rotation lives in the last three
    // of the displacement's six rigid-motion coefficients. Two conversions on
    // top of the measure and multiplier sign:
    //
    //   * the order. exokal's generators run by axis (e_x∧r, e_y∧r, e_z∧r);
    //     the weak multiplier runs by (i < j) plane. Pair p is the rotation
    //     about axis 2-p, with the alternating sign of ε_ijk: W_ij = -ω_axis
    //     for the xy and yz planes, +ω for xz.
    //   * the strain coupling. The coefficients are the L²-RM projection of
    //     u, and on a cell whose centred second moment M₂ is anisotropic that
    //     projection sees the strain: ω_proj = ω + J⁻¹ q(ε), with
    //     J = tr(M₂)I - M₂ and q_a(ε) = ε_{lan} ε_lm (M₂)_nm. The strain is
    //     C⁻¹σ from the cell stress -- exact for affine fields -- so
    //     ω = ω_proj - J⁻¹ q(ε) restores skw(grad u) on any cell. On a cube
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

  // The cell-average stress tensor, reconstructed from the facet tractions the
  // space already carries.
  //
  //     |E| sigma_ij = int_{dE} (sigma n_out)_i (x - x_E)_j
  //
  // which is the divergence theorem applied to sigma_ik d_k (x - x_E)_j, and it
  // is exact for a constant stress: the leading facet moment is int_f (sigma n)
  // because the chart has chi_0 = 1, and the remaining factor comes out of the
  // integral. So a mixed method whose stress is piecewise constant reproduces it
  // to round-off, and one whose stress varies is sampled at the facet centroids
  // -- second order, not first, because the moment is the facet mean.
  //
  // The expansion is about the cell centroid. The alternative, using x itself,
  // differs by x_{E,j} times the net force on the cell -- zero only when the
  // cell is in equilibrium with no body load. Under a reservoir pressurization
  // it is not, and the difference is the whole depletion signal.
  //
  // The result is symmetrized. Symmetry of the stress is imposed weakly in this
  // formulation -- that is what the rotation multiplier gamma is for -- so the
  // raw reconstruction carries an antisymmetric part of the size of the
  // discretization error.
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
      // int_f (sigma n) against the canonical normal: the leading moment per
      // component for the weak family, and the three mean slots read through
      // the facet frame for the strong one. The incidence turns it outward.
      const std::array<double, 3> force =
          wrench_layout() ? strong_facet_force(f, fr)
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

  // The traction on any facet, interior or boundary, in ambient components and
  // against the facet's canonical normal.
  //
  // In Hellinger-Reissner the facet traction moments are primary unknowns, so
  // this is the value on the plane itself rather than a cell-centred stress
  // sampled half a cell away -- which matters most exactly where it is read from
  // a fault, since no amount of refinement along the plane fixes an error
  // across it.
  //
  // It takes no coface, and that is the difference from `normal_traction`: the
  // orientation asked for is the canonical one, which a facet owns by itself,
  // so an interior facet is a legitimate argument. Going through a coface to
  // find a frame is a boundary accessor and fails on a fault.
  std::array<double, 3> facet_traction(Index facet) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const double area = exokal::measure(*mesh_, dim_ - 1, facet);
    if (wrench_layout()) {
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

  // The normal traction on a facet, read through the same form a condition
  // would impose there: n . (sigma n), against the facet's canonical normal and
  // divided by the measure it was integrated against.
  double normal_traction(Index facet) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cofacet_of(*mesh_, dim_, facet), facet);
    if (wrench_layout()) {
      // n . int_f (sigma n) is the normal-resultant slot alone -- 3 in space,
      // 1 in the plane: chi_0 = 1 and the tangential slots are orthogonal to
      // the normal
      return state_[s_offset_ + static_cast<std::size_t>(
                                    ms.global(dim_ - 1, facet, dim_ == 3 ? 3 : 1, 0))] /
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
    if (wrench_layout()) {
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
  // canonical frame -- the one FacetFrame::of shares with the discrete dofs --
  // and the chart comes from the facet's own centred second moments, computed
  // by the same quadrature that evaluates the moments. Degree 4 is exact for
  // every integrand here, so the affine datum is reproduced rather than
  // approximated.
  std::array<double, 6> strong_datum_coefficients(Index f, Index cell,
                                                  const std::array<double, 3>& a,
                                                  const std::array<double, 9>& B) const {
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cell, f);
    const std::array<exokal::Point, 2> tg = wrench_tangents(fr);
    const exokal::Point xE = exokal::centroid(*mesh_, dim_, cell);
    const exokal::Point xf = exokal::centroid(*mesh_, dim_ - 1, f);
    const exokal::QuadratureRule qr = exokal::facet_quadrature(*mesh_, dim_, f, 4);

    const auto xi = [&](const exokal::Point& x, int t) {
      double s = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        s += tg[static_cast<std::size_t>(t)][k] * (x[k] - xf[k]);
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
        ut1 += tg[0][k] * u[k];
        ut2 += tg[1][k] * u[k];
        un += fr.normal[k] * u[k];
        urot += nxdx[k] * u[k];
      }
      const double x1 = xi(x, 0), x2 = xi(x, 1);
      const double chi1 = chart.a11 * x1;
      if (dim_ == 2) {
        // the plane's three: {t, n chi_0, n chi_1}
        v[0] += w * ut1;
        v[1] += w * un;
        v[2] += w * un * chi1;
        continue;
      }
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

  // int_f (sigma n) in ambient components against the canonical normal, from
  // the six strong-symmetry slots. The rotation and first-moment slots
  // integrate to zero against a constant -- ∫chi_a = 0 for a >= 1 and the
  // rotation basis is centred -- so the mean is carried by slots 0, 1 and 3
  // in the facet's canonical frame, which FacetFrame::of shares with the
  // discrete basis.
  //
  // In the plane the wrench is {t, n chi_0, n chi_1}: the tangential and
  // normal resultants sit in slots 0 and 1, and the first moment in slot 2
  // integrates to zero against a constant.
  std::array<double, 3> strong_facet_force(Index facet, const FacetFrame& fr) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const auto slot = [&](int b) {
      return state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, b, 0))];
    };
    const std::array<exokal::Point, 2> t = wrench_tangents(fr);
    std::array<double, 3> force{};
    if (dim_ == 3) {
      const double s0 = slot(0), s1 = slot(1), s3 = slot(3);
      for (std::size_t k = 0; k < 3; ++k) {
        force[k] = s0 * t[0][k] + s1 * t[1][k] + s3 * fr.normal[k];
      }
      return force;
    }
    const double s0 = slot(0), s1 = slot(1);
    for (std::size_t k = 0; k < 3; ++k) force[k] = s0 * t[0][k] + s1 * fr.normal[k];
    return force;
  }

  // The tangents the discrete wrench basis uses. In space the frame's own
  // two, built by the rule the basis shares; in the plane the basis takes the
  // quarter turn of the canonical normal, (-n_y, n_x), which is what an
  // edge's tangent is in exokal's chart -- and not the edge's own direction,
  // whose sign the frame does not promise.
  std::array<exokal::Point, 2> wrench_tangents(const FacetFrame& fr) const {
    if (dim_ == 3) return fr.tangent;
    return {exokal::Point{-fr.normal[1], fr.normal[0], 0.0}, exokal::Point{0.0, 0.0, 0.0}};
  }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  ElasticMaterial material_;
  Realization how_;
  Formulation form_{Formulation::weak_symmetry};
  std::vector<double> lame_per_cell_;  // per cell; empty is the uniform material

  exokal::hodge::Coefficient lame_coefficient() const {
    return lame_per_cell_.empty() ? exokal::hodge::Coefficient::uniform(material_.lame)
                                  : exokal::hodge::Coefficient::per_cell(lame_per_cell_);
  }

  double degeneracy_percent_{-1.0};  // adaptive_vem's scan threshold; negative is unset
  double cond_threshold_{-1.0};      // adaptive_vem's conditioning threshold; negative is unset
  double rotation_jump_{0.0};        // diagonal_afw's rotation-jump constant; 0 is off
  std::size_t n_ill_conditioned_{0};
  MechanicsBoundary mechanics_;

  exokal::hodge::DeRhamGeometryCache geometry_;
  std::vector<double> body_force_;  // per cell, d components, cell-major
  CellVectorData body_force_data_;
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
