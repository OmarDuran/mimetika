#pragma once

#include <array>
#include <cmath>
#include <functional>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/geometry/degeneracy.hpp"
#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/hybrid_flux.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/flow.hpp"
#include "mimetika/model/conditioning.hpp"
#include "mimetika/model/hybrid_interface.hpp"
#include "mimetika/model/partition.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/boundary_terms.hpp"
#include "mimetika/linear_solver/linear.hpp"

// Single-phase flow, stated as data -- the poroelastic problem with one physics
// instead of two.
//
//     div q = 0,   q = -(K/mu) grad p
//
// It is the configuration a flow-only benchmark needs, and the smallest problem
// that exercises the flux space, the natural pressure datum and the strong flux
// condition without any mechanics in the way. When a poroelastic answer is
// wrong, this is what says whether the flow half is.
//
// Both of its closed forms hold in any dimension:
//
//     a column   p linear between the two ends          the Laplace solution
//     an annulus p = p_a + (p_b - p_a) ln(r/a)/ln(b/a)  Dupuit

namespace mimetika {

class FlowModel {
 public:
  // Which discrete Hodge, chosen here rather than fixed. exokal offers two de
  // Rham realizations of the flux star and one stabilized polytopal product,
  // and they are different discretizations of the same equations:
  //
  //   derham      d moments per facet. On a simplex this is BDM_1 -- 3 edges
  //               x 2 in the plane, 4 facets x 3 in space -- unisolvent with
  //               no enrichment. On a polytope the moments are completed with
  //               div-free curl modes. The space a coupled poroelastic model
  //               needs, because the stress space it pairs with carries d^2.
  //   derham_rt   one flux per facet: RT_0, the minimal de Rham pair, whose
  //               radial mode x - x_E is what lets div reach P_0 at all.
  //               Simplices in space only today; it refuses anything else
  //               rather than stabilizing it.
  //   stabilized  one flux per facet on any polytope, consistency plus a
  //               stabilization -- not a de Rham construction.
  //   diagonal_tpfa  one flux per facet and no reconstruction: the diagonal
  //               primal-dual star, exact only where the mesh is K-orthogonal.
  //   adaptive_rt one flux per facet: the per-cell selection between the two
  //               above, carried as eta in {0, 1}. Ones everywhere -- the
  //               stabilized product -- and 0 on the cells the metric-
  //               degeneracy scan flags, which take the diagonal star rather
  //               than a reconstruction over a collapsed cell. The threshold
  //               is exokal's default_degeneracy_percent unless
  //               set_degeneracy_percent() names one.
  //
  // The realization decides the space as well as the star, and the two must
  // agree on moments per facet or every index after the first facet is wrong,
  // so both are derived from this one field.
  using Realization = exokal::hodge::FluxOperators::Realization;

  FlowModel(const exokal::Mesh& mesh, int cell_dim, double mobility = 1.0,
                   Realization how = Realization::derham_bdm)
      : mesh_(&mesh), dim_(cell_dim), mobility_(mobility), how_(how) {}

  Realization realization() const { return how_; }
  const char* realization_name() const { return exokal::hodge::FluxOperators::name(how_); }
  double mobility() const { return mobility_; }

  // K per cell, in place of the uniform mobility.
  //
  // The star is the metric of the flux (d-1)-cochains, so K enters it and
  // nothing else: d is incidence and does not see the coefficient. A jump in K
  // is therefore a jump in one block of the Hodge star, which is what makes a
  // contrast test a test of the star rather than of the complex.
  // `components` is 1 for an isotropic K, 3 for a diagonal one and 6 for a
  // fully symmetric tensor (xx, yy, zz, xy, xz, yz) -- the layout a mesh file
  // supplies and exokal's Coefficient::from_components reads. A TENSOR is not
  // a convenience over three scalars: the anisotropy is what the auxiliary
  // spaces of the H(div) preconditioner have to see, since it is the direction
  // K conducts in that decides which gradients are near-null.
  void set_permeability(std::vector<double> k, int components = 1) {
    const auto cells = static_cast<std::size_t>(mesh_->topology().count(dim_));
    if (components != 1 && components != 3 && components != 6) {
      throw std::invalid_argument(
          "FlowModel::set_permeability: 1, 3 or 6 components (isotropic, diagonal, symmetric)");
    }
    if (!k.empty() && k.size() != cells * static_cast<std::size_t>(components)) {
      throw std::invalid_argument("FlowModel::set_permeability: " + std::to_string(components) +
                                 " value(s) per cell");
    }
    for (std::size_t e = 0; e < k.size() / static_cast<std::size_t>(components); ++e) {
      const double* v = k.data() + e * static_cast<std::size_t>(components);
      // POSITIVE DEFINITE, not merely positive: the star is K's inverse, so a
      // tensor that is not definite is not a permeability. Sylvester on the
      // leading minors says so exactly for a 3x3 symmetric.
      if (components == 6) {
        const double xx = v[0], yy = v[1], zz = v[2], xy = v[3], xz = v[4], yz = v[5];
        const double m2 = xx * yy - xy * xy;
        const double m3 = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) +
                          xz * (xy * yz - yy * xz);
        if (!(xx > 0.0) || !(m2 > 0.0) || !(m3 > 0.0)) {
          throw std::invalid_argument("FlowModel::set_permeability: K is not positive definite on "
                                      "cell " + std::to_string(e));
        }
      } else {
        for (int c = 0; c < components; ++c) {
          if (!(v[c] > 0.0)) {
            throw std::invalid_argument("FlowModel::set_permeability: K must be positive");
          }
        }
      }
    }
    permeability_ = std::move(k);
    permeability_components_ = components;
  }
  const std::vector<double>& permeability() const { return permeability_; }
  int permeability_components() const { return permeability_components_; }

  // THE SCALAR THE RIESZ NORM NEEDS, and it has two jobs.
  //
  // W stands for the Schur complement B star_K^-1 B^T of the constraint, and
  // the star's facet entry carries n.K n -- so the scalar has to say what a
  // cell contributes to that scale. Two things reduce it, and they are not the
  // same thing:
  //
  //   DIRECTION   the cell's own tensor conducts differently along each facet
  //               normal. Those facets are PARALLEL paths for the divergence,
  //               which sums their fluxes, so they combine ARITHMETICALLY: a
  //               cell with K = diag(1, 1, 1e-6) is not a 1e-6 cell.
  //   NEIGHBOUR   a facet whose other side is more resistive conducts less
  //               than this cell alone would. That is a SERIES path, so it
  //               combines HARMONICALLY, and a cell inside a hard inclusion is
  //               limited by the inclusion's boundary however large its own K.
  //
  // Hence the product: the area-weighted arithmetic mean of the cell's own
  // normal permeabilities, times the harmonic mean of the per-facet ratios
  // k_f / k_own, which is 1 wherever K does not jump. Uniform K gives back K.
  //
  // MEASURED, on the enclosure of Kolev & Vassilevski Fig. 6.1 with K_in =
  // 1e-8 .. 1e+8 (512 cells, stabilized_rt, ADS-CG): this rule holds 14-16
  // iterations across all sixteen orders. The cell's own K alone gives 14 for
  // a soft inclusion and 72 for a hard one; ignoring K gives 49 for a soft
  // inclusion and 14 for a hard one. Each of those is one of the two jobs
  // done and the other dropped.
  // THE SCALAR OVERRIDE. Which scalar represents a K in the divergence term is
  // a modelling choice, so a caller who knows their medium can name it -- and
  // the rule above can be measured against alternatives rather than argued.
  void set_norm_permeability(std::vector<double> k) {
    if (!k.empty() && k.size() != static_cast<std::size_t>(mesh_->topology().count(dim_))) {
      throw std::invalid_argument("FlowModel::set_norm_permeability: one value per cell");
    }
    for (const double v : k) {
      if (!(v > 0.0)) {
        throw std::invalid_argument("FlowModel::set_norm_permeability: must be positive");
      }
    }
    norm_permeability_ = std::move(k);
  }

  std::vector<double> norm_permeability() const {
    if (!norm_permeability_.empty()) return norm_permeability_;
    if (permeability_.empty()) return {};
    const graphos::Complex& c = mesh_->topology();
    const auto cells = static_cast<std::size_t>(c.count(dim_));
    const graphos::BoundaryOperator& bnd = c.boundary(dim_);
    const graphos::CoboundaryOperator cob = graphos::coboundary(c, dim_ - 1);
    const auto tensor_of = [&](std::size_t e) {
      const double* v =
          permeability_.data() + e * static_cast<std::size_t>(permeability_components_);
      if (permeability_components_ == 1) return exokal::hodge::SymTensor<>::isotropic(v[0]);
      if (permeability_components_ == 3) {
        return exokal::hodge::SymTensor<>::diagonal(v[0], v[1], v[2]);
      }
      return exokal::hodge::SymTensor<>{v[0], v[1], v[2], v[3], v[4], v[5]};
    };
    const auto normal_k = [&](const exokal::hodge::SymTensor<>& K,
                              const std::array<double, 3>& n) {
      const std::array<double, 3> Kn = K.apply(n);
      return n[0] * Kn[0] + n[1] * Kn[1] + n[2] * Kn[2];
    };
    std::vector<double> out(cells, 1.0);
    for (std::size_t e = 0; e < cells; ++e) {
      const exokal::hodge::SymTensor<> K = tensor_of(e);
      double own_num = 0.0, area_sum = 0.0, ratio_den = 0.0;
      for (Index k = bnd.offsets[e]; k < bnd.offsets[e + 1]; ++k) {
        const Index f = bnd.indices[static_cast<std::size_t>(k)];
        const exokal::Point a =
            exokal::facet_normal_vector(*mesh_, dim_, static_cast<Index>(e), f);
        const double area = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (!(area > 0.0)) continue;
        const std::array<double, 3> n{a[0] / area, a[1] / area, a[2] / area};
        const double k_own = normal_k(K, n);
        if (!(k_own > 0.0)) continue;
        // the other side of this facet, where there is one
        double k_facet = k_own;
        for (Index m = cob.offsets[static_cast<std::size_t>(f)];
             m < cob.offsets[static_cast<std::size_t>(f) + 1]; ++m) {
          const Index other = cob.indices[static_cast<std::size_t>(m)];
          if (static_cast<std::size_t>(other) == e) continue;
          const double k_other = normal_k(tensor_of(static_cast<std::size_t>(other)), n);
          if (k_other > 0.0) k_facet = 2.0 * k_own * k_other / (k_own + k_other);
          break;
        }
        own_num += area * k_own;
        area_sum += area;
        ratio_den += area * (k_own / k_facet);  // 1 / ratio, area-weighted
      }
      if (area_sum > 0.0 && ratio_den > 0.0) {
        out[e] = (own_num / area_sum) * (area_sum / ratio_den);
      }
    }
    return out;
  }
  int moments_per_facet() const {
    return exokal::hodge::FluxOperators::moments_per_facet(how_, dim_);
  }

  FlowBoundary& flow() { return flow_; }
  const FlowBoundary& flow() const { return flow_; }

  // The adaptive_rt threshold: scan for metrically degenerate cells at this
  // percentage of the node-star mean and give them the diagonal star, eta = 0.
  // eta is derived -- ones with the flagged cells zeroed -- so what is set here
  // is the one number the scan needs. Unset, exokal scans at its own
  // default_degeneracy_percent.
  void set_degeneracy_percent(double percent) { degeneracy_percent_ = percent; }

  // The second selector, by conditioning: a cell whose stabilized block has
  // lambda_max / lambda_min above this takes the diagonal star as well. The
  // scan sees a collapsed cell; this sees a sliver of ordinary volume, which
  // the scan does not. It composes with the scan -- either flag zeroes eta --
  // and costs one probe build of the stabilized member before the real one.
  void set_cond_threshold(double cond) { cond_threshold_ = cond; }

  // how many cells the conditioning selector switched, as built
  std::size_t n_ill_conditioned() const { return n_ill_conditioned_; }

  // The selection as built, one value per cell: 1 is the stabilized product,
  // 0 the diagonal star -- the field to write next to the solution, since it
  // is the one the operator was assembled from.
  const std::vector<double>& eta() const {
    if (sim_ == nullptr) {
      throw std::logic_error("FlowModel: not built yet; call build() or solve() first");
    }
    return flux_.eta();
  }

  // The validity gate of the diagonal star, which exokal records and leaves to
  // the consumer: a facet the cell centroid does not see squarely carries a
  // non-positive two-point weight, M is not positive there, and whatever is
  // built on it -- the Riesz map, a condensation -- is meaningless while still
  // reporting CONVERGED. Populated by diagonal_tpfa, and by adaptive_rt on the
  // cells its scan handed to the star.
  std::size_t n_not_star_shaped() const {
    if (sim_ == nullptr) {
      throw std::logic_error("FlowModel: not built yet; call build() or solve() first");
    }
    return flux_.n_not_star_shaped();
  }
  const std::vector<Index>& not_star_shaped() const {
    if (sim_ == nullptr) {
      throw std::logic_error("FlowModel: not built yet; call build() or solve() first");
    }
    return flux_.not_star_shaped();
  }

  int dim() const { return dim_; }
  const exokal::Mesh& mesh() const { return *mesh_; }
  bool built() const { return sim_ != nullptr; }

  // Dereferencing an unbuilt model is a null read, and from Python that is an
  // abort rather than an exception -- so it is refused by name instead.
  // Non-const, for the one caller that configures the simulation rather than
  // reading it: the partition, which tells it which sites to assemble.
  Simulation& simulation() {
    if (!sim_) {
      throw std::logic_error("FlowModel: not built yet; call build() or solve() first");
    }
    return *sim_;
  }

  const Simulation& simulation() const {
    if (sim_ == nullptr) {
      throw std::logic_error("FlowModel: not built yet; call build() or solve() first");
    }
    return *sim_;
  }
  const solver::SparseSystem& system() const { return system_; }
  std::size_t n_cells() const { return n_cells_; }

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
    // the K-independent mode selection, and only the P_1 realization needs it
    if (how_ == Realization::derham_bdm) {
      geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    }
    // The selection, derived rather than given: ones, with 0 on the cells the
    // scan flags at the caller's threshold. exokal forces its own default-
    // threshold zeros on top either way, so a caller can only widen the set.
    if ((degeneracy_percent_ >= 0.0 || cond_threshold_ >= 0.0) &&
        how_ != Realization::adaptive_rt) {
      throw std::invalid_argument(
          "FlowModel: the degeneracy and conditioning thresholds are adaptive_rt's "
          "cell selection");
    }
    std::vector<double> eta;
    if (how_ == Realization::adaptive_rt && degeneracy_percent_ >= 0.0) {
      eta.assign(static_cast<std::size_t>(c.count(dim_)), 1.0);
      for (const Index e : exokal::degenerate_cell_ids(*mesh_, dim_, degeneracy_percent_)) {
        eta[static_cast<std::size_t>(e)] = 0.0;
      }
    }
    n_ill_conditioned_ = 0;
    if (how_ == Realization::adaptive_rt && cond_threshold_ >= 0.0) {
      // the probe: the stabilized member on every cell (exokal still zeroes
      // the collapsed ones, and those are not judged again), read cell by cell
      if (eta.empty()) eta.assign(static_cast<std::size_t>(c.count(dim_)), 1.0);
      const std::vector<double> ones(eta.size(), 1.0);
      const exokal::hodge::FluxOperators probe = exokal::hodge::FluxOperators::build(
          *mesh_, dim_, coefficient(), how_, nullptr,
          exokal::hodge::default_enrichment_degree, exokal::hodge::default_max_facets, only,
          nullptr, &ones);
      for (std::size_t e = 0; e < eta.size(); ++e) {
        if (!probe.eta().empty() && probe.eta()[e] == 0.0) eta[e] = 0.0;
      }
      n_ill_conditioned_ = cond_selection(
          eta, cond_threshold_,
          [&](std::size_t e) -> const exokal::numerics::Dense& {
            return probe.cell(static_cast<Index>(e));
          });
    }
    flux_ = exokal::hodge::FluxOperators::build(
        *mesh_, dim_, coefficient(), how_,
        how_ == Realization::derham_bdm ? &geometry_ : nullptr,
        exokal::hodge::default_enrichment_degree,
        exokal::hodge::default_max_facets, only, nullptr,
        eta.empty() ? nullptr : &eta);
    pressure_data_ = BoundaryData(static_cast<std::size_t>(c.count(dim_ - 1)));
    const bool any_pressure = flow_.fill_pressure(pressure_data_, *mesh_, dim_);
    // and the same datum as facet-basis coefficients: one per flux moment, so
    // an affine pressure reaches every moment the facet carries rather than
    // the constant one alone
    pressure_moments_ = BoundaryMoments(static_cast<std::size_t>(c.count(dim_ - 1)),
                                        static_cast<std::size_t>(moments_per_facet()));
    flow_.fill_pressure_moments(pressure_moments_, *mesh_, dim_);

    ctx_.provide("flux_operators", flux_);
    ctx_.provide("boundary_pressure", pressure_data_);
    ctx_.provide("boundary_pressure_moments", pressure_moments_);

    physics::ModelOptions o;
    o.flux_moments = moments_per_facet();
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("flow", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    if (any_pressure) sim_->model().add("prescribed_pressure", exokal::forms::On::all(), {});

    const auto& sp = sim_->epoch().stratum(0).space();
    flow_.resolve(*mesh_, dim_, sp);
    flow_.impose(sim_->constraints());
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

    exokal::forms::TripletSink jac(sim_->n_dofs());
    sim_->jacobian(jac);
    system_ = solver::SparseSystem::from(jac);

    // The steady right-hand side. Everything the terms contribute that does not
    // depend on the unknowns is a load, and the residual at the zero state is
    // exactly minus that load -- so it is read off rather than re-derived. The
    // strongly imposed rows carry their own datum, scaled as their equation is.
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
    p_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
    q_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("q_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  const std::vector<double>& rhs() const { return rhs_; }

  void accept(std::vector<double> x) { state_ = std::move(x); }

  // ---- the hybridized route ------------------------------------------------
  //
  // A second elimination, and a different system. The mixed form condenses its
  // flux only when the star is diagonal; hybridization takes any product.
  // Each cell keeps its own facet flux, normal continuity moves to a
  // multiplier on the facets -- the facet pressure, in the flux's own moment
  // chart -- and what a solver sees is the interface system alone: symmetric
  // positive definite once a facet is pinned. So a conjugate gradient and an
  // algebraic multigrid apply on every realization, the stabilized and the
  // adaptive ones included, where the condensed mixed system exists for the
  // diagonal star alone and the Riesz map pays for an H(div) block.
  //
  // The boundary roles swap. The multiplier is the facet pressure, so a
  // pressure datum pins a multiplier to its value, and a normal-flux datum is
  // natural: it loads the free row of its facet. A boundary facet given no
  // condition is therefore sealed here, where the mixed form reads it as
  // p = 0 -- the homogeneous natural condition of each form is the other's
  // essential one. The Robin condition couples a facet flux to a cell
  // pressure and is not expressed in this form; it is refused.
  //
  // The recovery is exokal's and cell-local, and it reports `jump`: the worst
  // disagreement between the two cofacet recoveries of a shared facet, which
  // is the continuity the multiplier enforces -- a solve that left the
  // interface unconverged shows there.
  struct HybridReport {
    solver::SolveReport solve;
    std::size_t multipliers{0};
    double jump{0.0};
  };

  HybridReport hybridized(solver::LinearSolver& linear) {
    if (sim_ == nullptr) {
      throw std::logic_error("FlowModel::hybridized: not built yet; call build() first");
    }
    const exokal::hodge::HybridFluxOperators hops =
        exokal::hodge::HybridFluxOperators::build(*mesh_, dim_, flux_);
    const std::size_t nf = hops.facet_dofs();
    const graphos::Complex& topo = mesh_->topology();
    const auto n_facets = static_cast<std::size_t>(topo.count(dim_ - 1));

    // The free mask is the boundary condition. Every facet is free unless a
    // pressure is prescribed on it; the datum is the multiplier's value. The
    // chart has chi_0 = 1 and zero-mean higher functions, so a uniform datum
    // is its constant coefficient alone.
    std::vector<char> free(n_facets, 1);
    std::vector<double> pinned(n_facets * nf, 0.0);
    std::vector<double> flux_datum(n_facets, 0.0);  // canonical q.n times |f|
    std::vector<char> has_flux(n_facets, 0);
    for (std::size_t i = 0; i < flow_.size(); ++i) {
      const BoundaryCondition& bc = flow_.at(i);
      if (const auto* p = dynamic_cast<const PressureBC*>(&bc)) {
        for (const Index f : p->facets()) {
          free[static_cast<std::size_t>(f)] = 0;
          pinned[static_cast<std::size_t>(f) * nf] = p->value();
        }
      } else if (const auto* q = dynamic_cast<const NormalFluxBC*>(&bc)) {
        for (const Index f : q->facets()) {
          flux_datum[static_cast<std::size_t>(f)] =
              q->value() * exokal::measure(*mesh_, dim_ - 1, f);
          has_flux[static_cast<std::size_t>(f)] = 1;
        }
      } else {
        throw std::invalid_argument("FlowModel::hybridized: the '" + bc.name() +
                                    "' condition is not expressed in the hybridized form");
      }
    }

    // the cell loads, read off the assembled residual's balance rows
    const auto cells = static_cast<std::size_t>(n_cells_);
    std::vector<double> fp(cells, 0.0);
    for (std::size_t e = 0; e < cells; ++e) fp[e] = rhs_[p_offset_ + e];

    const solver::SparseSystem S = hybrid_interface_sparse(*mesh_, dim_, hops, free);
    std::vector<double> b =
        exokal::hodge::hybrid_interface_load(*mesh_, dim_, hops, fp, pinned, &free);

    // The normal-flux datum, natural here. The multiplier row of a facet
    // reads sum_E s_E w_E(f) = 0 -- continuity on an interior facet, a sealed
    // boundary facet where there is one cofacet -- and a prescribed canonical
    // flux g|f| on that facet moves the row's right-hand side off zero. The
    // local coupling carries +s into the flux row, so the load is -s g|f| on
    // the constant slot; the inflow column of the ctest pins that sign.
    {
      std::vector<std::ptrdiff_t> at(n_facets, -1);
      std::ptrdiff_t next = 0;
      for (std::size_t f = 0; f < n_facets; ++f) {
        if (free[f] != 0) {
          at[f] = next;
          next += static_cast<std::ptrdiff_t>(nf);
        }
      }
      const graphos::CoboundaryOperator cob = graphos::coboundary(topo, dim_ - 1);
      for (std::size_t f = 0; f < n_facets; ++f) {
        if (has_flux[f] == 0 || at[f] < 0 || flux_datum[f] == 0.0) continue;
        const auto k0 = static_cast<std::size_t>(cob.offsets[f]);
        const auto k1 = static_cast<std::size_t>(cob.offsets[f + 1]);
        if (k1 - k0 != 1) {
          throw std::invalid_argument(
              "FlowModel::hybridized: a normal-flux datum on an interior facet");
        }
        const auto& c = hops.cell(cob.indices[k0]);
        double sign = 0.0;
        for (std::size_t fi = 0; fi < c.faces.size(); ++fi) {
          if (static_cast<std::size_t>(c.faces[fi]) == f) sign = c.signs[fi];
        }
        b[static_cast<std::size_t>(at[f])] -= sign * flux_datum[f];
      }
    }

    HybridReport out;
    out.multipliers = S.n;
    std::vector<double> lambda;
    out.solve = linear.solve(S, b, lambda);
    if (!out.solve.converged) return out;

    const std::vector<double> all =
        exokal::hodge::hybrid_multiplier(*mesh_, dim_, hops, lambda, pinned, &free);
    const exokal::hodge::HybridFluxState st =
        exokal::hodge::hybrid_recovery(*mesh_, dim_, hops, all, fp);
    out.jump = st.jump;

    // back into the model's own state, so every accessor and every write of a
    // .vtu reads the hybrid answer exactly as it reads the monolithic one:
    // the recovered flux is the canonical facet moment the space numbers, and
    // the recovered p is the pressure itself by exokal's coupling sign
    if (state_.empty()) state_.assign(sim_->n_dofs(), 0.0);
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mq = sp.map(sp.index_of("q_0"));
    for (std::size_t f = 0; f < n_facets; ++f) {
      for (std::size_t bslot = 0; bslot < nf; ++bslot) {
        state_[q_offset_ + static_cast<std::size_t>(mq.global(
                               dim_ - 1, static_cast<Index>(f), static_cast<int>(bslot), 0))] =
            st.w[f * nf + bslot];
      }
    }
    for (std::size_t e = 0; e < cells; ++e) state_[p_offset_ + e] = st.p[e];
    return out;
  }
  double cell_pressure(Index e) const { return state_[p_offset_ + static_cast<std::size_t>(e)]; }

  // The cell's flux vector, from the facet fluxes that are the unknowns.
  //
  // The unknown on a facet is the moment int_f q.n against the canonical
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
  // The facets come from the topology rather than from the product, because
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
  std::vector<double> permeability_;  // per cell, `components` wide; empty is the mobility
  int permeability_components_{1};
  std::vector<double> norm_permeability_;  // the override; empty is the star's own average

  exokal::hodge::Coefficient coefficient() const {
    return permeability_.empty()
               ? exokal::hodge::Coefficient::uniform(mobility_)
               : exokal::hodge::Coefficient::from_components(permeability_,
                                                            permeability_components_,
                                                            mesh_->topology().count(dim_));
  }
  Realization how_;
  FlowBoundary flow_;
  double degeneracy_percent_{-1.0};  // adaptive_rt's scan threshold; negative is unset
  double cond_threshold_{-1.0};      // adaptive_rt's conditioning threshold; negative is unset
  std::size_t n_ill_conditioned_{0};

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::FluxOperators flux_;
  BoundaryData pressure_data_{0};
  BoundaryMoments pressure_moments_;
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
