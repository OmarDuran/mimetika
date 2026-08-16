#pragma once

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/linear_solver/linear.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/elasticity.hpp"
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
  //   stabilized_afw  the same d^2 degrees of freedom reconstructed on the FULL
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

  CauchyElasticityModel(const exokal::Mesh& mesh, int cell_dim, ElasticMaterial material,
                        Realization how = Realization::derham_afw)
      : mesh_(&mesh), dim_(cell_dim), material_(material), how_(how) {
    if (how != Realization::derham_afw && how != Realization::stabilized_afw) {
      throw std::invalid_argument(
          "CauchyElasticityModel: only the derham and stabilized_afw stress products are "
          "elements; derham_rt is unisolvent but its weak-symmetry inf-sup degenerates");
    }
  }

  MechanicsBoundary& mechanics() { return mechanics_; }
  const MechanicsBoundary& mechanics() const { return mechanics_; }

  int dim() const { return dim_; }
  const exokal::Mesh& mesh() const { return *mesh_; }
  Realization realization() const { return how_; }
  const char* realization_name() const { return exokal::hodge::StressOperators::name(how_); }
  const ElasticMaterial& material() const { return material_; }
  const Simulation& simulation() const { return *sim_; }
  const solver::SparseSystem& system() const { return system_; }
  const std::vector<double>& rhs() const { return rhs_; }
  std::size_t n_cells() const { return n_cells_; }
  // how many cells needed a stabilization: zero on a simplex mesh for either
  // realization, and that is the construction rather than a coincidence
  std::size_t n_stabilized() const { return stress_.n_stabilized(); }
  const exokal::hodge::StressOperators& stress_operators() const { return stress_; }

  void build() {
    const graphos::Complex& c = mesh_->topology();
    // the K-independent mode selection, which only the de Rham product has
    const bool derham = how_ == Realization::derham_afw;
    if (derham) geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    stress_ = exokal::hodge::StressOperators::build(*mesh_, dim_, material_.shear, material_.lame,
                                                    how_, derham ? &geometry_ : nullptr);
    ctx_.provide("stress_operators", stress_);

    displacement_data_ = BoundaryVectorData(static_cast<std::size_t>(c.count(dim_ - 1)));
    for (const auto& d : displacement_facets_) {
      displacement_data_.set_affine(d.facets, d.constant, d.gradient);
    }
    ctx_.provide("boundary_displacement", displacement_data_);

    reservoir_data_ = CellData(static_cast<std::size_t>(c.count(dim_)));
    for (const auto& r : reservoir_) reservoir_data_.set(r.cells, r.pressure);
    ctx_.provide("reservoir_pressure", reservoir_data_);

    // THE SPACE FOLLOWS THE STAR: d^2 traction moments per facet for both of
    // these, read off the operators rather than restated, so the layout and the
    // product cannot drift apart.
    physics::ModelOptions o;
    o.traction_moments = stress_.moments_per_facet();
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("linear_elasticity", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);
    // the natural displacement datum, attached only when one is actually given
    if (!displacement_facets_.empty()) {
      sim_->model().add("prescribed_displacement", exokal::forms::On::all(), {});
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
      const int nb = stress_.moments_per_facet();
      for (const Index f : prescribed_) {
        for (int b = 0; b < nb; ++b) {
          for (int k = 0; k < dim_; ++k) {
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
      const std::size_t d = stress_.cell(e).M.rows() + stress_.cell(e).Dv.rows() +
                            stress_.cell(e).As.rows();
      nnz += d * d;
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
      rhs_[i] = sim_->constraints().pinned(i)
                    ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i)
                    : -r[i];
    }

    // ONE FACTORIZATION for every later evaluation of S
    solver_.factorize(system_);
    load_ready_ = true;
    work_.assign(sim_->n_dofs(), 0.0);
    s_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
    u_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
    g_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("g_0")));
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
      rhs_[i] = sim_->constraints().pinned(i)
                    ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i)
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
        static_cast<std::size_t>(dim_) * static_cast<std::size_t>(stress_.moments_per_facet());
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
    const auto& op = stress_.cell(cell);
    const exokal::Point xE = exokal::centroid(*mesh_, dim_, cell);
    const double volume = exokal::measure(*mesh_, dim_, cell);

    std::array<double, 9> raw{};
    for (const Index f : op.faces) {
      const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cell, f);
      const exokal::Point xf = exokal::centroid(*mesh_, dim_ - 1, f);
      for (int i = 0; i < dim_; ++i) {
        // the leading moment is int_f (sigma n) against the CANONICAL normal;
        // the incidence turns it outward from this cell
        const double t = fr.incidence *
                         state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, f, 0, i))];
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
    double t = 0.0;
    for (int k = 0; k < dim_; ++k) {
      t += e[static_cast<std::size_t>(k)] *
           state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))];
    }
    return t / fr.measure;
  }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  ElasticMaterial material_;
  Realization how_;
  MechanicsBoundary mechanics_;

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::StressOperators stress_;
  exokal::forms::TermContext ctx_;
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
};

}  // namespace mimetika
