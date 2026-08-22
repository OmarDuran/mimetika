// mimetika Python bindings (mimetika_cxx._core).
//
// The surface here is the one the fixed-dimensional model tests exercise, and
// it is deliberately the SAME surface those tests use in C++: a mesh, a
// realization, a model assembled on it, conditions imposed as forms, a direct
// solve, and DOF addresses to read the answer back with. Nothing is
// precomputed on the Python side and nothing is summarised for it -- a test
// that measures `sigma_lateral` here does the same arithmetic, on the same
// numbers, as tests/model/test_confined_compression.cpp does.
//
// LIFETIME. Simulation keeps a pointer to its TermContext, and StratumSpec
// keeps a pointer to the Complex inside the Mesh; both must outlive it. A
// binding that handed those out separately would be one temporary away from
// the dangling-reference bug this codebase has already paid for once, so the
// owning object is a single non-copyable Problem that holds the mesh, the
// operators, the context and the simulation together, in that construction
// order.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <limits>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "exokal/io/vtu.hpp"
#include "exokal/preprocess/diagnostics.hpp"
#include "exokal/preprocess/curate_vtu.hpp"
#include "mimetika/model/partition.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/conditioning.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/model/compositions/elasticity.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/model/single_phase_model.hpp"

namespace py = pybind11;

using exokal::hodge::DeRhamGeometryCache;
using exokal::hodge::FluxOperators;
using exokal::hodge::StressOperators;
using graphos::Index;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::physics::Catalogue;

namespace {

using Point = exokal::Mesh::Point;

py::array_t<double> to_array(const std::vector<double>& v) {
  py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

// A dense block as a NumPy 2-D copy, so rank and norms are numpy's problem
// rather than a hand-rolled Gauss-Jordan repeated in two languages.
py::array_t<double> to_array(const exokal::numerics::Dense& A) {
  py::array_t<double> out({static_cast<py::ssize_t>(A.rows()), static_cast<py::ssize_t>(A.cols())});
  auto w = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < A.rows(); ++i) {
    for (std::size_t j = 0; j < A.cols(); ++j) {
      w(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = A(i, j);
    }
  }
  return out;
}

// ---- the owning problem ----------------------------------------------------
//
// One class per star, because the star decides the space and the space decides
// the model: an elasticity problem is d copies of a stress product, a flow
// problem is one flux product, and the two share no member beyond the mesh.

class ElasticityProblem {
 public:
  ElasticityProblem(exokal::Mesh mesh, int cell_dim, double mu, double lam,
                    StressOperators::Realization how, const std::string& model)
      : mesh_(std::move(mesh)), dim_(cell_dim) {
    // the RT star carries no de Rham geometry; the BDM ones do
    const bool bdm = how != StressOperators::Realization::derham_rt;
    if (bdm) geo_ = DeRhamGeometryCache::build(mesh_, cell_dim);
    ops_ = StressOperators::build(mesh_, cell_dim, mu, lam, how,
                                  StressOperators::Formulation::weak_symmetry,
                                  bdm ? &geo_ : nullptr);
    ctx_.provide("stress_operators", ops_);

    // THE SPACE FOLLOWS THE STAR: the layout is read off the operators rather
    // than restated here, so the two cannot drift apart.
    mimetika::physics::ModelOptions mo;
    mo.traction_moments = ops_.moments_per_facet();
    comp_ = Catalogue::instance().build(model, mo);
    sim_ = std::make_unique<Simulation>(
        comp_, std::vector<StratumSpec>{StratumSpec{"ambient", &mesh_.topology(), cell_dim, 0}},
        ctx_);
  }

  ElasticityProblem(const ElasticityProblem&) = delete;
  ElasticityProblem& operator=(const ElasticityProblem&) = delete;

  const exokal::Mesh& mesh() const { return mesh_; }
  int dim() const { return dim_; }
  std::size_t n_dofs() const { return sim_->n_dofs(); }
  std::size_t n_stabilized() const { return ops_.n_stabilized(); }
  int moments_per_facet() const { return ops_.moments_per_facet(); }
  Index count(int k) const { return mesh_.topology().count(k); }

  // the local constraint block [Dv; As] of one cell, stacked as the test wants
  // it: a locally surjective block has full row rank, a deficient one does not
  py::array_t<double> constraint_block(Index e) const {
    const auto& c = ops_.cell(e);
    exokal::numerics::Dense B(c.Dv.rows() + c.As.rows(), c.M.cols());
    for (std::size_t i = 0; i < c.Dv.rows(); ++i) {
      for (std::size_t j = 0; j < B.cols(); ++j) B(i, j) = c.Dv(i, j);
    }
    for (std::size_t i = 0; i < c.As.rows(); ++i) {
      for (std::size_t j = 0; j < B.cols(); ++j) B(c.Dv.rows() + i, j) = c.As(i, j);
    }
    return to_array(B);
  }

  std::vector<Index> boundary_facets() const {
    return mimetika::boundary_facets(mesh_.topology(), dim_);
  }

  void impose_traction(const std::string& field, const std::vector<Index>& facets,
                       const std::function<std::array<double, 9>(const Point&)>& stress) {
    mimetika::impose_traction(sim_->constraints(), space(), field, dim_, mesh_, facets, stress);
  }

  void impose_free_slip(const std::string& field, const std::vector<Index>& facets) {
    mimetika::impose_free_slip(sim_->constraints(), space(), field, dim_, mesh_, facets);
  }

  void freeze_constraints() { sim_->freeze_constraints(); }

  // Assemble and solve, exactly as the C++ test does: the pinned rows carry
  // scale * rhs and everything else is zero.
  py::tuple solve() {
    exokal::forms::TripletSink jac(sim_->n_dofs());
    sim_->jacobian(jac);
    const auto A = mimetika::solver::SparseSystem::from(jac);
    std::vector<double> b(sim_->n_dofs(), 0.0), x;
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      if (sim_->constraints().pinned(i)) {
        b[i] = sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i);
      }
    }
    mimetika::solver::PetscSolver petsc;
    const auto rep = petsc.solve(A, b, x);
    // a singular saddle point has no solution to measure; the caller asserts on
    // `converged` and must not read the vector
    return py::make_tuple(rep.converged, rep.reason,
                          rep.converged ? to_array(x) : to_array(std::vector<double>{}));
  }

  // The DOF address of one component, so Python indexes the solution the same
  // way the C++ does rather than through a convenience that could differ.
  Index dof(const std::string& field, int k, Index entity, int local, int comp) const {
    const auto& sp = space();
    const auto f = sp.index_of(field);
    return static_cast<Index>(sp.offset(f)) + sp.map(f).global(k, entity, local, comp);
  }

  Index cofacet_of(Index facet) const { return mimetika::cofacet_of(mesh_, dim_, facet); }

  // normal / outward / measure / tangents of a facet, as a dict: the frame is
  // what every boundary measurement is expressed in
  py::dict facet_frame(Index facet) const {
    const auto fr = mimetika::FacetFrame::of(mesh_, dim_, cofacet_of(facet), facet);
    py::dict out;
    out["normal"] = fr.normal;
    out["outward"] = fr.outward;
    out["incidence"] = fr.incidence;
    out["measure"] = fr.measure;
    out["n_tangents"] = fr.n_tangents;
    py::list tang;
    for (int a = 0; a < fr.n_tangents; ++a) tang.append(fr.tangent[static_cast<std::size_t>(a)]);
    out["tangent"] = tang;
    return out;
  }

  Point centroid(int k, Index i) const { return exokal::centroid(mesh_, k, i); }
  double measure(int k, Index i) const { return exokal::measure(mesh_, k, i); }

 private:
  const exokal::spaces::ProductSpace& space() const { return sim_->epoch().stratum(0).space(); }

  // construction order is the lifetime contract: the mesh outlives the
  // operators, which outlive the context, which outlives the simulation
  exokal::Mesh mesh_;
  int dim_;
  DeRhamGeometryCache geo_;
  StressOperators ops_;
  exokal::forms::TermContext ctx_;
  mimetika::physics::Composition comp_;
  std::unique_ptr<Simulation> sim_;
};

// ---- a model assembled but not solved ---------------------------------------
//
// What the structural tests need: the composition's space, and the Jacobian as
// triplets so the block algebra is numpy's rather than a dense loop written
// twice. Both stars are provided, because poroelasticity reads both and a
// composition that reads only one simply never asks for the other.
class AssembledModel {
 public:
  AssembledModel(exokal::Mesh mesh, int cell_dim, const std::string& model, double mu, double lam,
                 StressOperators::Realization stress_how, double mobility,
                 FluxOperators::Realization flux_how)
      : mesh_(std::move(mesh)), dim_(cell_dim) {
    geo_ = DeRhamGeometryCache::build(mesh_, cell_dim);
    stress_ = StressOperators::build(mesh_, cell_dim, mu, lam, stress_how,
                                     StressOperators::Formulation::weak_symmetry, &geo_);
    flux_ = FluxOperators::build(mesh_, exokal::hodge::Coefficient::uniform(mobility), flux_how);
    ctx_.provide("stress_operators", stress_);
    ctx_.provide("flux_operators", flux_);
    mimetika::physics::ModelOptions mo;
    mo.traction_moments = stress_.moments_per_facet();
    comp_ = Catalogue::instance().build(model, mo);
    sim_ = std::make_unique<Simulation>(
        comp_, std::vector<StratumSpec>{StratumSpec{"ambient", &mesh_.topology(), cell_dim, 0}},
        ctx_);
  }

  AssembledModel(const AssembledModel&) = delete;
  AssembledModel& operator=(const AssembledModel&) = delete;

  std::size_t n_dofs() const { return sim_->n_dofs(); }
  std::size_t n_packages() const { return comp_.size(); }
  std::size_t stress_size() const { return stress_.size(); }
  std::size_t stress_n_stabilized() const { return stress_.n_stabilized(); }
  std::size_t flux_n_cells() const { return flux_.n_cells(); }

  // the state the Jacobian is taken at, as the C++ test seeds it
  void seed_state() {
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      sim_->state()[i] = 0.2 + 0.01 * static_cast<double>(i % 11);
    }
  }

  void freeze_constraints() { sim_->freeze_constraints(); }

  // (rows, cols, values): the caller densifies or builds a sparse matrix, which
  // is what the C++ does by hand
  py::tuple jacobian_coo() {
    exokal::forms::TripletSink jac(sim_->n_dofs());
    sim_->jacobian(jac);
    const std::size_t nnz = jac.nnz();
    py::array_t<std::int64_t> rows(static_cast<py::ssize_t>(nnz));
    py::array_t<std::int64_t> cols(static_cast<py::ssize_t>(nnz));
    py::array_t<double> vals(static_cast<py::ssize_t>(nnz));
    auto r = rows.mutable_data(), c = cols.mutable_data();
    auto v = vals.mutable_data();
    for (std::size_t k = 0; k < nnz; ++k) {
      r[k] = static_cast<std::int64_t>(jac.row[k]);
      c[k] = static_cast<std::int64_t>(jac.col[k]);
      v[k] = jac.value[k];
    }
    return py::make_tuple(rows, cols, vals);
  }

  // [start, end) of one field in the global vector
  py::tuple field_range(const std::string& name) const {
    const auto& sp = space();
    const auto f = sp.index_of(name);
    const auto b = static_cast<std::size_t>(sp.offset(f));
    return py::make_tuple(b, b + static_cast<std::size_t>(sp.map(f).size()));
  }

  std::size_t n_fields() const { return space().n_fields(); }
  std::size_t space_size() const { return space().size(); }
  bool has(const std::string& name) const { return space().has(name); }
  std::size_t field_size(const std::string& name) const {
    const auto& sp = space();
    return static_cast<std::size_t>(sp.map(sp.index_of(name)).size());
  }
  int field_degree(const std::string& name) const {
    const auto& sp = space();
    return sp.map(sp.index_of(name)).layout().degree();
  }

 private:
  const exokal::spaces::ProductSpace& space() const { return sim_->epoch().stratum(0).space(); }

  exokal::Mesh mesh_;
  int dim_;
  DeRhamGeometryCache geo_;
  StressOperators stress_;
  FluxOperators flux_;
  exokal::forms::TermContext ctx_;
  mimetika::physics::Composition comp_;
  std::unique_ptr<Simulation> sim_;
};

// WHICH STAGE IS RUNNING, on request.
//
// Assembly, preconditioner and iteration are the three long stages, and from
// the outside a process in any of them is indistinguishable from a hung one.
//
// THE WRITE GOES TO stderr, unbuffered. Reporting through Python's stdout
// looks correct and is not: on a redirected stream the bytes sit in a buffer
// that `flush=True` does not push to the operating system, so a stage that
// takes ten minutes prints its label immediately and its duration only when
// the process exits -- which is exactly the case the reporting exists for, and
// exactly when it says nothing. Python's own output is flushed first so the
// two streams stay in order.
// PROGRESS IS ONE PROCESS'S JOB. Every rank runs the same script and reaches
// the same stages at slightly different moments; eight of them writing to an
// unbuffered stderr produces "assembling ... assembling ... assembling ..." on
// one line and their timings on the next eight. The stage lines are a picture
// of the run, so rank 0 draws it and the others stay quiet.
class Stage {
 public:
  explicit Stage(bool on) : on_(on && is_root()) {}

  // the same for a phase timed out here: the model's build
  static double slowest(double seconds) {
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    }
    return seconds;
  }

  static bool is_root() {
    mimetika::solver::PetscSession::instance();
    PetscMPIInt rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank == 0;
  }

  void begin(const char* what) {
    if (!on_) return;
    what_ = what;
    t0_ = std::chrono::steady_clock::now();
    flush_python();
    std::fprintf(stderr, "  %s ...", what);
    std::fflush(stderr);
  }

  void end() {
    if (!on_) return;
    report(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count());
  }

  // A stage whose duration was measured elsewhere -- the solver reports its own
  // three -- so that they print in the same shape as the ones timed here.
  void done(const char* what, double seconds) {
    if (!on_) return;
    flush_python();
    std::fprintf(stderr, "  %s ... %.2f s\n", what, seconds);
    std::fflush(stderr);
  }

 private:
  void report(double seconds) {
    std::fprintf(stderr, " %.2f s\n", seconds);
    std::fflush(stderr);
    (void)what_;
  }

  static void flush_python() {
    py::module_::import("sys").attr("stdout").attr("flush")();
  }

  bool on_;
  const char* what_{""};
  std::chrono::steady_clock::time_point t0_{};
};

// THE NORM OF THE MODEL'S SPACE, read off the model rather than restated.
//
// Factor 0 is the H(div) field -- flux or stress. EVERY LATER FIELD IS ONE L^2
// FACTOR, merged: the norm of X is (A s, s) + ||div s||^2 on the first and
// plain L^2 on the rest, and L^2 of a product is the product of the L^2s, so
// the displacement and the AFW rotation form a single factor. Splitting them
// would be a different preconditioner, not a finer statement of the same one.
//
// A cell field's dofs are laid out entity-major (see DofMap::global), so dof k
// of a field of size m over n cells belongs to cell k / (m / n) and its L^2
// weight is that cell's measure.
//
// The graph term is carried ONLY by the rows of the differential constraint.
// AFW's inf-sup is proved with ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2:
// skw is bounded L^2 -> L^2, so the rotation's rows contribute nothing and
// their weight is zero.
//
// THE NORM HAS NO LENGTH UNIT OF ITS OWN. Scaling the coordinates by L
// multiplies the material block by 1/L and B^T |E|^-1 B by 1/L^3, so the
// plain H(div) norm is really ||sigma||^2 + L_0^2 ||div sigma||^2 with L_0 = 1
// in whatever unit the mesh is written in. On a unit box that is the textbook
// norm; on an 18 km industrial patch the graph term is 1e8 times too weak, the
// pressure eigenvalues of P^-1 A collapse, and GMRES stalls with a true
// residual 700 times the right-hand side while the preconditioned one
// shrinks -- measured on industrial_2_patch_0 with every flux product, and
// gone on the same mesh scaled to a unit box. So W carries a length of the
// domain and the material scale: W = |E| K / L^2 for the flow (M ~ 1/K),
// W = mu / (|E| L^2) for elasticity (A ~ 1/mu). A uniform rescaling of both
// Riesz blocks changes nothing; what this fixes is the RATIO of the graph
// term to the material block, which decides whether the map is an
// isomorphism with constants of order one.
//
// L IS THE BOUNDING DIAGONAL OF THE MESH. The continuous argument: with
// ||sigma||^2 + L^2 ||div sigma||^2 against ||u||^2 / L^2, the inf-sup
// constant is L / sqrt(C^2 + L^2) with C the domain's Poincare scale, so it
// saturates once L reaches the diameter and shrinks linearly below it.
// Measured: the edge of the cube with the domain's measure -- (sum |E|)^(1/d),
// 1 on the unit boxes -- doubled stabilized_vem's count on hybrid_mesh_l_2
// (33 -> 67) against the diagonal; the unit-box ladders that once seemed to
// prefer it were reading a Gram-Schmidt artefact (see build_riesz), and with
// that gone the diagonal is flat and lowest on every ladder: 11, 11, 12 for
// the flow annulus against 69, 66, 59 under the unit-bound norm.
inline double bounding_diagonal(const exokal::Mesh& mesh) {
  std::array<double, 3> lo{}, hi{};
  lo.fill(std::numeric_limits<double>::infinity());
  hi.fill(-std::numeric_limits<double>::infinity());
  for (const auto& q : mesh.points()) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], q[k]);
      hi[k] = std::max(hi[k], q[k]);
    }
  }
  double d2 = 0.0;
  for (int k = 0; k < 3; ++k) d2 += (hi[k] - lo[k]) * (hi[k] - lo[k]);
  return std::sqrt(d2);
}

inline double norm_scale(const mimetika::SinglePhaseModel& m) { return m.mobility(); }
inline double norm_scale(const mimetika::CauchyElasticityModel& m) { return m.material().shear; }

template <class Model>
void attach_norm(mimetika::solver::PetscSolver& petsc, const Model& m, const exokal::Mesh& mesh,
                 int dim, bool divergence_is_an_integral, bool merge_multipliers = true) {
  const auto blocks = mimetika::solver::field_blocks(m.simulation().epoch());
  if (blocks.size() < 2) {
    throw std::runtime_error("riesz: the space has fewer than two factors");
  }
  const auto n_cells = static_cast<std::size_t>(mesh.count(dim));
  std::vector<double> measure(n_cells);
  for (std::size_t e = 0; e < n_cells; ++e) {
    measure[e] = exokal::measure(mesh, dim, static_cast<Index>(e));
  }
  const double L = bounding_diagonal(mesh);
  const double scale = norm_scale(m) / (L * L);

  mimetika::solver::SpaceNorm norm;
  std::vector<int> first;
  first.reserve(blocks[0].size());
  for (const Index g : blocks[0].indices()) first.push_back(static_cast<int>(g));
  norm.factors.push_back(std::move(first));

  std::vector<int> rest;
  std::vector<double> l2;
  for (std::size_t f = 1; f < blocks.size(); ++f) {
    const std::size_t size = blocks[f].size();
    if (size % n_cells != 0) {
      throw std::runtime_error("riesz: factor '" + blocks[f].name +
                               "' is not a cell field; its L2 norm is not the cell measure");
    }
    const std::size_t components = size / n_cells;
    std::size_t k = 0;
    for (const Index g : blocks[f].indices()) {
      rest.push_back(static_cast<int>(g));
      const double me = measure[k / components];
      // W is the SCHUR SCALE of this constraint, not the variational L2 mass:
      // an unscaled row (flow's incidence) gives |E|, a row already divided by
      // the measure (elasticity's Dv, As) gives 1/|E|.
      l2.push_back(scale * (divergence_is_an_integral ? me : 1.0 / me));
      ++k;
    }
  }
  // ONE FACTOR, OR ONE PER FIELD. Merged is what the FULL system wants: the
  // rotation adds no term to the stress norm in theory and a great deal in
  // practice -- 41 iterations against 85 without it, measured -- so the
  // multipliers are preconditioned together.
  //
  // A system that will be CONDENSED wants the opposite. Eliminating the first
  // field leaves the multipliers as the whole system, and for diagonal_vem
  // that is a saddle point of displacement against volumetric stress; the
  // reduced Riesz map needs them apart to have anything to split on.
  if (merge_multipliers) {
    norm.factors.push_back(std::move(rest));
    norm.l2_weight.push_back(std::move(l2));
  } else {
    std::size_t at = 0;
    for (std::size_t f = 1; f < blocks.size(); ++f) {
      const std::size_t size = blocks[f].size();
      norm.factors.emplace_back(rest.begin() + static_cast<std::ptrdiff_t>(at),
                                rest.begin() + static_cast<std::ptrdiff_t>(at + size));
      norm.l2_weight.emplace_back(l2.begin() + static_cast<std::ptrdiff_t>(at),
                                  l2.begin() + static_cast<std::ptrdiff_t>(at + size));
      at += size;
    }
  }

  // THE COMPLEX'S OWN BOUNDARY OPERATORS, for an auxiliary-space solver.
  //
  // d_1 (edges x vertices) is the discrete gradient and d_2 (faces x edges)
  // the discrete curl -- not built here, only copied out of the topology with
  // the incidence numbers already on them. They address the FACETS, so they
  // are supplied whenever the first factor is carried by the facets at all:
  // one unknown per facet is the space ADS is written for, and a space with
  // several moments per facet reaches it through its facet-constant subspace.
  const graphos::Complex& topo = mesh.topology();
  const auto& space = m.simulation().epoch().stratum(0).space();
  const auto& facet_map = space.map(0);
  const auto& layout = facet_map.layout();
  // THE TWO NUMBERS THAT DESCRIBE A FACET-CARRIED SPACE, and they are not the
  // same number even when they are equal. `moments` is how much of the facet
  // P_1 basis is measured -- 1 for RT, d for BDM -- and `copies` is how many
  // H(div) fields sit side by side: one for a flux, d for the rows of a
  // stress. ADS solves ONE copy of ONE moment, so both are what decides which
  // route it is reached by.
  const int moments = layout.on(dim - 1);
  const int copies = layout.components;
  const bool on_facets = layout.carries(dim - 1) &&
                         blocks[0].size() == static_cast<std::size_t>(topo.count(dim - 1)) *
                                                 static_cast<std::size_t>(moments * copies);
  // THE STRONG FAMILY REACHES ADS THROUGH A ROTATED SUBSPACE. Its six moments
  // per facet are one traction vector against a facet-intrinsic frame -- not
  // d copies of a scalar layout -- so neither the curl's rows nor a subset
  // injection describe it. But the three MEAN slots {t1, t2, n} rotated by
  // the facet's own frame ARE three scalar H(div) cochains in global
  // components: int_f (tau n).e_k. The injection is then WEIGHTED by the
  // frame rather than a matrix of ones, and the two-level cycle -- facet-
  // local smoother, per-component ADS on the coarse space -- carries over
  // unchanged, because nothing in it ever assumed the weights were ones.
  const bool strong_stress = copies == 1 && moments == 6;
  if (dim == 3 && on_facets) {
    const auto copy_out = [](const graphos::BoundaryOperator& b, int rows, int cols) {
      mimetika::solver::SpaceNorm::Incidence out;
      out.rows = rows;
      out.cols = cols;
      for (Index r = 0; r < static_cast<Index>(b.rows()); ++r) {
        for (Index k = b.offsets[static_cast<std::size_t>(r)];
             k < b.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
          out.row.push_back(static_cast<int>(r));
          out.col.push_back(static_cast<int>(b.indices[static_cast<std::size_t>(k)]));
          out.value.push_back(static_cast<double>(b.signs[static_cast<std::size_t>(k)]));
        }
      }
      return out;
    };
    norm.discrete_gradient = copy_out(topo.boundary(1), topo.count(1), topo.count(0));
    norm.discrete_curl = copy_out(topo.boundary(2), topo.count(2), topo.count(1));
    // and the vertices themselves: ADS interpolates its auxiliary vector
    // spaces by applying the two maps above to the coordinate functions, so
    // this is where the metric enters a construction that is otherwise the
    // complex alone.
    norm.space_dim = 3;
    norm.vertex_coordinates.reserve(static_cast<std::size_t>(topo.count(0)) * 3);
    for (Index v = 0; v < topo.count(0); ++v) {
      const auto& x = mesh.point(v);
      norm.vertex_coordinates.insert(norm.vertex_coordinates.end(), {x[0], x[1], x[2]});
    }

    // WHO OWNS EACH VERTEX, EDGE AND FACE, on several processes.
    //
    // The maps above are the complex's, in its own serial numbering; hypre
    // needs the three spaces distributed, and consistently -- so the ownership
    // is taken from the SAME partition the unknowns were taken from, through
    // the same exokal call, with a layout that puts one unknown on each entity
    // of the dimension in question. A k-entity's owner is then just the owner
    // of that one unknown.
    PetscMPIInt size = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    norm.entity_owner = mimetika::entity_owners(mesh, dim, static_cast<int>(size));

    // THE FACET-CONSTANT SUBSPACE, for every space that is not already it.
    //
    // ADS takes one scalar H(div) problem: one copy, one moment per facet.
    // Anything else -- BDM's d moments of a flux, AFW's d copies of those --
    // reaches it through the subspace spanned by the CONSTANT moment of each
    // copy, and that subspace is not interpolated here. The moments are taken
    // against the facet P_1 basis {1, in-facet coordinates} with the constant
    // first, so the lowest-order space is a SUBSET of the unknowns and the
    // injection is a matrix of ones.
    //
    // A facet's unknowns are ordered (moment, copy) with the COPY fastest --
    // StressOperators permutes its own (copy, moment) ordering into this one
    // when it builds, see its `perm` -- so the constants are the FIRST
    // `copies` unknowns of the facet's block, contiguous rather than strided.
    // Getting that backwards is not a crash: the coarse space is then one
    // copy's worth of moments, carries no divergence, and the cycle merely
    // converges badly (204 iterations against 1070, measured).
    //
    // Columns are copy-major, so each copy of the coarse space is a contiguous
    // run and can be handed to ADS as the scalar problem it expects.
    if (strong_stress) {
      // the frame-weighted injection: per facet, rows are the three mean
      // slots (0: t1, 1: t2, 3: n chi_0) and the columns the three global
      // components, entries the frame's own direction cosines -- the same
      // frame the discrete basis is stated in, so the coarse unknown k at
      // facet f is exactly int_f (tau n) . e_k
      const auto base = static_cast<Index>(m.simulation().epoch().offset(0)) +
                        static_cast<Index>(space.offset(0));
      const Index n_facet = topo.count(dim - 1);
      auto& inj = norm.lowest_order;
      inj.rows = static_cast<int>(m.simulation().n_dofs());
      inj.cols = static_cast<int>(n_facet) * 3;
      const graphos::CoboundaryOperator cob = graphos::coboundary(topo, dim - 1);
      for (Index f = 0; f < n_facet; ++f) {
        const Index cell =
            cob.indices[static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)])];
        const mimetika::FacetFrame fr = mimetika::FacetFrame::of(mesh, dim, cell, f);
        const std::array<const exokal::Point*, 3> dirs{&fr.tangent[0], &fr.tangent[1],
                                                       &fr.normal};
        constexpr std::array<int, 3> slots{0, 1, 3};
        for (int s = 0; s < 3; ++s) {
          const auto row =
              static_cast<int>(base + facet_map.global(dim - 1, f, slots[static_cast<std::size_t>(
                                                                      s)], 0));
          for (int k = 0; k < 3; ++k) {
            const double v = (*dirs[static_cast<std::size_t>(s)])[static_cast<std::size_t>(k)];
            if (v == 0.0) continue;
            inj.row.push_back(row);
            inj.col.push_back(static_cast<int>(k * n_facet + f));
            inj.value.push_back(v);
          }
        }
      }
      // measured, not assumed: adding the rotation slot as a fourth coarse
      // component leaves the count at 54 and only enlarges the coarse solve
      // (27 s -> 35 s at 300k dofs), so the mean slots alone carry the cycle
      norm.lowest_order_components = 3;
    } else if (moments * copies > 1) {
      const auto base = static_cast<Index>(m.simulation().epoch().offset(0)) +
                        static_cast<Index>(space.offset(0));
      const Index n_facet = topo.count(dim - 1);
      auto& inj = norm.lowest_order;
      inj.rows = static_cast<int>(m.simulation().n_dofs());
      inj.cols = static_cast<int>(n_facet) * copies;
      const auto n_entry = static_cast<std::size_t>(n_facet) * static_cast<std::size_t>(copies);
      inj.row.reserve(n_entry);
      inj.col.reserve(n_entry);
      inj.value.assign(n_entry, 1.0);
      for (int c = 0; c < copies; ++c) {
        for (Index f = 0; f < n_facet; ++f) {
          inj.row.push_back(static_cast<int>(base + facet_map.global(dim - 1, f, 0, 0) + c));
          inj.col.push_back(static_cast<int>(c * n_facet + f));
        }
      }
      norm.lowest_order_components = copies;
    }
  }

  // the constrained unknowns, with the diagonal A gave them
  const auto& c = m.simulation().constraints();
  for (std::size_t i = 0; i < m.simulation().n_dofs(); ++i) {
    if (!c.pinned(i)) continue;
    norm.pinned.push_back(static_cast<int>(i));
    norm.pinned_diagonal.push_back(c.scale_at(i));
  }
  petsc.set_norm(std::move(norm));
}

// ASSEMBLE ONLY: the Jacobian and the preconditioner, and no iteration.
//
// The two are what a mesh has to survive before a solve is even attempted, and
// they are the part that scales with the mesh rather than with the physics. A
// caller measuring them wants them without waiting for a Krylov method to
// converge -- and wants to know that they COMPLETED, which a timing alone does
// not say.
// THE PARTITION, IN TWO HALVES, because it is needed on both sides of the
// build: the model must know it before it assembles, and the solver only
// after, when the unknowns it lays out exist.
//
// Neither half decides anything -- mimetika's own `distribute_model` reads
// exokal's geometric partition and its ownership rule -- and both are skipped
// on one process, which is then exactly the program it was before.
template <class Model>
void request_partition(Model& m, const mimetika::solver::SolverOptions& opts) {
  PetscMPIInt size = 1, rank = 0;
  mimetika::solver::PetscSession::instance();
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (size < 2 || !opts.partition) return;
  // NO REDUCTION IS NEEDED, and that is worth saying rather than leaving as an
  // absence: a process assembles every cell that contributes to a row it owns,
  // so the diagonal a constrained row is scaled by is already complete where
  // it is used. Summing across the processes would count the halo twice.
  m.distribute_over(static_cast<int>(size), static_cast<int>(rank), {});
}

template <class Model>
void attach_partition(mimetika::solver::PetscSolver& petsc, const Model& m) {
  if (m.distribution().empty()) return;
  petsc.set_local_assembly(true);
  petsc.set_owners(m.distribution().owner_of_dof);
}

template <class Model>
mimetika::solver::SolveReport assemble_only(Model& m, bool progress,
                                            const mimetika::solver::SolverOptions& opts,
                                            bool divergence_is_an_integral) {
  Stage stage(progress);
  request_partition(m, opts);
  stage.begin("assembling");
  const auto t_build = std::chrono::steady_clock::now();
  m.build();
  const double assembly_seconds = Stage::slowest(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count());
  stage.end();
  mimetika::solver::PetscSolver petsc(opts);
  attach_partition(petsc, m);
  if (opts.preconditioner == "riesz") {
    attach_norm(petsc, m, m.mesh(), m.dim(), divergence_is_an_integral);
  }
  // THE SAME GATE THE SOLVE USES: a facet-diagonal block is eliminated and
  // the reduced system's preconditioner is what gets built -- measuring the
  // saddle's instead reports a cost, in time and in memory, no solve pays.
  petsc.set_condensable(mimetika::solver::first_field_dofs(m.simulation().epoch()));
  stage.begin("preconditioner");
  mimetika::solver::SolveReport r = petsc.prepare(m.system(), m.rhs());
  stage.end();
  stage.done("  matrix", r.matrix_seconds);
  stage.done(opts.direct() && !r.condensed ? "  factorization" : "  preconditioner",
             r.preconditioner_seconds);
  r.assembly_seconds = assembly_seconds;
  return r;
}

// THE HYBRIDIZED ROUTE, FROM PYTHON.
//
// A second elimination rather than a second solver: the stress leaves cell by
// cell, traction continuity becomes a multiplier on the facets, and what a
// Krylov method sees is the interface system alone -- SPD once a facet is
// pinned, so a conjugate gradient applies where the condensed mixed system was
// quasi-definite.
//
// THE BOUNDARY ROLES SWAP. The multiplier IS the facet displacement, so
// prescribe_displacement PINS one and a traction loads the free rows. The
// model does that mapping; nothing here restates it.
//
// SERIAL. The assembly walks every cell and scatters into a global multiplier
// numbering; distributed, each rank holds only its own cells and would emit
// some interface rows twice and others never. That needs the row ownership the
// condensation has and does not have here, so a distributed call is refused
// rather than answered wrongly.
mimetika::solver::SolveReport solve_elasticity_hybrid(mimetika::CauchyElasticityModel& m,
                                                      bool progress,
                                                      const mimetika::solver::SolverOptions& opts) {
  PetscMPIInt size = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (size > 1) {
    throw std::runtime_error(
        "solve_hybrid: the interface assembly is serial -- its multiplier rows are not "
        "partitioned, so a distributed run would double-count some and drop others. Run on one "
        "process, or use the mixed route.");
  }
  Stage stage(progress);
  stage.begin("assembling");
  const auto t_build = std::chrono::steady_clock::now();
  m.build();
  const double assembly_seconds = Stage::slowest(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count());
  stage.end();

  // the interface system is SPD: a conjugate gradient and a multigrid, and an
  // unnamed method becomes exactly that rather than a factorization
  mimetika::solver::SolverOptions o = opts;
  if (o.direct()) {
    o.method = "cg";
    o.preconditioner = "hypre";
  }
  o.condense = false;  // the elimination already happened, cell by cell
  mimetika::solver::PetscSolver petsc(o);

  stage.begin("hybridizing and solving");
  const auto report = m.hybridized(petsc);
  stage.end();
  if (!report.solve.converged) {
    throw std::runtime_error("cauchy elasticity (hybrid): " + report.solve.reason);
  }
  mimetika::solver::SolveReport out = report.solve;
  out.assembly_seconds = assembly_seconds;
  // WHAT THE SOLVER ACTUALLY SAW. The model's dof count is the monolithic
  // space, and after hybridization the linear solver never sees it: what it
  // gets is the facet multipliers alone. Reporting the space instead of the
  // system is how a run gets read as a solve of something it never touched --
  // so the size travels back on the same fields the condensation uses.
  out.condensed = true;
  out.condensed_dofs = report.multipliers;
  return out;
}

mimetika::solver::SolveReport solve_elasticity(mimetika::CauchyElasticityModel& m, bool progress,
                                               const mimetika::solver::SolverOptions& opts) {
  Stage stage(progress);
  request_partition(m, opts);
  stage.begin("assembling");
  const auto t_build = std::chrono::steady_clock::now();
  m.build();
  const double assembly_seconds = Stage::slowest(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count());
  stage.end();
  mimetika::solver::PetscSolver petsc(opts);
  attach_partition(petsc, m);
  // the momentum row is Dv, whose entries already carry 1/|E|: it is an average
  // THE SAME QUESTION THE SOLVER ASKS, asked here because the norm is built
  // before the solve: a system whose first block is diagonal will be condensed,
  // and then the split it needs is the unmerged one.
  const std::vector<int> eliminable = mimetika::solver::first_field_dofs(m.simulation().epoch());
  const bool will_condense =
      opts.condense && mimetika::solver::block_is_diagonal(m.system(), eliminable);
  if (opts.preconditioner == "riesz") {
    attach_norm(petsc, m, m.mesh(), m.dim(), false, !will_condense);
  }
  std::vector<double> x;
  // THE SOLVER TIMES ITS OWN THREE, because they are not one stage: the matrix
  // is linear in the assembly, the preconditioner is what decides whether a
  // mesh is reachable, and the iteration is what the preconditioner shortens.
  stage.begin("solving");
  petsc.set_condensable(mimetika::solver::first_field_dofs(m.simulation().epoch()));
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  stage.end();
  stage.done("  matrix", rep.matrix_seconds);
  stage.done(opts.direct() ? "  factorization" : "  preconditioner", rep.preconditioner_seconds);
  stage.done("  iteration", rep.solve_seconds);
  const_cast<mimetika::solver::SolveReport&>(rep).assembly_seconds = assembly_seconds;
  if (!rep.converged) throw std::runtime_error("cauchy elasticity: " + rep.reason);
  m.accept(std::move(x));
  return rep;
}

// THE HYBRIDIZED FLOW SOLVE: the flux twin of solve_elasticity_hybrid. The
// boundary roles swap -- a pressure datum PINS a multiplier, a normal flux
// loads a free row -- and the interface system is SPD, so an unnamed method
// becomes a conjugate gradient under an algebraic multigrid. Serial, for the
// same reason as the stress: the multiplier rows are not partitioned.
mimetika::solver::SolveReport solve_single_phase_hybrid(mimetika::SinglePhaseModel& m,
                                                        bool progress,
                                                        const mimetika::solver::SolverOptions& opts) {
  PetscMPIInt size = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (size > 1) {
    throw std::runtime_error(
        "solve_hybrid: the interface assembly is serial -- its multiplier rows are not "
        "partitioned, so a distributed run would double-count some and drop others. Run on one "
        "process, or use the mixed route.");
  }
  Stage stage(progress);
  stage.begin("assembling");
  const auto t_build = std::chrono::steady_clock::now();
  m.build();
  const double assembly_seconds = Stage::slowest(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count());
  stage.end();
  mimetika::solver::SolverOptions o = opts;
  if (o.direct()) {
    o.method = "cg";
    o.preconditioner = "hypre";
  }
  o.condense = false;  // the elimination already happened, cell by cell
  mimetika::solver::PetscSolver petsc(o);
  stage.begin("hybridizing and solving");
  const auto report = m.hybridized(petsc);
  stage.end();
  if (!report.solve.converged) {
    throw std::runtime_error("single phase (hybrid): " + report.solve.reason);
  }
  mimetika::solver::SolveReport out = report.solve;
  out.assembly_seconds = assembly_seconds;
  out.condensed = true;
  out.condensed_dofs = report.multipliers;
  return out;
}

mimetika::solver::SolveReport solve_single_phase(mimetika::SinglePhaseModel& m, bool progress,
                                                 const mimetika::solver::SolverOptions& opts) {
  Stage stage(progress);
  request_partition(m, opts);
  stage.begin("assembling");
  const auto t_build = std::chrono::steady_clock::now();
  m.build();
  const double assembly_seconds = Stage::slowest(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count());
  stage.end();
  mimetika::solver::PetscSolver petsc(opts);
  attach_partition(petsc, m);
  // the mass-balance row is the incidence: (Bq)_E is the integral of div q
  // THE SAME QUESTION THE SOLVER ASKS, asked here because the norm is built
  // before the solve: a system whose first block is diagonal will be condensed,
  // and then the split it needs is the unmerged one.
  const std::vector<int> eliminable = mimetika::solver::first_field_dofs(m.simulation().epoch());
  const bool will_condense =
      opts.condense && mimetika::solver::block_is_diagonal(m.system(), eliminable);
  if (opts.preconditioner == "riesz") {
    attach_norm(petsc, m, m.mesh(), m.dim(), true, !will_condense);
  }
  std::vector<double> x;
  // THE SOLVER TIMES ITS OWN THREE, because they are not one stage: the matrix
  // is linear in the assembly, the preconditioner is what decides whether a
  // mesh is reachable, and the iteration is what the preconditioner shortens.
  stage.begin("solving");
  petsc.set_condensable(mimetika::solver::first_field_dofs(m.simulation().epoch()));
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  stage.end();
  stage.done("  matrix", rep.matrix_seconds);
  stage.done(opts.direct() ? "  factorization" : "  preconditioner", rep.preconditioner_seconds);
  stage.done("  iteration", rep.solve_seconds);
  const_cast<mimetika::solver::SolveReport&>(rep).assembly_seconds = assembly_seconds;
  if (!rep.converged) throw std::runtime_error("single phase: " + rep.reason);
  m.accept(std::move(x));
  return rep;
}

// A CellData field from a NumPy array: (n,) is a scalar, (n,3) a vector,
// (n,9) or (n,3,3) a tensor -- the three tuple sizes VTK reads back as such.
exokal::CellField cell_field(const std::string& name, const py::array& a) {
  const auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
  if (!v) throw std::invalid_argument("write_vtu: field '" + name + "' is not real-valued");
  int components = 1;
  if (v.ndim() == 2) {
    components = static_cast<int>(v.shape(1));
  } else if (v.ndim() == 3) {
    components = static_cast<int>(v.shape(1) * v.shape(2));
  } else if (v.ndim() != 1) {
    throw std::invalid_argument("write_vtu: field '" + name + "' must have 1, 2 or 3 axes");
  }
  const double* p = v.data();
  return {name, std::vector<double>(p, p + v.size()), components};
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python interface to the mimetika C++ application";

  // The factors of a model's space, as the solver sees them: name, size and
  // where each run begins. What a block preconditioner is built from, and the
  // first thing to look at when one misbehaves.
  m.def(
      "field_blocks",
      [](const mimetika::SinglePhaseModel& s) {
        py::list out;
        for (const auto& b : mimetika::solver::field_blocks(s.simulation().epoch())) {
          py::list runs;
          for (const auto& r : b.ranges) runs.append(py::make_tuple(r.begin, r.end));
          out.append(py::dict(py::arg("name") = b.name, py::arg("size") = b.size(),
                              py::arg("ranges") = runs));
        }
        return out;
      },
      py::arg("model"));

  m.def(
      "field_blocks",
      [](const mimetika::CauchyElasticityModel& s) {
        py::list out;
        for (const auto& b : mimetika::solver::field_blocks(s.simulation().epoch())) {
          py::list runs;
          for (const auto& r : b.ranges) runs.append(py::make_tuple(r.begin, r.end));
          out.append(py::dict(py::arg("name") = b.name, py::arg("size") = b.size(),
                              py::arg("ranges") = runs));
        }
        return out;
      },
      py::arg("model"));

  // ---- how the linear system is solved -------------------------------------
  py::class_<mimetika::solver::SolverOptions>(m, "SolverOptions")
      .def(py::init([](std::string method, std::string factorization, std::string preconditioner,
                       bool condense,
                       std::string riesz_block_pc, std::string riesz_block_factorization, int riesz_block_its, double riesz_block_rtol,
                       int riesz_exact_limit, int riesz_ads_limit, int riesz_coarse_its,
                       double riesz_coarse_rtol, bool partition, double rtol,
                       double atol, int max_iterations) {
             return mimetika::solver::SolverOptions{std::move(method),
                                                    std::move(factorization),
                                                    std::move(preconditioner),
                                                    condense,
                                                    std::move(riesz_block_pc),
                                                    std::move(riesz_block_factorization),
                                                    riesz_block_its,
                                                    riesz_block_rtol,
                                                    riesz_exact_limit,
                                                    riesz_ads_limit,
                                                    riesz_coarse_its,
                                                    riesz_coarse_rtol,
                                                    partition,
                                                    rtol,
                                                    atol,
                                                    max_iterations};
           }),
           py::arg("method") = "direct", py::arg("factorization") = "superlu",
           py::arg("preconditioner") = "lu", py::arg("condense") = true,
           py::arg("riesz_block_pc") = "", py::arg("riesz_block_factorization") = "mumps",
           py::arg("riesz_block_its") = -1, py::arg("riesz_block_rtol") = 1e-4,
           py::arg("riesz_exact_limit") = 400000, py::arg("riesz_ads_limit") = 400000,
           py::arg("riesz_coarse_its") = 0, py::arg("riesz_coarse_rtol") = 1e-2,
           py::arg("partition") = true, py::arg("rtol") = 1e-10,
           py::arg("atol") = 1e-50, py::arg("max_iterations") = 1000)
      .def_readwrite("condense", &mimetika::solver::SolverOptions::condense)
      .def_readwrite("riesz_block_pc", &mimetika::solver::SolverOptions::riesz_block_pc)
      .def_readwrite("riesz_block_factorization",
                     &mimetika::solver::SolverOptions::riesz_block_factorization)
      .def_readwrite("riesz_block_its", &mimetika::solver::SolverOptions::riesz_block_its)
      .def_readwrite("riesz_block_rtol", &mimetika::solver::SolverOptions::riesz_block_rtol)
      .def_readwrite("riesz_exact_limit", &mimetika::solver::SolverOptions::riesz_exact_limit)
      .def_readwrite("riesz_ads_limit", &mimetika::solver::SolverOptions::riesz_ads_limit)
      .def_readwrite("riesz_coarse_its", &mimetika::solver::SolverOptions::riesz_coarse_its)
      .def_readwrite("riesz_coarse_rtol", &mimetika::solver::SolverOptions::riesz_coarse_rtol)
      .def_readwrite("partition", &mimetika::solver::SolverOptions::partition)
      .def_readwrite("method", &mimetika::solver::SolverOptions::method)
      .def_readwrite("factorization", &mimetika::solver::SolverOptions::factorization)
      .def_readwrite("preconditioner", &mimetika::solver::SolverOptions::preconditioner)
      .def_readwrite("rtol", &mimetika::solver::SolverOptions::rtol)
      .def_readwrite("atol", &mimetika::solver::SolverOptions::atol)
      .def_readwrite("max_iterations", &mimetika::solver::SolverOptions::max_iterations)
      .def_property_readonly("direct", &mimetika::solver::SolverOptions::direct)
      .def("__repr__", [](const mimetika::solver::SolverOptions& o) {
        return o.direct() ? "SolverOptions(direct, " + o.factorization + ")"
                          : "SolverOptions(" + o.method + " + " + o.preconditioner + ")";
      });

  // The assembled operator as triplets. For diagnosis on small problems: a
  // preconditioner is a claim about the spectrum of P^{-1}A, and that claim is
  // checkable directly rather than inferred from an iteration count.
  m.def(
      "system_triplets",
      [](const mimetika::CauchyElasticityModel& s) {
        const auto& A = s.system();
        py::array_t<int> r(static_cast<py::ssize_t>(A.nnz())), c(static_cast<py::ssize_t>(A.nnz()));
        py::array_t<double> v(static_cast<py::ssize_t>(A.nnz()));
        for (std::size_t k = 0; k < A.nnz(); ++k) {
          r.mutable_data()[k] = static_cast<int>(A.row[k]);
          c.mutable_data()[k] = static_cast<int>(A.col[k]);
          v.mutable_data()[k] = A.value[k];
        }
        return py::make_tuple(r, c, v, A.n);
      },
      py::arg("model"));

  py::class_<mimetika::solver::SolveReport>(m, "SolveReport")
      .def_readonly("converged", &mimetika::solver::SolveReport::converged)
      .def_readonly("iterations", &mimetika::solver::SolveReport::iterations)
      .def_readonly("residual", &mimetika::solver::SolveReport::residual)
      .def_readonly("reason", &mimetika::solver::SolveReport::reason)
      .def_readonly("assembly_seconds", &mimetika::solver::SolveReport::assembly_seconds)
      .def_readonly("matrix_seconds", &mimetika::solver::SolveReport::matrix_seconds)
      .def_readonly("preconditioner_seconds",
                    &mimetika::solver::SolveReport::preconditioner_seconds)
      .def_readonly("solve_seconds", &mimetika::solver::SolveReport::solve_seconds)
      .def_readonly("off_rank_fraction", &mimetika::solver::SolveReport::off_rank_fraction)
      .def_readonly("block_solver", &mimetika::solver::SolveReport::block_solver)
      .def_readonly("condensed", &mimetika::solver::SolveReport::condensed)
      .def_readonly("condensed_dofs", &mimetika::solver::SolveReport::condensed_dofs)
      .def("__repr__", [](const mimetika::solver::SolveReport& r) {
        return "SolveReport(converged=" + std::string(r.converged ? "True" : "False") +
               ", iterations=" + std::to_string(r.iterations) + ", reason='" + r.reason + "')";
      });

  // ---- mesh ----------------------------------------------------------------
  py::class_<exokal::Mesh>(m, "Mesh")
      .def_static("from_simplices", &exokal::Mesh::from_simplices, py::arg("dim"),
                  py::arg("points"), py::arg("cells"))
      .def_static("from_polygons", &exokal::Mesh::from_polygons, py::arg("points"),
                  py::arg("cells"))
      // the nested form; the flat-CSR overload is the reader's path, not Python's
      .def_static(
          "from_polyhedra",
          [](std::vector<exokal::Mesh::Point> points,
             const std::vector<std::vector<std::vector<Index>>>& cells) {
            return exokal::Mesh::from_polyhedra(std::move(points), cells);
          },
          py::arg("points"), py::arg("cells"))
      .def_property_readonly("dim", &exokal::Mesh::dim)
      .def("count", &exokal::Mesh::count, py::arg("k"))
      .def(
          "point", [](const exokal::Mesh& x, Index v) { return x.point(v); }, py::arg("vertex"))
      .def("__repr__",
           [](const exokal::Mesh& x) { return "Mesh(dim=" + std::to_string(x.dim()) + ")"; });

  m.def(
      "centroid", [](const exokal::Mesh& x, int k, Index i) { return exokal::centroid(x, k, i); },
      py::arg("mesh"), py::arg("k"), py::arg("cell"));
  m.def(
      "measure", [](const exokal::Mesh& x, int k, Index i) { return exokal::measure(x, k, i); },
      py::arg("mesh"), py::arg("k"), py::arg("cell"));

  m.def("read_vtu", &exokal::read_vtu, py::arg("path"), py::arg("tag_array") = "tag");

  // THE SAME COMPLEX AT ANOTHER SCALE: points mapped to (x - origin) * factor,
  // topology and orientation untouched. What a scale-dependent construction
  // -- a norm with a hidden length unit -- is tested against.
  m.def(
      "scaled",
      [](const exokal::Mesh& x, double factor, std::array<double, 3> origin) {
        std::vector<exokal::Mesh::Point> pts = x.points();
        for (auto& q : pts) {
          for (int k = 0; k < 3; ++k) q[k] = (q[k] - origin[k]) * factor;
        }
        return exokal::Mesh(x.topology(), std::move(pts));
      },
      py::arg("mesh"), py::arg("factor"), py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
      "the mesh with points (x - origin) * factor");

  // ---- what the mesh is, before anything is solved on it -------------------
  //
  // One call into exokal's consumer primitive diagnose_vtu: the formal
  // findings annotated to the file's own cell ids, the shape-regularity
  // classification at the library constants, and the metric-degeneracy
  // witnesses. One read of the file, one judge, nothing recomputed here.
  //
  // percent = 100 |s| / mean over the node star, so a uniform mesh reports 100
  // at every cell whatever its valence; neighborhood_total is carried for a
  // volume-weighted consumer and is not the denominator. vtk_cell_id is the id
  // of the ORIGINAL file -- the cell ParaView selects.
  m.def(
      "diagnose_vtu",
      [](const std::string& path, double degeneracy_percent) {
        exokal::VtuDiagnostics d = exokal::diagnose_vtu(path, degeneracy_percent);
        const exokal::Mesh& mesh = d.file.mesh;
        const int n = mesh.dim();

        std::ostringstream report;
        report << "fundamental-principle diagnostics: " << path << "\n";
        report << "complex: v " << mesh.count(0) << " e " << mesh.count(1) << " f "
               << mesh.count(2) << " c " << mesh.count(3) << "\n\n";
        report << "== ambient complex, realization, and induced full subcomplexes ==\n";
        d.report.print(report);

        std::ostringstream quality;
        exokal::write_mesh_quality_report(quality, d.quality);

        const std::vector<long long> ids = exokal::file_cell_ids(d.file, n);
        py::list degenerate;
        for (const auto& r : d.degenerate) {
          const auto i = static_cast<std::size_t>(r.cell);
          degenerate.append(
              py::dict(py::arg("vtk_cell_id") = i < ids.size() ? ids[i] : -1,
                       py::arg("cell") = static_cast<int>(r.cell), py::arg("dim") = r.dim,
                       py::arg("measure") = r.measure,
                       py::arg("neighborhood_mean") = r.neighborhood_mean,
                       py::arg("neighborhood_total") = r.neighborhood_total,
                       py::arg("n_neighbors") = static_cast<int>(r.n_neighbors),
                       py::arg("percent") = r.percent));
        }
        return py::dict(py::arg("report") = report.str(), py::arg("quality") = quality.str(),
                        py::arg("classification") = exokal::mesh_class_name(d.quality.cls),
                        py::arg("gamma_all") = d.quality.gamma_all,
                        py::arg("gamma_minus_eps") = d.quality.gamma_minus_eps,
                        py::arg("gamma_median") = d.quality.gamma_median,
                        py::arg("cells_checked") = static_cast<long long>(d.quality.cells_checked),
                        py::arg("degenerate") = degenerate);
      },
      py::arg("path"), py::arg("degeneracy_percent") = exokal::default_degeneracy_percent);
  m.attr("default_degeneracy_percent") = py::float_(exokal::default_degeneracy_percent);

  // THE CURATION PASS: remove the cells the scan named, by their vtk ids, by
  // a bistellar flip with a neighbour or by contracting a 0-cell, and write
  // the mesh that is left. The degenerate cell is what breaks the discrete
  // inf-sup constant under the stabilized product; removing it is the robust
  // route where the selection only relocates the problem.
  m.def(
      "curate_vtu",
      [](const std::string& path, const std::vector<long long>& vtk_cells, double tol,
         const std::string& out_path, const std::string& method) {
        exokal::CurationMethod how;
        if (method == "bistellar") {
          how = exokal::CurationMethod::bistellar;
        } else if (method == "contraction") {
          how = exokal::CurationMethod::contraction;
        } else {
          throw std::invalid_argument("curate_vtu: method is 'bistellar' or 'contraction'");
        }
        const exokal::CurationOutcome o = exokal::curate_vtu(path, vtk_cells, tol, out_path, how);
        return py::dict(py::arg("applied") = o.applied, py::arg("notes") = o.notes,
                        py::arg("written") = o.written);
      },
      py::arg("path"), py::arg("vtk_cells"), py::arg("tol") = 1e-8, py::arg("out_path") = "",
      py::arg("method") = "bistellar",
      "remove the named cells (vtk ids) and write the curated mesh; out_path empty "
      "overwrites the origin");

  // THE COLLAPSE PERCENT OF EVERY CELL, not only the witnesses below a
  // threshold: 100 |E| over the mean measure of its node star, the number the
  // degeneracy scan compares against. Asked at an infinite threshold the scan
  // reports every cell with a neighbour; a cell with none gets NaN, which is
  // the honest value for "not judged".
  m.def(
      "cell_collapse_percent",
      [](const exokal::Mesh& mesh, int dim) {
        const auto n = static_cast<std::size_t>(mesh.count(dim));
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        std::fill(out.mutable_data(), out.mutable_data() + n,
                  std::numeric_limits<double>::quiet_NaN());
        for (const exokal::DegenerateCell& r :
             exokal::degenerate_cells(mesh, dim, std::numeric_limits<double>::infinity())) {
          out.mutable_data()[static_cast<std::size_t>(r.cell)] = r.percent;
        }
        return out;
      },
      py::arg("mesh"), py::arg("dim"),
      "100 |cell| / mean measure over its node star, for every cell");

  // THE FILE'S OWN CELL IDS, per cell of the top stratum: the reader reorders
  // cells top-first, so the complex id and the id ParaView selects agree only
  // on a single-stratum file. Every per-cell diagnostic is reported in both.
  m.def(
      "vtu_cell_ids",
      [](const std::string& path) {
        const exokal::VtuFile file = exokal::read_vtu_file(path);
        return exokal::file_cell_ids(file, file.mesh.dim());
      },
      py::arg("path"), "the vtk cell id of each top cell, in complex order");

  // THE SPECTRUM OF EVERY CELL'S INNER PRODUCT, and nothing assembled: the
  // operators are built as the model would build them, each cell's M is
  // handed to a Jacobi eigensolver, and what comes back is one row per cell
  // -- its size, its extreme eigenvalues, and their ratio, the conditioning
  // of that cell's flux or stress block. The selection (eta) and the star's
  // validity ride along where the realization has them, so a bad cell can
  // be read in every light at once.
  const auto spectra_of = [](auto product_of, std::size_t n_cells) {
    py::array_t<double> lmin(static_cast<py::ssize_t>(n_cells)), lmax(static_cast<py::ssize_t>(n_cells)),
        cond(static_cast<py::ssize_t>(n_cells));
    py::array_t<int> ndof(static_cast<py::ssize_t>(n_cells));
    for (std::size_t e = 0; e < n_cells; ++e) {
      const exokal::numerics::Dense M = product_of(static_cast<Index>(e));
      const std::size_t n = M.rows();
      ndof.mutable_data()[e] = static_cast<int>(n);
      if (n == 0) {
        lmin.mutable_data()[e] = lmax.mutable_data()[e] = cond.mutable_data()[e] = 0.0;
        continue;
      }
      const exokal::numerics::SymmetricEigen eig = exokal::numerics::symmetric_eigen(M);
      const double lo = eig.values.front(), hi = eig.values.back();
      lmin.mutable_data()[e] = lo;
      lmax.mutable_data()[e] = hi;
      cond.mutable_data()[e] = lo > 0.0 ? hi / lo : std::numeric_limits<double>::infinity();
    }
    return py::dict(py::arg("n_dofs") = ndof, py::arg("lambda_min") = lmin,
                    py::arg("lambda_max") = lmax, py::arg("cond") = cond);
  };

  m.def(
      "flux_cell_spectra",
      [spectra_of](const exokal::Mesh& mesh, int dim, FluxOperators::Realization how,
                   double mobility, py::object degeneracy_percent, py::object cond_threshold) {
        const auto n = static_cast<std::size_t>(mesh.count(dim));
        const bool adaptive = how == FluxOperators::Realization::adaptive_rt;
        std::vector<double> eta;
        if (adaptive && !degeneracy_percent.is_none()) {
          eta.assign(n, 1.0);
          for (const Index c :
               exokal::degenerate_cell_ids(mesh, dim, py::cast<double>(degeneracy_percent))) {
            eta[static_cast<std::size_t>(c)] = 0.0;
          }
        }
        const auto build = [&](const std::vector<double>* sel) {
          return FluxOperators::build(mesh, dim, exokal::hodge::Coefficient::uniform(mobility),
                                      how, nullptr, exokal::hodge::default_enrichment_degree,
                                      exokal::hodge::default_max_facets, nullptr, nullptr, sel);
        };
        // the conditioning selector reads the stabilized member's blocks from
        // a probe build, exactly as the model does, then builds the selection
        std::size_t ill = 0;
        if (adaptive && !cond_threshold.is_none()) {
          if (eta.empty()) eta.assign(n, 1.0);
          const std::vector<double> ones(n, 1.0);
          const FluxOperators probe = build(&ones);
          for (std::size_t e = 0; e < n; ++e) {
            if (!probe.eta().empty() && probe.eta()[e] == 0.0) eta[e] = 0.0;
          }
          ill = mimetika::cond_selection(
              eta, py::cast<double>(cond_threshold),
              [&](std::size_t e) -> const exokal::numerics::Dense& {
                return probe.cell(static_cast<Index>(e));
              });
        }
        const FluxOperators ops = build(eta.empty() ? nullptr : &eta);
        py::dict out = spectra_of([&](Index e) { return ops.cell(e); }, n);
        out["n_ill_conditioned"] = ill;
        py::array_t<double> eta_out(static_cast<py::ssize_t>(n));
        for (std::size_t e = 0; e < n; ++e) {
          eta_out.mutable_data()[e] = ops.eta().empty() ? 1.0 : ops.eta()[e];
        }
        py::array_t<int> star(static_cast<py::ssize_t>(n));
        std::fill(star.mutable_data(), star.mutable_data() + n, 0);
        for (const Index c : ops.not_star_shaped()) star.mutable_data()[c] = 1;
        out["eta"] = eta_out;
        out["star_invalid"] = star;
        return out;
      },
      py::arg("mesh"), py::arg("dim"), py::arg("realization"), py::arg("mobility") = 1.0,
      py::arg("degeneracy_percent") = py::none(), py::arg("cond_threshold") = py::none(),
      "per-cell spectrum of the flux inner product, nothing assembled; for adaptive_rt "
      "eta is zeroed on the cells the scan flags and on those whose stabilized block's "
      "conditioning exceeds cond_threshold");

  m.def(
      "stress_cell_spectra",
      [spectra_of](const exokal::Mesh& mesh, int dim, StressOperators::Realization how,
                   StressOperators::Formulation form, double mu, double lam,
                   py::object degeneracy_percent, py::object cond_threshold) {
        const auto n = static_cast<std::size_t>(mesh.count(dim));
        const bool adaptive = how == StressOperators::Realization::adaptive_vem;
        std::vector<double> eta;
        if (adaptive && !degeneracy_percent.is_none()) {
          eta.assign(n, 1.0);
          for (const Index c :
               exokal::degenerate_cell_ids(mesh, dim, py::cast<double>(degeneracy_percent))) {
            eta[static_cast<std::size_t>(c)] = 0.0;
          }
        }
        const auto build = [&](const std::vector<double>* sel) {
          return StressOperators::build(mesh, dim, mu, lam, how, form, nullptr, nullptr, sel);
        };
        std::size_t ill = 0;
        if (adaptive && !cond_threshold.is_none()) {
          if (eta.empty()) eta.assign(n, 1.0);
          const std::vector<double> ones(n, 1.0);
          const StressOperators probe = build(&ones);
          static const exokal::numerics::Dense empty;
          for (std::size_t e = 0; e < n; ++e) {
            if (!probe.eta().empty() && probe.eta()[e] == 0.0) eta[e] = 0.0;
          }
          ill = mimetika::cond_selection(
              eta, py::cast<double>(cond_threshold),
              [&](std::size_t e) -> const exokal::numerics::Dense& {
                const auto& cell = probe.compact(static_cast<Index>(e));
                return cell.diag.empty() ? cell.M : empty;
              });
        }
        const StressOperators ops = build(eta.empty() ? nullptr : &eta);
        py::dict out = spectra_of(
            [&](Index e) {
              const auto& c = ops.compact(e);
              if (c.diag.empty()) return c.M;
              exokal::numerics::Dense d(c.diag.size(), c.diag.size());
              for (std::size_t i = 0; i < c.diag.size(); ++i) d(i, i) = c.diag[i];
              return d;
            },
            n);
        py::array_t<double> eta_out(static_cast<py::ssize_t>(n));
        for (std::size_t e = 0; e < n; ++e) {
          eta_out.mutable_data()[e] = ops.eta().empty() ? 1.0 : ops.eta()[e];
        }
        py::array_t<int> star(static_cast<py::ssize_t>(n));
        for (std::size_t e = 0; e < n; ++e) {
          int bad = 0;
          for (const double v : ops.compact(static_cast<Index>(e)).diag) bad |= !(v > 0.0);
          star.mutable_data()[e] = bad;
        }
        out["eta"] = eta_out;
        out["star_invalid"] = star;
        out["n_ill_conditioned"] = ill;
        return out;
      },
      // no enum default here: this binding precedes the enum's registration,
      // and a default of an unregistered type fails the whole module import
      py::arg("mesh"), py::arg("dim"), py::arg("realization"), py::arg("formulation"),
      py::arg("mu") = 1.0, py::arg("lam") = 1.0, py::arg("degeneracy_percent") = py::none(),
      py::arg("cond_threshold") = py::none(),
      "per-cell spectrum of the stress inner product, nothing assembled; for adaptive_vem "
      "eta is zeroed on the cells the scan flags and on those whose stabilized block's "
      "conditioning exceeds cond_threshold");

  // the one cell a boundary facet bounds; the affine datum is written about
  // that cell's centroid
  m.def(
      "cofacet_of",
      [](const exokal::Mesh& x, int cell_dim, Index facet) {
        return mimetika::cofacet_of(x, cell_dim, facet);
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("facet"));

  // Batched: the coboundary is built once for the whole array rather than once
  // per facet, which is what makes a loop over the boundary affordable.
  m.def(
      "cofacets_of",
      [](const exokal::Mesh& x, int cell_dim, const std::vector<Index>& facets) {
        const auto v = mimetika::cofacets_of(x, cell_dim, facets);
        py::array_t<Index> out(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return out;
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("facets"));

  // Writes the top cells with one tuple per cell of each named field. Values
  // are in cell order: field[i] belongs to cell i of dimension mesh.dim().
  m.def(
      "write_vtu",
      [](const exokal::Mesh& x, const std::string& path, const py::dict& fields, bool binary) {
        std::vector<exokal::CellField> cf;
        cf.reserve(fields.size());
        for (const auto& [k, v] : fields) {
          cf.push_back(cell_field(py::cast<std::string>(k), py::cast<py::array>(v)));
        }
        exokal::write_vtu(x, path, "tag", {},
                          binary ? exokal::VtuFormat::binary : exokal::VtuFormat::ascii, -1, cf);
      },
      py::arg("mesh"), py::arg("path"), py::arg("fields") = py::dict(), py::arg("binary") = false);

  // ---- the stress star -----------------------------------------------------
  py::enum_<StressOperators::Realization>(m, "StressRealization")
      .value("derham_bdm", StressOperators::Realization::derham_bdm)
      .value("derham_rt", StressOperators::Realization::derham_rt)
      .value("stabilized_bdm", StressOperators::Realization::stabilized_bdm)
      // NO RECONSTRUCTION: d per facet, one constant traction vector, and M the
      // diagonal primal-dual star at K = 2 mu -- the two-point stress. It needs
      // four fields, where the compliance is (2 mu)^-1 and applies
      // componentwise; in three the trace couples the components and no mesh
      // makes M diagonal. Consistent where the mesh is FACE-ORTHOGONAL, and
      // solvable everywhere: half the unknowns of the BDM products and an
      // eighth of the matrix entries.
      .value("diagonal_afw", StressOperators::Realization::diagonal_afw)
      // THE STRONGLY-SYMMETRIC FAMILY (Dassi-Lovadina-Visinoni): six traction
      // moments per facet carried whole, reconstruction onto constant
      // symmetric tensors, no rotation multiplier. stabilized_vem builds
      // either strong formulation; diagonal_vem is the two-point member and
      // adaptive_vem the per-cell selection between them -- both demand
      // strong_symmetry_total, where M can be diagonal at all.
      .value("stabilized_vem", StressOperators::Realization::stabilized_vem)
      .value("diagonal_vem", StressOperators::Realization::diagonal_vem)
      .value("adaptive_vem", StressOperators::Realization::adaptive_vem);

  // THREE FIELDS OR FOUR, which is a discretization and not a solver setting.
  // weak_symmetry carries the volumetric response in the compliance;
  // weak_symmetry_total gives the total pressure p = lambda div u a field of
  // its own, one scalar per cell, and the compliance is then lambda-free --
  // uniform in the incompressible limit, and the only form diagonal_afw has.
  py::enum_<StressOperators::Formulation>(m, "StressFormulation")
      .value("weak_symmetry", StressOperators::Formulation::weak_symmetry)
      .value("weak_symmetry_total", StressOperators::Formulation::weak_symmetry_total)
      // THE RIGID-MOTION ANSATZ: symmetry lives in the reconstruction space,
      // so there is no rotation field -- sigma and u, with the total pressure
      // independent in the _total form. The vem realizations build these.
      .value("strong_symmetry", StressOperators::Formulation::strong_symmetry)
      .value("strong_symmetry_total", StressOperators::Formulation::strong_symmetry_total);

  m.def("stress_formulation_name",
        static_cast<const char* (*)(StressOperators::Formulation)>(&StressOperators::name),
        py::arg("formulation"));

  m.def("stress_realization_name",
        static_cast<const char* (*)(StressOperators::Realization)>(&StressOperators::name),
        py::arg("realization"));

  // the static overload: what the space would carry, without building anything
  m.def(
      "stress_moments_per_facet",
      [](StressOperators::Realization r, int dim) {
        return StressOperators::moments_per_facet(r, dim);
      },
      py::arg("realization"), py::arg("dim"));

  // ---- the elasticity problem ---------------------------------------------
  py::class_<ElasticityProblem>(m, "ElasticityProblem")
      .def(py::init<exokal::Mesh, int, double, double, StressOperators::Realization,
                    const std::string&>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("mu") = 1.0, py::arg("lam") = 1.0,
           py::arg("realization") = StressOperators::Realization::derham_bdm,
           py::arg("model") = "linear_elasticity")
      .def_property_readonly("dim", &ElasticityProblem::dim)
      .def_property_readonly("n_dofs", &ElasticityProblem::n_dofs)
      .def_property_readonly("n_stabilized", &ElasticityProblem::n_stabilized)
      .def_property_readonly("moments_per_facet", &ElasticityProblem::moments_per_facet)
      .def("count", &ElasticityProblem::count, py::arg("k"))
      .def("constraint_block", &ElasticityProblem::constraint_block, py::arg("cell"))
      .def("boundary_facets", &ElasticityProblem::boundary_facets)
      .def("impose_traction", &ElasticityProblem::impose_traction, py::arg("field"),
           py::arg("facets"), py::arg("stress"))
      .def("impose_free_slip", &ElasticityProblem::impose_free_slip, py::arg("field"),
           py::arg("facets"))
      .def("freeze_constraints", &ElasticityProblem::freeze_constraints)
      .def("solve", &ElasticityProblem::solve)
      .def("dof", &ElasticityProblem::dof, py::arg("field"), py::arg("k"), py::arg("entity"),
           py::arg("local") = 0, py::arg("comp") = 0)
      .def("cofacet_of", &ElasticityProblem::cofacet_of, py::arg("facet"))
      .def("facet_frame", &ElasticityProblem::facet_frame, py::arg("facet"))
      .def("centroid", &ElasticityProblem::centroid, py::arg("k"), py::arg("cell"))
      .def("measure", &ElasticityProblem::measure, py::arg("k"), py::arg("cell"));

  // ---- the structured families --------------------------------------------
  py::enum_<mimetika::mesh::Family>(m, "Family")
      .value("cartesian", mimetika::mesh::Family::cartesian)
      .value("simplex", mimetika::mesh::Family::simplex)
      .value("prism", mimetika::mesh::Family::prism);

  m.def("family_name", &mimetika::mesh::name, py::arg("family"));
  m.def("column", &mimetika::mesh::column, py::arg("n"), py::arg("dim"), py::arg("family"),
        py::arg("height") = 1.0, py::arg("width") = 1.0);
  // A BOX, WHICH IS THE MESH A SCALING STUDY WANTS: the resolution is a
  // number per axis, the shape is a choice, and nothing about the geometry
  // varies as it is refined. The annulus curves and grades, so refining it
  // changes the conditioning as well as the size; a box changes only the size,
  // which is what makes a timing at two resolutions comparable.
  //
  // Cartesian gives one hexahedron per cell of the grid, simplex gives six
  // tetrahedra (the Freudenthal cut of the cube), and both are 2D as well:
  // quadrilaterals and two triangles.
  m.def("box", &mimetika::mesh::box, py::arg("n"), py::arg("dim"), py::arg("family"),
        py::arg("lengths") = std::array<double, 3>{1.0, 1.0, 1.0},
        py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
        "a structured box of n[k] cells along each axis");

  m.def("annulus", &mimetika::mesh::annulus, py::arg("nr"), py::arg("nt"), py::arg("dim"),
        py::arg("family"), py::arg("a") = 1.0, py::arg("b") = 10.0, py::arg("height") = 1.0,
        py::arg("layers") = 1);
  m.def(
      "boundary_outward_normal",
      [](const exokal::Mesh& x, int cell_dim, Index f) {
        return exokal::boundary_outward_normal(x, cell_dim, f);
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("facet"));

  // ---- the material --------------------------------------------------------
  py::class_<mimetika::ElasticMaterial>(m, "ElasticMaterial")
      .def(py::init(
               [](double shear, double lame) { return mimetika::ElasticMaterial{shear, lame}; }),
           py::arg("shear") = 1.0, py::arg("lame") = 1.0)
      .def_static("from_young_poisson", &mimetika::ElasticMaterial::from_young_poisson,
                  py::arg("E"), py::arg("nu"))
      .def_readwrite("shear", &mimetika::ElasticMaterial::shear)
      .def_readwrite("lame", &mimetika::ElasticMaterial::lame)
      .def("poisson", &mimetika::ElasticMaterial::poisson)
      .def("young", &mimetika::ElasticMaterial::young)
      .def("oedometer", &mimetika::ElasticMaterial::oedometer);

  // ---- the Cauchy elasticity model ----------------------------------------
  //
  // The boundary conditions are typed in C++ (`emplace<TractionBC>`), which a
  // template cannot carry across the binding, so each one gets a named adder.
  // Adding a condition here is the only change a new one needs.
  py::class_<mimetika::CauchyElasticityModel>(m, "CauchyElasticityModel")
      .def(py::init<const exokal::Mesh&, int, mimetika::ElasticMaterial,
                    StressOperators::Realization, StressOperators::Formulation>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("material"),
           py::arg("realization") = StressOperators::Realization::derham_bdm,
           py::arg("formulation") = StressOperators::Formulation::weak_symmetry,
           py::keep_alive<1, 2>())  // the mesh must outlive the model
      .def_property_readonly("formulation", &mimetika::CauchyElasticityModel::formulation)
      // p = lambda div u, the fourth unknown. It is SOLVED FOR rather than
      // reconstructed, so it exists only in the four-field formulation and
      // asking for it in three raises rather than returning a guess.
      .def("total_pressure", &mimetika::CauchyElasticityModel::total_pressure, py::arg("cell"))
      .def(
          "add_traction",
          [](mimetika::CauchyElasticityModel& s, const std::vector<Index>& facets,
             const std::array<double, 9>& stress) {
            s.mechanics().emplace<mimetika::TractionBC>(facets, stress);
          },
          py::arg("facets"), py::arg("stress"))
      .def(
          "add_free_slip",
          [](mimetika::CauchyElasticityModel& s, const std::vector<Index>& facets) {
            s.mechanics().emplace<mimetika::FreeSlipBC>(facets);
          },
          py::arg("facets"))
      .def_property_readonly("n_constrained",
                             [](const mimetika::CauchyElasticityModel& s) {
                               const auto& c = s.simulation().constraints();
                               std::size_t n = 0;
                               for (std::size_t i = 0; i < s.simulation().n_dofs(); ++i) {
                                 if (c.pinned(i)) ++n;
                               }
                               return n;
                             })
      .def("build", [](mimetika::CauchyElasticityModel& s) { s.build(); })
      .def(
          "assemble",
          [](mimetika::CauchyElasticityModel& s, bool progress,
             const mimetika::solver::SolverOptions& o) {
            return assemble_only(s, progress, o, false);
          },
          py::arg("progress") = false, py::arg("options") = mimetika::solver::SolverOptions{})
      .def("prescribe_displacement", &mimetika::CauchyElasticityModel::prescribe_displacement,
           py::arg("facets"), py::arg("constant"), py::arg("gradient") = std::array<double, 9>{})
      .def("solve_hybrid", &solve_elasticity_hybrid, py::arg("progress") = false,
           py::arg("options") = mimetika::solver::SolverOptions{})
      .def("solve", &solve_elasticity, py::arg("progress") = false,
           py::arg("options") = mimetika::solver::SolverOptions{})
      .def_property_readonly("dim", &mimetika::CauchyElasticityModel::dim)
      .def_property_readonly("n_cells", &mimetika::CauchyElasticityModel::n_cells)
      .def_property_readonly("n_stabilized", &mimetika::CauchyElasticityModel::n_stabilized)
      .def_property_readonly(
          "n_dofs",
          [](const mimetika::CauchyElasticityModel& s) { return s.simulation().n_dofs(); })
      .def_property_readonly("realization_name", &mimetika::CauchyElasticityModel::realization_name)
      .def("material", &mimetika::CauchyElasticityModel::material, py::return_value_policy::copy)
      .def("displacement", &mimetika::CauchyElasticityModel::displacement, py::arg("cell"),
           py::arg("axis"))
      .def("rotation", &mimetika::CauchyElasticityModel::rotation, py::arg("cell"),
           py::arg("generator"))
      .def_property_readonly("n_rotations", &mimetika::CauchyElasticityModel::n_rotations)
      .def("normal_traction", &mimetika::CauchyElasticityModel::normal_traction, py::arg("facet"))
      .def("cell_stress", &mimetika::CauchyElasticityModel::cell_stress, py::arg("cell"))
      .def("facet_traction", &mimetika::CauchyElasticityModel::facet_traction, py::arg("facet"))
      .def_property_readonly(
          "n_invalid_star", &mimetika::CauchyElasticityModel::n_invalid_star,
          "cells whose diagonal star carries a non-positive facet weight: the centroid "
          "does not see that facet squarely, M is not positive there, and the condensed "
          "solve is meaningless -- refuse or switch products when this is nonzero")
      .def("set_degeneracy_percent", &mimetika::CauchyElasticityModel::set_degeneracy_percent,
           py::arg("percent"),
           "adaptive_vem's scan threshold: cells whose measure falls below this "
           "percentage of their node-star mean take the diagonal star (eta = 0); "
           "every other cell keeps the stabilized vem product (eta = 1)")
      .def("set_cond_threshold", &mimetika::CauchyElasticityModel::set_cond_threshold,
           py::arg("cond"),
           "adaptive_vem's second selector: cells whose stabilized vem block has "
           "lambda_max / lambda_min above this take the diagonal star (eta = 0) as "
           "well; composes with the scan, costs one probe build")
      .def_property_readonly("n_ill_conditioned",
                             &mimetika::CauchyElasticityModel::n_ill_conditioned,
                             "cells the conditioning selector switched, as built")
      .def_property_readonly(
          "eta",
          [](const mimetika::CauchyElasticityModel& s) {
            const auto& e = s.eta();
            return py::array_t<double>(static_cast<py::ssize_t>(e.size()), e.data());
          },
          "the adaptive_vem selection as built, one value per cell: "
          "1 is stabilized_vem, 0 is diagonal_vem");

  m.def(
      "boundary_facets",
      [](const exokal::Mesh& x, int cell_dim) {
        return mimetika::boundary_facets(x.topology(), cell_dim);
      },
      py::arg("mesh"), py::arg("cell_dim"));

  // ---- the flux star and the single-phase model ---------------------------
  py::enum_<FluxOperators::Realization>(m, "FluxRealization")
      .value("derham_bdm", FluxOperators::Realization::derham_bdm)
      .value("derham_rt", FluxOperators::Realization::derham_rt)
      .value("stabilized_rt", FluxOperators::Realization::stabilized_rt)
      // ONE FLUX PER FACET AND NO RECONSTRUCTION: M is the diagonal
      // primal-dual star, which is the two-point flux approximation. Exact
      // where the mesh is K-ORTHOGONAL and only there -- see the flow example.
      .value("diagonal_tpfa", FluxOperators::Realization::diagonal_tpfa)
      // THE PER-CELL SELECTION between the two products above, carried as
      // eta in {0, 1}: ones everywhere -- the stabilized product -- and 0 on
      // the cells the metric-degeneracy scan flags, which take the diagonal
      // star instead of a reconstruction over a collapsed cell. The scan runs
      // at exokal's default threshold, or at the percentage
      // SinglePhaseModel.set_degeneracy_percent names; the model's .eta is
      // the selection as built, one value per cell.
      .value("adaptive_rt", FluxOperators::Realization::adaptive_rt);

  // HOW MANY PROCESSES THE SOLVER WILL USE, which is a question worth being
  // able to ask: launched under mpirun with a mismatched runtime, MPI falls
  // back to singletons and every rank solves the whole problem believing it is
  // alone. That looks like a working parallel run and is not one.
  m.def(
      "mpi_size",
      [] {
        mimetika::solver::PetscSession::instance();
        PetscMPIInt size = 1;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        return static_cast<int>(size);
      },
      "the number of processes in PETSC_COMM_WORLD");
  m.def(
      "mpi_rank",
      [] {
        mimetika::solver::PetscSession::instance();
        PetscMPIInt rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        return static_cast<int>(rank);
      },
      "this process's rank in PETSC_COMM_WORLD");

  // A CELL FIELD THE WHOLE MESH CAN SEE.
  //
  // Distributed, a process builds the per-cell operators of its own cells and
  // its halo, and no others -- that is what makes the assembly divide. So
  // anything reconstructed FROM those operators, stress above all, exists only
  // where they do; asked for elsewhere it comes back zero, which reads as a
  // wrong answer rather than as an absent one.
  //
  // Each cell is owned by exactly one process, so zeroing what this one does
  // not own and summing over all of them assembles the exact field, once. The
  // displacement and the rotation do not need this: they are read from the
  // solution, which every process already has in full.
  m.def(
      "gather_cells",
      [](const mimetika::CauchyElasticityModel& model, py::array_t<double> values) {
        const auto& owned = model.distribution().owned_cells;
        if (owned.empty()) return values;
        auto v = values.mutable_unchecked();
        const auto rows = static_cast<std::size_t>(values.shape(0));
        if (rows != owned.size()) {
          throw std::invalid_argument("gather_cells: not one row per cell");
        }
        const auto width = values.ndim() == 1
                               ? std::size_t{1}
                               : static_cast<std::size_t>(values.shape(1));
        double* data = values.mutable_data();
        for (std::size_t e = 0; e < rows; ++e) {
          if (owned[e] != 0) continue;
          for (std::size_t k = 0; k < width; ++k) data[e * width + k] = 0.0;
        }
        MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(rows * width), MPI_DOUBLE, MPI_SUM,
                      PETSC_COMM_WORLD);
        (void)v;
        return values;
      },
      py::arg("model"), py::arg("values"),
      "sum a per-cell field over the processes, each contributing the cells it owns");

  // the same for the flow model, whose flux is reconstructed per cell too
  m.def(
      "gather_cells",
      [](const mimetika::SinglePhaseModel& model, py::array_t<double> values) {
        const auto& owned = model.distribution().owned_cells;
        if (owned.empty()) return values;
        auto v = values.mutable_unchecked();
        const auto rows = static_cast<std::size_t>(values.shape(0));
        if (rows != owned.size()) {
          throw std::invalid_argument("gather_cells: not one row per cell");
        }
        const auto width = values.ndim() == 1
                               ? std::size_t{1}
                               : static_cast<std::size_t>(values.shape(1));
        double* data = values.mutable_data();
        for (std::size_t e = 0; e < rows; ++e) {
          if (owned[e] != 0) continue;
          for (std::size_t k = 0; k < width; ++k) data[e * width + k] = 0.0;
        }
        MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(rows * width), MPI_DOUBLE, MPI_SUM,
                      PETSC_COMM_WORLD);
        (void)v;
        return values;
      },
      py::arg("model"), py::arg("values"),
      "sum a per-cell field over the processes, each contributing the cells it owns");

  // THE PARTITION AS A FIELD, so it can be looked at rather than trusted.
  //
  // The same geometric bisection the solver uses, asked for any number of
  // parts: written into a .vtu it colours the mesh by process, which is how a
  // partition that has gone wrong -- a rank with a disconnected piece, or with
  // most of the mesh -- is seen rather than inferred from a timing.
  m.def(
      "cell_ranks",
      [](const exokal::Mesh& mesh, int dim, int n_ranks) {
        const auto part = mimetika::partition_cells(mesh, dim, std::max(n_ranks, 2), 0);
        py::array_t<int> out(static_cast<py::ssize_t>(mesh.count(dim)));
        const auto owners =
            exokal::spaces::rcb(exokal::spaces::cell_centroids(mesh, dim), std::max(n_ranks, 2));
        for (Index e = 0; e < mesh.count(dim); ++e) {
          out.mutable_data()[e] = owners.owner(e);
        }
        return out;
      },
      py::arg("mesh"), py::arg("dim"), py::arg("n_ranks"),
      "the rank owning each cell, under an n_ranks-way partition");

  m.def("flux_realization_name", &FluxOperators::name, py::arg("realization"));

  py::class_<mimetika::SinglePhaseModel>(m, "SinglePhaseModel")
      .def(py::init<const exokal::Mesh&, int, double, FluxOperators::Realization>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("mobility") = 1.0,
           py::arg("realization") = FluxOperators::Realization::derham_bdm,
           py::keep_alive<1, 2>())  // the mesh must outlive the model
      .def(
          "add_normal_flux",
          [](mimetika::SinglePhaseModel& s, const std::vector<Index>& facets) {
            s.flow().emplace<mimetika::NormalFluxBC>(facets);
          },
          py::arg("facets"))
      .def(
          "add_pressure",
          [](mimetika::SinglePhaseModel& s, const std::vector<Index>& facets, double value) {
            s.flow().emplace<mimetika::PressureBC>(facets, value);
          },
          py::arg("facets"), py::arg("value"))
      .def("set_degeneracy_percent", &mimetika::SinglePhaseModel::set_degeneracy_percent,
           py::arg("percent"),
           "adaptive_rt's scan threshold: cells whose measure falls below this "
           "percentage of their node-star mean take the diagonal star (eta = 0); "
           "every other cell keeps the stabilized product (eta = 1)")
      .def("set_cond_threshold", &mimetika::SinglePhaseModel::set_cond_threshold,
           py::arg("cond"),
           "adaptive_rt's second selector: cells whose stabilized flux block has "
           "lambda_max / lambda_min above this take the diagonal star (eta = 0) as "
           "well; composes with the scan, costs one probe build")
      .def_property_readonly("n_ill_conditioned", &mimetika::SinglePhaseModel::n_ill_conditioned,
                             "cells the conditioning selector switched, as built")
      .def_property_readonly("n_not_star_shaped", &mimetika::SinglePhaseModel::n_not_star_shaped,
                             "cells whose two-point star carries a non-positive facet weight: "
                             "the centroid does not see that facet, M is not positive there, "
                             "and the solve is meaningless -- repair the mesh or switch products")
      .def_property_readonly(
          "not_star_shaped",
          [](const mimetika::SinglePhaseModel& s) {
            const auto& v = s.not_star_shaped();
            return std::vector<long long>(v.begin(), v.end());
          },
          "the cells the star's premise fails on, by complex id")
      .def_property_readonly(
          "eta",
          [](const mimetika::SinglePhaseModel& s) {
            const auto& e = s.eta();
            return py::array_t<double>(static_cast<py::ssize_t>(e.size()), e.data());
          },
          "the adaptive_rt selection as built, one value per cell: "
          "1 is stabilized_rt, 0 is diagonal_tpfa")
      // THE OPERATORS AND THE SYSTEM, NOTHING MORE: what a diagnostic needs --
      // the star's validity, the selection as built -- without paying for a
      // preconditioner it will never apply
      .def("build", [](mimetika::SinglePhaseModel& s) { s.build(); })
      .def(
          "assemble",
          [](mimetika::SinglePhaseModel& s, bool progress,
             const mimetika::solver::SolverOptions& o) {
            return assemble_only(s, progress, o, true);
          },
          py::arg("progress") = false, py::arg("options") = mimetika::solver::SolverOptions{})
      .def("solve", &solve_single_phase, py::arg("progress") = false,
           py::arg("options") = mimetika::solver::SolverOptions{})
      .def("solve_hybrid", &solve_single_phase_hybrid, py::arg("progress") = false,
           py::arg("options") = mimetika::solver::SolverOptions{},
           "eliminate the flux cell by cell and solve the SPD facet-pressure system: "
           "cg + multigrid on any product; a pressure datum pins a multiplier, a "
           "normal flux loads a free row, an unconditioned boundary facet is sealed")
      .def_property_readonly("dim", &mimetika::SinglePhaseModel::dim)
      .def_property_readonly("n_cells", &mimetika::SinglePhaseModel::n_cells)
      .def_property_readonly(
          "n_dofs", [](const mimetika::SinglePhaseModel& s) { return s.simulation().n_dofs(); })
      .def_property_readonly("moments_per_facet", &mimetika::SinglePhaseModel::moments_per_facet)
      .def_property_readonly("realization_name", &mimetika::SinglePhaseModel::realization_name)
      .def("cell_pressure", &mimetika::SinglePhaseModel::cell_pressure, py::arg("cell"))
      .def("cell_flux", &mimetika::SinglePhaseModel::cell_flux, py::arg("cell"));

  // ---- the assembled model, for the structural checks ---------------------
  py::class_<AssembledModel>(m, "AssembledModel")
      .def(py::init<exokal::Mesh, int, const std::string&, double, double,
                    StressOperators::Realization, double, FluxOperators::Realization>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("model"), py::arg("mu") = 1.0,
           py::arg("lam") = 1.0,
           py::arg("stress_realization") = StressOperators::Realization::derham_bdm,
           py::arg("mobility") = 1.0,
           py::arg("flux_realization") = FluxOperators::Realization::derham_bdm)
      .def_property_readonly("n_dofs", &AssembledModel::n_dofs)
      .def_property_readonly("n_packages", &AssembledModel::n_packages)
      .def_property_readonly("n_fields", &AssembledModel::n_fields)
      .def_property_readonly("space_size", &AssembledModel::space_size)
      .def_property_readonly("stress_size", &AssembledModel::stress_size)
      .def_property_readonly("stress_n_stabilized", &AssembledModel::stress_n_stabilized)
      .def_property_readonly("flux_n_cells", &AssembledModel::flux_n_cells)
      .def("seed_state", &AssembledModel::seed_state)
      .def("freeze_constraints", &AssembledModel::freeze_constraints)
      .def("jacobian_coo", &AssembledModel::jacobian_coo)
      .def("field_range", &AssembledModel::field_range, py::arg("field"))
      .def("has", &AssembledModel::has, py::arg("field"))
      .def("field_size", &AssembledModel::field_size, py::arg("field"))
      .def("field_degree", &AssembledModel::field_degree, py::arg("field"));

  // ---- compositions and their spaces --------------------------------------
  //
  // A composition can be inspected without ever assembling: the space it lays
  // out is a function of the packages and the complex alone, which is what the
  // counts tests are about.
  py::class_<exokal::spaces::ProductSpace>(m, "ProductSpace")
      .def_property_readonly("n_fields", &exokal::spaces::ProductSpace::n_fields)
      .def_property_readonly("size", &exokal::spaces::ProductSpace::size)
      .def(
          "has",
          [](const exokal::spaces::ProductSpace& s, const std::string& f) { return s.has(f); },
          py::arg("field"))
      .def(
          "field_size",
          [](const exokal::spaces::ProductSpace& s, const std::string& f) {
            return static_cast<std::size_t>(s.map(s.index_of(f)).size());
          },
          py::arg("field"))
      .def(
          "field_degree",
          [](const exokal::spaces::ProductSpace& s, const std::string& f) {
            return s.map(s.index_of(f)).layout().degree();
          },
          py::arg("field"));

  py::class_<mimetika::physics::Composition>(m, "Composition")
      .def(py::init<>())
      .def_property_readonly("size", &mimetika::physics::Composition::size)
      .def("add_flow",
           [](mimetika::physics::Composition& s) { s.emplace<mimetika::physics::Flow>(); })
      .def("add_mechanics",
           [](mimetika::physics::Composition& s) { s.emplace<mimetika::physics::Mechanics>(); })
      .def("add_poro_coupling",
           [](mimetika::physics::Composition& s) { s.emplace<mimetika::physics::PoroCoupling>(); })
      .def(
          "validate", [](const mimetika::physics::Composition& s, int dim) { s.validate(dim); },
          py::arg("dim"))
      .def(
          "space",
          [](const mimetika::physics::Composition& s, const exokal::Mesh& mesh, int cell_dim) {
            return s.space(mesh.topology(), cell_dim);
          },
          py::arg("mesh"), py::arg("cell_dim"));

  // Composition owns unique_ptr packages, so it is move-only: it has to reach
  // Python through a holder rather than by value.
  m.def(
      "build_composition",
      [](const std::string& name) {
        return std::make_unique<mimetika::physics::Composition>(
            Catalogue::instance().build(name, {}));
      },
      py::arg("model"));

  // the stress star on its own, for the counts that are about the star rather
  // than about a model built on it
  m.def(
      "stress_operator_counts",
      [](const exokal::Mesh& mesh, int cell_dim, double mu, double lam,
         StressOperators::Realization how) {
        DeRhamGeometryCache geo = DeRhamGeometryCache::build(mesh, cell_dim);
        const StressOperators ops = StressOperators::build(
            mesh, cell_dim, mu, lam, how, StressOperators::Formulation::weak_symmetry, &geo);
        return py::make_tuple(ops.size(), ops.n_stabilized());
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("mu") = 1.0, py::arg("lam") = 1.0,
      py::arg("realization") = StressOperators::Realization::derham_bdm);

  m.def("catalogue_names", [] { return Catalogue::instance().names(); });
}
