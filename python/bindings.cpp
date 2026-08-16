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
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "exokal/io/vtu.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/boundary_conditions.hpp"
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
    const bool bdm = how != StressOperators::Realization::derham_afw_rt;
    if (bdm) geo_ = DeRhamGeometryCache::build(mesh_, cell_dim);
    ops_ = StressOperators::build(mesh_, cell_dim, mu, lam, how, bdm ? &geo_ : nullptr);
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
    stress_ = StressOperators::build(mesh_, cell_dim, mu, lam, stress_how, &geo_);
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
class Stage {
 public:
  explicit Stage(bool on) : on_(on) {}

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
template <class Model>
void attach_norm(mimetika::solver::PetscSolver& petsc, const Model& m, const exokal::Mesh& mesh,
                 int dim, bool divergence_is_an_integral) {
  const auto blocks = mimetika::solver::field_blocks(m.simulation().epoch());
  if (blocks.size() < 2) {
    throw std::runtime_error("riesz: the space has fewer than two factors");
  }
  const auto n_cells = static_cast<std::size_t>(mesh.count(dim));
  std::vector<double> measure(n_cells);
  for (std::size_t e = 0; e < n_cells; ++e) {
    measure[e] = exokal::measure(mesh, dim, static_cast<Index>(e));
  }

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
      l2.push_back(divergence_is_an_integral ? me : 1.0 / me);
      ++k;
    }
  }
  norm.factors.push_back(std::move(rest));
  norm.l2_weight.push_back(std::move(l2));

  // the constrained unknowns, with the diagonal A gave them
  const auto& c = m.simulation().constraints();
  for (std::size_t i = 0; i < m.simulation().n_dofs(); ++i) {
    if (!c.pinned(i)) continue;
    norm.pinned.push_back(static_cast<int>(i));
    norm.pinned_diagonal.push_back(c.scale_at(i));
  }
  petsc.set_norm(std::move(norm));
}

mimetika::solver::SolveReport solve_elasticity(mimetika::CauchyElasticityModel& m, bool progress,
                                               const mimetika::solver::SolverOptions& opts) {
  Stage stage(progress);
  stage.begin("assembling");
  m.build();
  stage.end();
  mimetika::solver::PetscSolver petsc(opts);
  // the momentum row is Dv, whose entries already carry 1/|E|: it is an average
  if (opts.preconditioner == "riesz") attach_norm(petsc, m, m.mesh(), m.dim(), false);
  std::vector<double> x;
  // THE SOLVER TIMES ITS OWN THREE, because they are not one stage: the matrix
  // is linear in the assembly, the preconditioner is what decides whether a
  // mesh is reachable, and the iteration is what the preconditioner shortens.
  stage.begin("solving");
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  stage.end();
  stage.done("  matrix", rep.matrix_seconds);
  stage.done(opts.direct() ? "  factorization" : "  preconditioner", rep.preconditioner_seconds);
  stage.done("  iteration", rep.solve_seconds);
  if (!rep.converged) throw std::runtime_error("cauchy elasticity: " + rep.reason);
  m.accept(std::move(x));
  return rep;
}

mimetika::solver::SolveReport solve_single_phase(mimetika::SinglePhaseModel& m, bool progress,
                                                 const mimetika::solver::SolverOptions& opts) {
  Stage stage(progress);
  stage.begin("assembling");
  m.build();
  stage.end();
  mimetika::solver::PetscSolver petsc(opts);
  // the mass-balance row is the incidence: (Bq)_E is the integral of div q
  if (opts.preconditioner == "riesz") attach_norm(petsc, m, m.mesh(), m.dim(), true);
  std::vector<double> x;
  // THE SOLVER TIMES ITS OWN THREE, because they are not one stage: the matrix
  // is linear in the assembly, the preconditioner is what decides whether a
  // mesh is reachable, and the iteration is what the preconditioner shortens.
  stage.begin("solving");
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  stage.end();
  stage.done("  matrix", rep.matrix_seconds);
  stage.done(opts.direct() ? "  factorization" : "  preconditioner", rep.preconditioner_seconds);
  stage.done("  iteration", rep.solve_seconds);
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
                       std::string riesz_block_pc, int riesz_block_its, double riesz_block_rtol,
                       int riesz_exact_limit, double rtol, double atol, int max_iterations) {
             return mimetika::solver::SolverOptions{std::move(method),
                                                    std::move(factorization),
                                                    std::move(preconditioner),
                                                    std::move(riesz_block_pc),
                                                    riesz_block_its,
                                                    riesz_block_rtol,
                                                    riesz_exact_limit,
                                                    rtol,
                                                    atol,
                                                    max_iterations};
           }),
           py::arg("method") = "direct", py::arg("factorization") = "superlu",
           py::arg("preconditioner") = "lu", py::arg("riesz_block_pc") = "",
           py::arg("riesz_block_its") = -1, py::arg("riesz_block_rtol") = 1e-4,
           py::arg("riesz_exact_limit") = 50000, py::arg("rtol") = 1e-10,
           py::arg("atol") = 1e-50, py::arg("max_iterations") = 1000)
      .def_readwrite("riesz_block_pc", &mimetika::solver::SolverOptions::riesz_block_pc)
      .def_readwrite("riesz_block_its", &mimetika::solver::SolverOptions::riesz_block_its)
      .def_readwrite("riesz_block_rtol", &mimetika::solver::SolverOptions::riesz_block_rtol)
      .def_readwrite("riesz_exact_limit", &mimetika::solver::SolverOptions::riesz_exact_limit)
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

  // the one cell a boundary facet bounds; the affine datum is written about
  // that cell's centroid
  m.def(
      "cofacet_of",
      [](const exokal::Mesh& x, int cell_dim, Index facet) {
        return mimetika::cofacet_of(x, cell_dim, facet);
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("facet"));

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
      .value("derham_afw", StressOperators::Realization::derham_afw)
      .value("derham_afw_rt", StressOperators::Realization::derham_afw_rt)
      .value("stabilized_afw", StressOperators::Realization::stabilized_afw);

  m.def("stress_realization_name", &StressOperators::name, py::arg("realization"));

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
           py::arg("realization") = StressOperators::Realization::derham_afw,
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
                    StressOperators::Realization>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("material"),
           py::arg("realization") = StressOperators::Realization::derham_afw,
           py::keep_alive<1, 2>())  // the mesh must outlive the model
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
      .def("prescribe_displacement", &mimetika::CauchyElasticityModel::prescribe_displacement,
           py::arg("facets"), py::arg("constant"), py::arg("gradient") = std::array<double, 9>{})
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
      .def("facet_traction", &mimetika::CauchyElasticityModel::facet_traction, py::arg("facet"));

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
      .value("stabilized_rt", FluxOperators::Realization::stabilized_rt);

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
      .def("solve", &solve_single_phase, py::arg("progress") = false,
           py::arg("options") = mimetika::solver::SolverOptions{})
      .def_property_readonly("dim", &mimetika::SinglePhaseModel::dim)
      .def_property_readonly("n_cells", &mimetika::SinglePhaseModel::n_cells)
      .def_property_readonly(
          "n_dofs", [](const mimetika::SinglePhaseModel& s) { return s.simulation().n_dofs(); })
      .def_property_readonly("moments_per_facet", &mimetika::SinglePhaseModel::moments_per_facet)
      .def_property_readonly("realization_name", &mimetika::SinglePhaseModel::realization_name)
      .def("cell_pressure", &mimetika::SinglePhaseModel::cell_pressure, py::arg("cell"));

  // ---- the assembled model, for the structural checks ---------------------
  py::class_<AssembledModel>(m, "AssembledModel")
      .def(py::init<exokal::Mesh, int, const std::string&, double, double,
                    StressOperators::Realization, double, FluxOperators::Realization>(),
           py::arg("mesh"), py::arg("cell_dim"), py::arg("model"), py::arg("mu") = 1.0,
           py::arg("lam") = 1.0,
           py::arg("stress_realization") = StressOperators::Realization::derham_afw,
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
        const StressOperators ops = StressOperators::build(mesh, cell_dim, mu, lam, how, &geo);
        return py::make_tuple(ops.size(), ops.n_stabilized());
      },
      py::arg("mesh"), py::arg("cell_dim"), py::arg("mu") = 1.0, py::arg("lam") = 1.0,
      py::arg("realization") = StressOperators::Realization::derham_afw);

  m.def("catalogue_names", [] { return Catalogue::instance().names(); });
}
