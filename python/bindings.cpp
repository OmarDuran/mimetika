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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/model/compositions/elasticity.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/model/compositions/single_phase_flow.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/model/single_phase_model.hpp"
#include "mimetika/solver/petsc.hpp"

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
  py::array_t<double> out({static_cast<py::ssize_t>(A.rows()),
                           static_cast<py::ssize_t>(A.cols())});
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
  const exokal::spaces::ProductSpace& space() const {
    return sim_->epoch().stratum(0).space();
  }

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
  AssembledModel(exokal::Mesh mesh, int cell_dim, const std::string& model, double mu,
                 double lam, StressOperators::Realization stress_how, double mobility,
                 FluxOperators::Realization flux_how)
      : mesh_(std::move(mesh)), dim_(cell_dim) {
    geo_ = DeRhamGeometryCache::build(mesh_, cell_dim);
    stress_ = StressOperators::build(mesh_, cell_dim, mu, lam, stress_how, &geo_);
    flux_ = FluxOperators::build(mesh_, exokal::constitutive::Coefficient::uniform(mobility),
                                 flux_how);
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
  const exokal::spaces::ProductSpace& space() const {
    return sim_->epoch().stratum(0).space();
  }

  exokal::Mesh mesh_;
  int dim_;
  DeRhamGeometryCache geo_;
  StressOperators stress_;
  FluxOperators flux_;
  exokal::forms::TermContext ctx_;
  mimetika::physics::Composition comp_;
  std::unique_ptr<Simulation> sim_;
};

// Build, solve and accept, in one call. The three are inseparable -- a model
// whose state was never accepted answers `displacement` with the wrong vector
// -- so exposing them apart would only create a way to get it wrong.
void solve_elasticity(mimetika::CauchyElasticityModel& m) {
  m.build();
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  if (!rep.converged) throw std::runtime_error("cauchy elasticity: " + rep.reason);
  m.accept(std::move(x));
}

void solve_single_phase(mimetika::SinglePhaseModel& m) {
  m.build();
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  if (!rep.converged) throw std::runtime_error("single phase: " + rep.reason);
  m.accept(std::move(x));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python interface to the mimetika C++ application";

  // ---- mesh ----------------------------------------------------------------
  py::class_<exokal::Mesh>(m, "Mesh")
      .def_static("from_simplices", &exokal::Mesh::from_simplices, py::arg("dim"),
                  py::arg("points"), py::arg("cells"))
      .def_static("from_polygons", &exokal::Mesh::from_polygons, py::arg("points"),
                  py::arg("cells"))
      .def_static("from_polyhedra", &exokal::Mesh::from_polyhedra, py::arg("points"),
                  py::arg("cells"))
      .def_property_readonly("dim", &exokal::Mesh::dim)
      .def("count", &exokal::Mesh::count, py::arg("k"))
      .def("__repr__", [](const exokal::Mesh& x) {
        return "Mesh(dim=" + std::to_string(x.dim()) + ")";
      });

  m.def("centroid", [](const exokal::Mesh& x, int k, Index i) { return exokal::centroid(x, k, i); },
        py::arg("mesh"), py::arg("k"), py::arg("cell"));
  m.def("measure", [](const exokal::Mesh& x, int k, Index i) { return exokal::measure(x, k, i); },
        py::arg("mesh"), py::arg("k"), py::arg("cell"));

  // ---- the stress star -----------------------------------------------------
  py::enum_<StressOperators::Realization>(m, "StressRealization")
      .value("derham_afw", StressOperators::Realization::derham_afw)
      .value("derham_afw_rt", StressOperators::Realization::derham_afw_rt)
      .value("stabilized_afw", StressOperators::Realization::stabilized_afw);

  m.def("stress_realization_name", &StressOperators::name, py::arg("realization"));

  // the static overload: what the space would carry, without building anything
  m.def("stress_moments_per_facet",
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
  m.def("boundary_outward_normal",
        [](const exokal::Mesh& x, int cell_dim, Index f) {
          return exokal::boundary_outward_normal(x, cell_dim, f);
        },
        py::arg("mesh"), py::arg("cell_dim"), py::arg("facet"));

  // ---- the material --------------------------------------------------------
  py::class_<mimetika::ElasticMaterial>(m, "ElasticMaterial")
      .def(py::init([](double shear, double lame) {
             return mimetika::ElasticMaterial{shear, lame};
           }),
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
      .def("add_traction",
           [](mimetika::CauchyElasticityModel& s, const std::vector<Index>& facets,
              const std::array<double, 9>& stress) {
             s.mechanics().emplace<mimetika::TractionBC>(facets, stress);
           },
           py::arg("facets"), py::arg("stress"))
      .def("add_free_slip",
           [](mimetika::CauchyElasticityModel& s, const std::vector<Index>& facets) {
             s.mechanics().emplace<mimetika::FreeSlipBC>(facets);
           },
           py::arg("facets"))
      .def("solve", &solve_elasticity)
      .def_property_readonly("dim", &mimetika::CauchyElasticityModel::dim)
      .def_property_readonly("n_cells", &mimetika::CauchyElasticityModel::n_cells)
      .def_property_readonly("n_stabilized", &mimetika::CauchyElasticityModel::n_stabilized)
      .def_property_readonly(
          "n_dofs",
          [](const mimetika::CauchyElasticityModel& s) { return s.simulation().n_dofs(); })
      .def_property_readonly("realization_name",
                             &mimetika::CauchyElasticityModel::realization_name)
      .def("material", &mimetika::CauchyElasticityModel::material,
           py::return_value_policy::copy)
      .def("displacement", &mimetika::CauchyElasticityModel::displacement, py::arg("cell"),
           py::arg("axis"))
      .def("normal_traction", &mimetika::CauchyElasticityModel::normal_traction,
           py::arg("facet"))
      .def("cell_stress", &mimetika::CauchyElasticityModel::cell_stress, py::arg("cell"))
      .def("facet_traction", &mimetika::CauchyElasticityModel::facet_traction,
           py::arg("facet"));

  m.def("boundary_facets",
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
      .def("add_normal_flux",
           [](mimetika::SinglePhaseModel& s, const std::vector<Index>& facets) {
             s.flow().emplace<mimetika::NormalFluxBC>(facets);
           },
           py::arg("facets"))
      .def("add_pressure",
           [](mimetika::SinglePhaseModel& s, const std::vector<Index>& facets, double value) {
             s.flow().emplace<mimetika::PressureBC>(facets, value);
           },
           py::arg("facets"), py::arg("value"))
      .def("solve", &solve_single_phase)
      .def_property_readonly("dim", &mimetika::SinglePhaseModel::dim)
      .def_property_readonly("n_cells", &mimetika::SinglePhaseModel::n_cells)
      .def_property_readonly(
          "n_dofs", [](const mimetika::SinglePhaseModel& s) { return s.simulation().n_dofs(); })
      .def_property_readonly("moments_per_facet",
                             &mimetika::SinglePhaseModel::moments_per_facet)
      .def_property_readonly("realization_name",
                             &mimetika::SinglePhaseModel::realization_name)
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
      .def("has", [](const exokal::spaces::ProductSpace& s,
                     const std::string& f) { return s.has(f); },
           py::arg("field"))
      .def("field_size",
           [](const exokal::spaces::ProductSpace& s, const std::string& f) {
             return static_cast<std::size_t>(s.map(s.index_of(f)).size());
           },
           py::arg("field"))
      .def("field_degree",
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
           [](mimetika::physics::Composition& s) {
             s.emplace<mimetika::physics::PoroCoupling>();
           })
      .def("validate",
           [](const mimetika::physics::Composition& s, int dim) { s.validate(dim); },
           py::arg("dim"))
      .def("space",
           [](const mimetika::physics::Composition& s, const exokal::Mesh& mesh, int cell_dim) {
             return s.space(mesh.topology(), cell_dim);
           },
           py::arg("mesh"), py::arg("cell_dim"));

  // Composition owns unique_ptr packages, so it is move-only: it has to reach
  // Python through a holder rather than by value.
  m.def("build_composition",
        [](const std::string& name) {
          return std::make_unique<mimetika::physics::Composition>(
              Catalogue::instance().build(name, {}));
        },
        py::arg("model"));

  // the stress star on its own, for the counts that are about the star rather
  // than about a model built on it
  m.def("stress_operator_counts",
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
