// The Python interface to the DIRECT hypre path.
//
// A SEPARATE MODULE FROM mimetika_cxx, AND IT HAS TO BE. mimetika_cxx links
// PETSc, PETSc links its own libHYPRE, and two hypre copies in one process
// export the same HYPRE_* names -- which one a call reaches is then decided by
// load order rather than by intent. This module links hypre and nothing else,
// and its HYPRE symbols are hidden at link time (see python/CMakeLists.txt), so
// importing both in one interpreter is safe.
//
// It is self-contained rather than sharing mimetika_cxx's mesh and model: a
// pybind11 type registered in two modules is two types, so a mesh built there
// could not be handed here. mimetika is header-only, so compiling the model
// again costs build time and no dependency.
//
// The surface is deliberately small -- a mesh, a flow model, a solve -- because
// what this exists to expose is ADS itself, including the two options PETSc
// registers and never queries.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/io/vtu.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/hypre.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

namespace py = pybind11;
using graphos::Index;
using mimetika::FlowModel;
using mimetika::solver::HypreSolver;
using mimetika::solver::SpaceNorm;
using FluxRealization = FlowModel::Realization;

namespace {

// The norm the Riesz map is the Gram matrix of, for flow: the flux is the
// first factor and the pressure carries plain L2 with W the Schur scale of the
// divergence constraint, which for an unscaled incidence row is the cell
// measure. This is the same norm mimetika_cxx builds for PETSc -- stated again
// here rather than shared, because sharing it would mean importing the module
// that links PETSc.
SpaceNorm flow_norm(const FlowModel& model, const exokal::Mesh& mesh, int dim) {
  const auto blocks = mimetika::solver::field_blocks(model.simulation().epoch());
  if (blocks.size() < 2) throw std::runtime_error("flow: the space has fewer than two factors");
  const auto n_cells = static_cast<std::size_t>(mesh.count(dim));

  auto lo = mesh.point(0), hi = mesh.point(0);
  for (Index v = 1; v < mesh.count(0); ++v) {
    const auto& x = mesh.point(v);
    for (std::size_t d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], x[d]);
      hi[d] = std::max(hi[d], x[d]);
    }
  }
  double diag2 = 0.0;
  for (std::size_t d = 0; d < 3; ++d) diag2 += (hi[d] - lo[d]) * (hi[d] - lo[d]);
  const double scale = model.mobility() / diag2;

  SpaceNorm norm;
  std::vector<int> flux;
  flux.reserve(blocks[0].size());
  for (const Index g : blocks[0].indices()) flux.push_back(static_cast<int>(g));
  norm.factors.push_back(std::move(flux));

  // W CARRIES K. It stands for the Schur complement B star_K^-1 B^T of the
  // divergence constraint, and star_K^-1 scales with K, so W does too. The
  // measure alone names the norm of a HOMOGENEOUS coefficient only: with K
  // jumping across facets the pairing the map is built from is no longer the
  // pairing the operator has, and the count pays the inf-sup constant of the
  // mismatch -- measured, 17 iterations become the 500-step cap at 1e6.
  const std::vector<double> k_cell = model.norm_permeability();
  if (!k_cell.empty() && k_cell.size() != n_cells) {
    throw std::runtime_error("flow: the coefficient does not cover the cells");
  }
  std::vector<int> rest;
  std::vector<double> l2;
  for (std::size_t f = 1; f < blocks.size(); ++f) {
    const std::size_t components = blocks[f].size() / n_cells;
    std::size_t k = 0;
    for (const Index g : blocks[f].indices()) {
      rest.push_back(static_cast<int>(g));
      const std::size_t cell = k / components;
      const double k_e = k_cell.empty() ? 1.0 : k_cell[cell];
      l2.push_back(scale * k_e * exokal::measure(mesh, dim, static_cast<Index>(cell)));
      ++k;
    }
  }
  norm.factors.push_back(std::move(rest));
  norm.l2_weight.push_back(std::move(l2));

  const graphos::Complex& topo = mesh.topology();
  const auto incidence = [](auto b, int rows, int cols) {
    SpaceNorm::Incidence out;
    out.rows = rows;
    out.cols = cols;
    for (Index r = 0; r < rows; ++r) {
      for (auto k = b.offsets[static_cast<std::size_t>(r)];
           k < b.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
        out.row.push_back(static_cast<int>(r));
        out.col.push_back(static_cast<int>(b.indices[static_cast<std::size_t>(k)]));
        out.value.push_back(static_cast<double>(b.signs[static_cast<std::size_t>(k)]));
      }
    }
    return out;
  };
  norm.discrete_gradient = incidence(topo.boundary(1), topo.count(1), topo.count(0));
  norm.discrete_curl = incidence(topo.boundary(2), topo.count(2), topo.count(1));
  norm.space_dim = 3;
  norm.vertex_coordinates.reserve(static_cast<std::size_t>(topo.count(0)) * 3);
  for (Index v = 0; v < topo.count(0); ++v) {
    const auto& x = mesh.point(v);
    norm.vertex_coordinates.insert(norm.vertex_coordinates.end(), {x[0], x[1], x[2]});
  }
  return norm;
}

}  // namespace

PYBIND11_MODULE(_hypre, m) {
  m.doc() =
      "mimetika through hypre's ADS, called directly rather than through PETSc's PCHYPRE";
  m.attr("hypre_version") = HYPRE_RELEASE_VERSION;

  // MPI, up front. Building a model partitions it, and that asks MPI for the
  // communicator size; on the PETSc path PetscInitialize has already run by
  // then, and here nothing has. Idempotent.
  m.def("init", &mimetika::solver::HypreSession::ensure,
        "Initialize MPI and hypre. Call before building a model.");

  py::class_<exokal::Mesh>(m, "Mesh", py::module_local())
      .def("count", &exokal::Mesh::count, py::arg("k"))
      .def("point", [](const exokal::Mesh& x, Index v) {
        const auto& p = x.point(v);
        return std::array<double, 3>{p[0], p[1], p[2]};
      });

  py::enum_<mimetika::mesh::Family>(m, "Family", py::module_local())
      .value("cartesian", mimetika::mesh::Family::cartesian)
      .value("simplex", mimetika::mesh::Family::simplex)
      .value("prism", mimetika::mesh::Family::prism);

  m.def("box", &mimetika::mesh::box, py::arg("n"), py::arg("dim"), py::arg("family"),
        py::arg("lengths") = std::array<double, 3>{1.0, 1.0, 1.0},
        py::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
        "a structured box of n[k] cells along each axis");
  m.def("read_vtu", &exokal::read_vtu, py::arg("path"), py::arg("tag_array") = "tag");
  m.def(
      "boundary_facets",
      [](const exokal::Mesh& x, int cell_dim) {
        return mimetika::boundary_facets(x.topology(), cell_dim);
      },
      py::arg("mesh"), py::arg("cell_dim"));
  m.def(
      "centroid", [](const exokal::Mesh& x, int k, Index i) { return exokal::centroid(x, k, i); },
      py::arg("mesh"), py::arg("k"), py::arg("entity"));

  // ADS is written for ONE unknown per facet in 3D. A facet carrying d moments
  // reaches it only through the facet-constant subspace, which the direct path
  // does not build, so those realizations are not offered here.
  py::enum_<FluxRealization>(m, "FluxRealization", py::module_local())
      .value("derham_rt", FluxRealization::derham_rt)
      .value("stabilized_rt", FluxRealization::stabilized_rt)
      .value("adaptive_rt", FluxRealization::adaptive_rt);

  py::class_<HypreSolver::Options>(m, "AdsOptions", py::module_local())
      .def(py::init<>())
      .def_readwrite("rtol", &HypreSolver::Options::rtol)
      .def_readwrite("max_iterations", &HypreSolver::Options::max_iterations)
      .def_readwrite("cycle_type", &HypreSolver::Options::ads_cycle_type,
                     "1-8 use the monolithic Pi, 11-14 three scalar AMG solves. 13 is the "
                     "5-level multiplicative (034515430)")
      .def_readwrite("amg_theta", &HypreSolver::Options::amg_theta,
                     "strength threshold of the vector AMG inside ADS -- PETSc registers "
                     "-pc_hypre_ads_amg_theta and never queries it, so this is reachable "
                     "only on the direct path")
      .def_readwrite("ams_theta", &HypreSolver::Options::ams_theta,
                     "the same for the AMS solve inside ADS")
      .def_readwrite("print_level", &HypreSolver::Options::print_level)
      .def_readwrite("amg_interp_type", &HypreSolver::Options::amg_interp_type)
      .def_readwrite("amg_pmax", &HypreSolver::Options::amg_pmax)
      .def_readwrite("block_iterations", &HypreSolver::Options::block_iterations,
                     "inner CG steps on the block; 0 applies one ADS cycle instead")
      .def_readwrite("block_rtol", &HypreSolver::Options::block_rtol)
      .def_readwrite("ads_iterations", &HypreSolver::Options::ads_iterations,
                     "cycles per application of the block; 1 is the Riesz map itself");

  py::class_<HypreSolver::Report>(m, "AdsReport", py::module_local())
      .def_readonly("converged", &HypreSolver::Report::converged)
      .def_readonly("iterations", &HypreSolver::Report::iterations)
      .def_readonly("residual", &HypreSolver::Report::residual)
      .def_readonly("setup_seconds", &HypreSolver::Report::setup_seconds)
      .def_readonly("solve_seconds", &HypreSolver::Report::solve_seconds)
      .def_readonly("reason", &HypreSolver::Report::reason)
      .def("__repr__", [](const HypreSolver::Report& r) {
        return "<AdsReport " + std::to_string(r.iterations) + " its, " + r.reason + ">";
      });

  // The solve, taking the system and its norm as arrays.
  //
  // This is how mimetika_cxx reaches ADS: it assembles and calls ads_handoff,
  // which returns exactly these fields, and the answer goes back through
  // mimetika_cxx.accept. Nothing but plain data crosses, because the two
  // modules cannot share a type.
  m.def(
      "solve_system",
      [](int n, py::array_t<int> row, py::array_t<int> col, py::array_t<double> value,
         py::array_t<double> rhs, py::array_t<int> flux, py::array_t<int> rest,
         py::array_t<double> l2_weight, const py::dict& gradient, const py::dict& curl,
         py::array_t<double> coordinates, int space_dim, int moments_per_facet,
         py::array_t<int> pinned, py::array_t<double> pinned_diagonal,
         py::array_t<int> owner_of_dof, py::array_t<int> vertex_owner,
         py::array_t<int> edge_owner, py::array_t<int> face_owner,
         const py::dict& lowest_order, int lowest_order_components,
         const py::dict& rt_interpolation, const py::dict& nd_interpolation,
         py::array_t<int> interpolation_owner, bool degree2,
         const HypreSolver::Options& opts) {
        (void)degree2;
        const auto ints = [](const py::array_t<int>& a) {
          return std::vector<int>(a.data(), a.data() + a.size());
        };
        const auto reals = [](const py::array_t<double>& a) {
          return std::vector<double>(a.data(), a.data() + a.size());
        };
        const auto inc = [&](const py::dict& d) {
          SpaceNorm::Incidence i;
          i.rows = d["rows"].cast<int>();
          i.cols = d["cols"].cast<int>();
          i.row = ints(d["row"].cast<py::array_t<int>>());
          i.col = ints(d["col"].cast<py::array_t<int>>());
          i.value = reals(d["value"].cast<py::array_t<double>>());
          return i;
        };
        mimetika::solver::SparseSystem A;
        A.n = static_cast<std::size_t>(n);
        for (py::ssize_t k = 0; k < value.size(); ++k) {
          A.row.push_back(row.data()[k]);
          A.col.push_back(col.data()[k]);
          A.value.push_back(value.data()[k]);
        }
        SpaceNorm norm;
        norm.factors.push_back(ints(flux));
        norm.factors.push_back(ints(rest));
        norm.l2_weight.push_back(reals(l2_weight));
        norm.discrete_gradient = inc(gradient);
        norm.discrete_curl = inc(curl);
        // the degree-2 complex's interpolations, when ADS is taking the BDM
        // block directly rather than through the facet-constant subspace
        norm.rt_interpolation = inc(rt_interpolation);
        norm.nd_interpolation = inc(nd_interpolation);
        norm.interpolation_owner = ints(interpolation_owner);
        norm.vertex_coordinates = reals(coordinates);
        norm.space_dim = space_dim;
        norm.pinned = ints(pinned);
        norm.pinned_diagonal = reals(pinned_diagonal);
        if (vertex_owner.size() > 0) {
          norm.entity_owner.push_back(ints(vertex_owner));
          norm.entity_owner.push_back(ints(edge_owner));
          norm.entity_owner.push_back(ints(face_owner));
        }
        norm.lowest_order = inc(lowest_order);
        norm.lowest_order_components = lowest_order_components;

        HypreSolver hypre;
        hypre.set_owners(ints(owner_of_dof));
        std::vector<double> x;
        const auto r = hypre.solve(A, reals(rhs), x, norm, opts);
        py::array_t<double> out(static_cast<py::ssize_t>(x.size()));
        std::copy(x.begin(), x.end(), out.mutable_data());
        return py::make_tuple(out, r);
      },
      py::arg("n"), py::arg("row"), py::arg("col"), py::arg("value"), py::arg("rhs"),
      py::arg("flux"), py::arg("rest"), py::arg("l2_weight"), py::arg("gradient"),
      py::arg("curl"), py::arg("coordinates"), py::arg("space_dim"),
      py::arg("moments_per_facet"), py::arg("pinned"), py::arg("pinned_diagonal"),
      py::arg("owner_of_dof"), py::arg("vertex_owner"), py::arg("edge_owner"),
      py::arg("face_owner"), py::arg("lowest_order"), py::arg("lowest_order_components"),
      py::arg("rt_interpolation"), py::arg("nd_interpolation"), py::arg("interpolation_owner"),
      py::arg("degree2") = false,
      py::arg("options") = HypreSolver::Options{},
      "Solve an assembled system with the Riesz map, its first block inverted by ADS.");

  py::class_<FlowModel>(m, "FlowModel", py::module_local())
      .def(py::init<const exokal::Mesh&, int, double, FluxRealization>(), py::arg("mesh"),
           py::arg("cell_dim"), py::arg("mobility") = 1.0,
           py::arg("realization") = FluxRealization::stabilized_rt,
           py::keep_alive<1, 2>())
      .def(
          "add_pressure",
          [](FlowModel& s, const std::vector<Index>& facets, double value) {
            s.flow().emplace<mimetika::PressureBC>(facets, value);
          },
          py::arg("facets"), py::arg("value"))
      .def(
          "add_normal_flux",
          [](FlowModel& s, const std::vector<Index>& facets) {
            s.flow().emplace<mimetika::NormalFluxBC>(facets);
          },
          py::arg("facets"))
      .def(
          "set_permeability",
          [](FlowModel& s, std::vector<double> k) { s.set_permeability(std::move(k)); },
          py::arg("per_cell"))
      .def_property_readonly("n_cells", &FlowModel::n_cells)
      .def_property_readonly("n_dofs",
                             [](const FlowModel& s) { return s.simulation().n_dofs(); })
      .def("cell_pressure", &FlowModel::cell_pressure, py::arg("cell"))
      .def(
          "solve",
          [](FlowModel& s, const exokal::Mesh& mesh, int dim,
             const HypreSolver::Options& opts) {
            s.build();
            const SpaceNorm norm = flow_norm(s, mesh, dim);
            HypreSolver hypre;
            std::vector<double> x;
            const auto r = hypre.solve(s.system(), s.rhs(), x, norm, opts);
            s.accept(x);
            return r;
          },
          py::arg("mesh"), py::arg("cell_dim"), py::arg("options") = HypreSolver::Options{},
          "Assemble and solve with the Riesz map, its first block inverted by ADS.");
}
