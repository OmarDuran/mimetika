#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/model/simulation.hpp"

// EXPORTING AN ASSEMBLED SYSTEM, so the linear algebra can happen elsewhere.
//
// The claim being tested during the port is that the C++ assembles the SAME
// operator the Python does. Nothing about that claim needs a C++ solver, and
// building one to check it would put a second unvalidated component between
// the assembly and the answer — so the system is written out and solved where
// a solver already exists and is trusted.
//
// WHAT IS WRITTEN IS EVERYTHING NEEDED TO REBUILD THE PROBLEM, not just the
// matrix: the triplets, the residual, the field layout, and the constraint
// mask with its values. A comparison that had only the matrix could not tell
// a difference in the operator from a difference in what was pinned, and
// those fail in completely different ways.
//
// The format is deliberately dull — a magic string, counts, then flat arrays
// in native byte order. It is read by `python/read_system.py` in twenty
// lines, and it is not a serialization format anyone should grow attached to:
// when the C++ side gains a solver this file stops being on the critical path.

namespace mimetika::io {

inline void write_system(const std::string& path, const Simulation& sim,
                         const exokal::forms::TripletSink& jac) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("write_system: cannot open " + path);

  const auto put = [&f](const void* p, std::size_t n) {
    f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
  };
  const auto put_i64 = [&](std::int64_t v) { put(&v, sizeof(v)); };

  f.write("MIMSYS01", 8);

  const auto n = static_cast<std::int64_t>(sim.n_dofs());
  const auto nnz = static_cast<std::int64_t>(jac.nnz());
  put_i64(n);
  put_i64(nnz);

  // the field layout, so a reader can slice the solution into physics
  const auto& sp = sim.epoch().stratum(0).space();
  put_i64(static_cast<std::int64_t>(sp.n_fields()));
  for (std::size_t i = 0; i < sp.n_fields(); ++i) {
    const std::string& name = sp.name(i);
    put_i64(static_cast<std::int64_t>(name.size()));
    put(name.data(), name.size());
    put_i64(static_cast<std::int64_t>(sp.offset(i)));
    put_i64(static_cast<std::int64_t>(sp.map(i).size()));
  }

  // the triplets, as 64-bit indices so a reader needs no width negotiation
  std::vector<std::int64_t> idx(static_cast<std::size_t>(nnz));
  for (std::size_t k = 0; k < static_cast<std::size_t>(nnz); ++k) idx[k] = jac.row[k];
  put(idx.data(), idx.size() * sizeof(std::int64_t));
  for (std::size_t k = 0; k < static_cast<std::size_t>(nnz); ++k) idx[k] = jac.col[k];
  put(idx.data(), idx.size() * sizeof(std::int64_t));
  put(jac.value.data(), jac.value.size() * sizeof(double));

  // the residual, the state, and what was pinned to what
  put(jac.residual.data(), jac.residual.size() * sizeof(double));
  put(sim.state().data(), sim.state().size() * sizeof(double));

  std::vector<std::int8_t> mask(static_cast<std::size_t>(n), 0);
  std::vector<double> value(static_cast<std::size_t>(n), 0.0);
  for (std::size_t d = 0; d < static_cast<std::size_t>(n); ++d) {
    if (sim.constraints().pinned(d)) {
      mask[d] = 1;
      value[d] = sim.constraints().value_at(d);
    }
  }
  put(mask.data(), mask.size());
  put(value.data(), value.size() * sizeof(double));
  if (!f) throw std::runtime_error("write_system: failed writing " + path);
}

// Assemble and write in one step, which is what a driver actually wants.
inline void export_system(const std::string& path, const Simulation& sim) {
  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  write_system(path, sim, jac);
}

}  // namespace mimetika::io
