#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exokal/forms/epoch.hpp"
#include "exokal/forms/model.hpp"
#include "mimetika/model/constraints.hpp"
#include "mimetika/physics/package.hpp"

// THE COMPUTATIONAL MODEL: everything needed to advance a state, behind one
// object.
//
// A benchmark, a driver and a solver all want the same three things from a
// discretized problem, and today they each have to assemble them by hand out
// of an epoch, a model, a context, a workspace and a state vector. That is
// six objects whose lifetimes and wiring order matter — the offsets must be
// set before the carrier maps are completed, the context must outlive the
// model, the space must outlive the epoch — and every consumer that
// rediscovers that order is a consumer that can get it wrong.
//
// Simulation is that wiring, done once. What it exposes is exactly what a
// solver needs and nothing else:
//
//     residual(r)          r(x)
//     jacobian(sink)       the tangent, as triplets
//     apply(v, y)          y = J(x) v, with no matrix
//
// all three from ONE form source, and all three respecting the essential
// constraints, which is the part a hand-wired consumer most often forgets on
// one path and not the others.
//
// WHAT IT DELIBERATELY IS NOT. It does not solve. A linear solver is a
// dependency and a choice — direct or iterative, and with which
// preconditioner — and binding one into the model would make the model a
// hostage to it. Simulation produces the operators; something else consumes
// them.

namespace mimetika {

using exokal::forms::Epoch;
using exokal::forms::Model;
using exokal::forms::Sink;
using exokal::forms::StratifiedEpoch;
using exokal::forms::TermContext;
using exokal::forms::Workspace;

// One stratum's contribution to the model: its complex, the dimension of its
// cells, and where it sits in the hierarchy.
struct StratumSpec {
  std::string name;
  const graphos::Complex* complex{nullptr};
  int cell_dim{0};
  int codim{0};
};

class Simulation {
 public:
  // The context carries the data named terms read — closures, discrete
  // operators — and is NOT owned, so it must outlive the simulation. That is
  // the same contract Epoch and Model already keep, kept once here instead
  // of once per consumer.
  Simulation(const physics::Composition& composition, std::vector<StratumSpec> strata,
             const TermContext& ctx)
      : ctx_(&ctx) {
    if (strata.empty()) throw std::invalid_argument("Simulation: no strata");
    for (const StratumSpec& s : strata) {
      if (s.complex == nullptr) throw std::invalid_argument("Simulation: null complex");
      // the composition decides the fields; the stratum decides their names,
      // through the codimension
      epoch_.add(s.name, s.codim,
                 Epoch(*s.complex, composition.space(*s.complex, s.cell_dim, s.codim),
                       s.cell_dim));
    }
    model_.use(ctx);
    composition.attach(model_, ctx);
    state_.assign(static_cast<std::size_t>(epoch_.size()), 0.0);
  }

  const StratifiedEpoch& epoch() const { return epoch_; }
  const Model& model() const { return model_; }
  std::size_t n_dofs() const { return state_.size(); }

  std::vector<double>& state() { return state_; }
  const std::vector<double>& state() const { return state_; }

  // The constraints are set up through this, then frozen. Freezing is what
  // builds the membership mask every assembly consults.
  Constraints& constraints() { return constraints_; }
  const Constraints& constraints() const { return constraints_; }
  void freeze_constraints() {
    constraints_.finalize(state_.size());
    constraints_.apply_to_state(state_);
  }

  // ---- the three operators, from one form source ------------------------

  void residual(std::vector<double>& r) const {
    r.assign(state_.size(), 0.0);
    exokal::forms::ResidualSink sink(r);
    model_.assemble(epoch_, state_, sink, ws_);
    if (!constraints_.empty()) constraints_.apply_to_residual(state_, r);
  }

  // The tangent as triplets. Constrained rows are emitted as the identity,
  // and the entries the terms produced on those rows are dropped rather than
  // added to — a row that keeps both would not be a constraint.
  void jacobian(exokal::forms::TripletSink& sink) const {
    model_.assemble(epoch_, state_, sink, ws_);
    if (constraints_.empty()) return;
    filter_constrained_rows(sink);
  }

  // y = J(x) v, matrix-free.
  void apply(const std::vector<double>& v, std::vector<double>& y) const {
    if (v.size() != state_.size()) throw std::invalid_argument("Simulation::apply: size");
    y.assign(state_.size(), 0.0);
    exokal::forms::ActionSink sink(v, y);
    model_.assemble(epoch_, state_, sink, ws_);
    if (!constraints_.empty()) constraints_.apply_to_action(v, y);
  }

 private:
  // Substitution on the assembled triplets: drop what the terms wrote on a
  // constrained row, then write the identity there.
  void filter_constrained_rows(exokal::forms::TripletSink& sink) const {
    std::size_t w = 0;
    for (std::size_t k = 0; k < sink.row.size(); ++k) {
      if (constraints_.pinned(static_cast<std::size_t>(sink.row[k]))) continue;
      sink.row[w] = sink.row[k];
      sink.col[w] = sink.col[k];
      sink.value[w] = sink.value[k];
      ++w;
    }
    sink.row.resize(w);
    sink.col.resize(w);
    sink.value.resize(w);
    for (std::size_t d = 0; d < state_.size(); ++d) {
      if (!constraints_.pinned(d)) continue;
      sink.row.push_back(static_cast<Index>(d));
      sink.col.push_back(static_cast<Index>(d));
      sink.value.push_back(1.0);
      sink.residual[d] = state_[d] - constraints_.value_at(d);
    }
  }

  const TermContext* ctx_;
  StratifiedEpoch epoch_;
  Model model_;
  Constraints constraints_;
  std::vector<double> state_;
  mutable Workspace ws_;
};

}  // namespace mimetika
