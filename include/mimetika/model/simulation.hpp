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
                 Epoch(*s.complex, composition.space(*s.complex, s.cell_dim, s.codim), s.cell_dim));
    }
    model_.use(ctx);
    composition.attach(model_, ctx);
    state_.assign(static_cast<std::size_t>(epoch_.size()), 0.0);
  }

  const StratifiedEpoch& epoch() const { return epoch_; }
  const Model& model() const { return model_; }
  // MUTABLE UNTIL THE CONSTRAINTS ARE FROZEN. A composition declares the
  // physics; a PROBLEM may still need a boundary term the physics cannot know
  // about -- a prescribed pressure applies to the borehole and not to the
  // column, and it is a property of the configuration rather than of the
  // model. Adding it here keeps that distinction where it belongs.
  Model& model() { return model_; }
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
    if (!constraints_.empty()) ensure_scales();
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
    // the scale of each constrained equation is the diagonal of the row about
    // to be replaced, and it is sitting in this very sink -- so it is read off
    // here rather than paid for with an assembly of its own
    if (!constraints_.scaled()) measure_scales(sink);
    filter_constrained_rows(sink);
  }

  // y = J(x) v, matrix-free.
  void apply(const std::vector<double>& v, std::vector<double>& y) const {
    if (v.size() != state_.size()) throw std::invalid_argument("Simulation::apply: size");
    if (!constraints_.empty()) ensure_scales();
    y.assign(state_.size(), 0.0);
    exokal::forms::ActionSink sink(v, y);
    model_.assemble(epoch_, state_, sink, ws_);
    if (!constraints_.empty()) constraints_.apply_to_action(v, y);
  }

 private:
  // THE SCALE, MEASURED FROM AN ASSEMBLED TANGENT.
  //
  // What is wanted is the diagonal the terms write on each constrained row:
  // the constitutive factor of the equation being replaced, which for a linear
  // model does not move with the state.
  void measure_scales(const exokal::forms::TripletSink& sink) const {
    std::vector<double> diagonal(state_.size(), 0.0);
    for (std::size_t k = 0; k < sink.nnz(); ++k) {
      if (sink.row[k] == sink.col[k]) {
        diagonal[static_cast<std::size_t>(sink.row[k])] += sink.value[k];
      }
    }
    constraints_.set_scales(diagonal);
  }

  // A residual or a tangent-action asked for BEFORE any tangent has nothing to
  // read the scale from, so it assembles one. That is the only path that pays
  // for it, and a consumer that assembles a tangent first — which is every
  // solver — never reaches this.
  void ensure_scales() const {
    if (constraints_.scaled()) return;
    exokal::forms::TripletSink probe(state_.size());
    model_.assemble(epoch_, state_, probe, ws_);
    measure_scales(probe);
  }

  // Substitution on the assembled triplets: drop what the terms wrote on a
  // constrained row, then write the constraint there, scaled by the diagonal
  // of the equation it replaces (see Constraints).
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
    // THE ROW IS THE FORM. A one-term form gives back a single scaled diagonal,
    // which is the familiar substitution; a normal traction on a facet whose
    // normal is not an axis gives d entries in that row, which is the same
    // statement written where it is actually true.
    for (std::size_t d = 0; d < state_.size(); ++d) {
      if (!constraints_.pinned(d)) continue;
      const double s = constraints_.scale_at(d);
      const Constraints::Form& f = constraints_.form_at(d);
      for (std::size_t j = 0; j < f.dofs.size(); ++j) {
        sink.row.push_back(static_cast<Index>(d));
        sink.col.push_back(f.dofs[j]);
        sink.value.push_back(s * f.coeff[j]);
      }
      sink.residual[d] = s * (Constraints::evaluate(f, state_) - f.value);
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
