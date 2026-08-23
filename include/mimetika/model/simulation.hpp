#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "exokal/forms/epoch.hpp"
#include "exokal/forms/model.hpp"
#include "mimetika/model/constraints.hpp"
#include "mimetika/physics/package.hpp"

// The computational model: everything needed to advance a state, behind one
// object.
//
// A benchmark, a driver and a solver want the same three things from a
// discretized problem, otherwise assembled by hand out of an epoch, a model, a
// context, a workspace and a state vector. Those are six objects whose
// lifetimes and wiring order matter — the offsets must be set before the
// carrier maps are completed, the context must outlive the model, the space
// must outlive the epoch.
//
// Simulation is that wiring, done once. What it exposes:
//
//     residual(r)          r(x)
//     jacobian(sink)       the tangent, as triplets
//     apply(v, y)          y = J(x) v, with no matrix
//
// all three from one form source, and all three respecting the essential
// constraints, which is the part a hand-wired consumer most often forgets on
// one path and not the others.
//
// It does not solve. A linear solver is a dependency and a choice — direct or
// iterative, and with which preconditioner — so Simulation produces the
// operators and something else consumes them.

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
  // operators — and is not owned, so it must outlive the simulation. That is
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
  // Mutable until the constraints are frozen. A composition declares the
  // physics; a problem may still need a boundary term the physics cannot know
  // about -- a prescribed pressure applies to the borehole and not to the
  // column, and is a property of the configuration rather than of the model.
  Model& model() { return model_; }
  std::size_t n_dofs() const { return state_.size(); }

  std::vector<double>& state() { return state_; }
  const std::vector<double>& state() const { return state_; }

  // ---- what this process is responsible for -------------------------------
  //
  // Set together or not at all. `cells` is which sites of the top stratum this
  // process assembles, and `dofs` is which unknowns it writes a constraint row
  // for -- and those are different questions. A term's contribution is a sum,
  // so it may be split across processes and added back together; a constraint
  // row is a replacement, so exactly one process may write it or the equation
  // appears several times over.
  //
  // `reduce` sums a vector across the processes. It is used on the scales
  // alone -- one compact array, once per assembly -- because the diagonal a
  // constrained row is scaled by may have been assembled by a process that
  // does not own that row.
  void distribute_over(std::vector<char> cells, std::vector<char> facets, std::vector<char> dofs,
                       std::function<void(std::vector<double>&)> reduce) {
    owned_cells_ = std::move(cells);
    owned_facets_ = std::move(facets);
    owned_dofs_ = std::move(dofs);
    reduce_ = std::move(reduce);
    restricted_.clear();
  }

  bool distributed() const { return !owned_cells_.empty(); }
  const std::vector<char>& owned_dofs() const { return owned_dofs_; }

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
    model_.assemble(epoch_, state_, sink, ws_, colors());
    if (!constraints_.empty()) constraints_.apply_to_residual(state_, r);
  }

  // The tangent as triplets. Constrained rows are emitted as the identity,
  // and the entries the terms produced on those rows are dropped rather than
  // added to — a row that keeps both would not be a constraint.
  void jacobian(exokal::forms::TripletSink& sink) const {
    model_.assemble(epoch_, state_, sink, ws_, colors());
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
    model_.assemble(epoch_, state_, sink, ws_, colors());
    if (!constraints_.empty()) constraints_.apply_to_action(v, y);
  }

 private:
  // The restricted colouring of each (stratum, coupling), built once and kept:
  // a filter over the stratum's own, so the colours stay race-free and the
  // cells keep their numbering.
  exokal::forms::Model::ColorSource colors() const {
    if (!distributed()) return {};
    return [this](std::size_t stratum, exokal::forms::Coupling kind,
                  const exokal::forms::Epoch& e) -> const exokal::spaces::Coloring& {
      const auto key = std::pair<std::size_t, int>{stratum, static_cast<int>(kind)};
      const auto it = restricted_.find(key);
      if (it != restricted_.end()) return it->second;
      // the sites of a coupling are cells or facets, and exokal says which
      const std::vector<char>& keep =
          exokal::forms::evaluated_on_facets(kind) ? owned_facets_ : owned_cells_;
      return restricted_.emplace(key, exokal::spaces::restricted(e.colors(kind), keep))
          .first->second;
    };
  }

  bool writes_constraint(std::size_t dof) const {
    return owned_dofs_.empty() || owned_dofs_[dof] != 0;
  }

  // The scale, measured from an assembled tangent.
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
    // The diagonal of a row is not local, even where the row is: an interior
    // facet is assembled from both its cells, and those can belong to two
    // processes. A constrained row scaled by half its diagonal on one process
    // and half on another is two different equations, so the scales are summed
    // across the processes before they are used. Only the constrained rows
    // need it, and that is what is sent.
    if (reduce_ && !constraints_.empty()) {
      std::vector<std::size_t> which;
      std::vector<double> theirs;
      for (std::size_t i = 0; i < diagonal.size(); ++i) {
        if (!constraints_.pinned(i)) continue;
        which.push_back(i);
        theirs.push_back(diagonal[i]);
      }
      reduce_(theirs);
      for (std::size_t k = 0; k < which.size(); ++k) diagonal[which[k]] = theirs[k];
    }
    constraints_.set_scales(diagonal);
  }

  // A residual or a tangent-action asked for before any tangent has nothing to
  // read the scale from, so it assembles one. A consumer that assembles a
  // tangent first — which is every solver — never reaches this.
  void ensure_scales() const {
    if (constraints_.scaled()) return;
    exokal::forms::TripletSink probe(state_.size());
    model_.assemble(epoch_, state_, probe, ws_, colors());
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
    // The row is the form. A one-term form gives back a single scaled diagonal,
    // the familiar substitution; a normal traction on a facet whose normal is
    // not an axis gives d entries in that row.
    for (std::size_t d = 0; d < state_.size(); ++d) {
      if (!constraints_.pinned(d)) continue;
      // a replacement, so exactly one process writes it
      if (!writes_constraint(d)) continue;
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
  // the partition, as this layer needs it: which sites to assemble, which
  // constrained rows to write, and how to sum a vector across the processes
  std::vector<char> owned_cells_, owned_facets_, owned_dofs_;
  std::function<void(std::vector<double>&)> reduce_;
  mutable std::map<std::pair<std::size_t, int>, exokal::spaces::Coloring> restricted_;
};

}  // namespace mimetika
