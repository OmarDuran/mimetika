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
// discretized problem, otherwise wired by hand out of six objects whose
// lifetimes and order matter — an epoch, a model, a context, a constraint set,
// a workspace and a state vector: the offsets must be set before the carrier
// maps are completed, the context must outlive the model, the space must
// outlive the epoch.
//
// Simulation is that wiring, done once. What it exposes:
//
//     residual(r)          r(x)
//     jacobian(sink)       the tangent, as triplets
//     apply(v, y)          y = J(x) v, with no matrix
//
// all three from one form source, and all three respecting the essential
// constraints.
//
// It does not solve: Simulation produces the operators, and the choice of
// linear solver and preconditioner belongs to the consumer.

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

  // Any sink at all, for a caller that wants what a tangent assembly produces
  // without the tangent: SchurSink keeps a diagonal and a divergence and drops
  // the rest, so the preconditioner is built from the same kernels the operator
  // applies and costs an assembly's arithmetic but none of its storage.
  //
  // The constrained rows are NOT filtered here, because filtering is a
  // TripletSink operation -- it needs the rows in hand to drop them. A caller
  // reading a preconditioner off this gets the terms' own entries on those
  // rows where jacobian() would have written the identity. That is a
  // preconditioner being slightly wrong, not an operator: Simulation::apply
  // carries the constraints itself.
  void assemble_into(exokal::forms::Sink& sink) const {
    if (!constraints_.empty()) ensure_scales();
    model_.assemble(epoch_, state_, sink, ws_, colors());
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

  // ---- symmetric elimination of the pinned unknowns -----------------------
  //
  // The three operators above are row-eliminated and NOT column-eliminated: a
  // constrained row carries the form, and the column of the unknown it leads
  // still carries whatever the terms wrote there. Constraints says why, and for
  // a general solve the price is a system that is merely unsymmetric.
  //
  // It is not merely unsymmetric for a method that assumes symmetry. MINRES
  // builds its Krylov space from a symmetric Lanczos recurrence and is silent
  // about the assumption: on the row-eliminated operator it converges, and to
  // something else. So a route that needs a symmetric operator asks for one
  // here, and gets it where the constraints admit one.
  //
  // Where every form is a single pin x_d = g_d, the elimination is the
  // classical one. Write P for the projector that zeroes the pinned entries and
  // x_c for the lift, (x_c)_d = g_d and zero elsewhere. Then
  //
  //     N = P A P + diag(s_d)|_pinned,     b~ = b - P A x_c,
  //
  // and N is symmetric whenever A is: P is diagonal, so P A P is the free block
  // of A bordered by zeros, and the pinned block is a decoupled diagonal.
  //
  // N x = b~ IS the row-eliminated system, not an approximation of it. Its
  // pinned rows read s_d x_d = s_d g_d, which is the row filter_constrained_rows
  // writes and the datum FlowModel puts in the load; its free rows are the free
  // rows of A x = b with the pinned columns carried across, which is what
  // substituting x_d = g_d into them gives. Same solution, same scale, both
  // paths.
  //
  // A MULTI-TERM FORM HAS NO SUCH ELIMINATION. Its row is s a^T and reaches
  // free columns, so symmetry would want column d to be s a -- a different
  // matrix, not a column moved across. Those are refused here rather than
  // silently left asymmetric.

  // Whether the elimination below applies: every form a single pin.
  bool pins_only() const {
    for (const Constraints::Form& f : constraints_.forms()) {
      if (f.dofs.size() != 1) return false;
    }
    return true;
  }

  // x_c: the value each pinned unknown is held at, zero elsewhere.
  std::vector<double> pinned_lift() const {
    require_pins("Simulation::pinned_lift");
    std::vector<double> x(state_.size(), 0.0);
    for (std::size_t d = 0; d < state_.size(); ++d) {
      if (constraints_.pinned(d)) x[d] = constraints_.value_at(d);
    }
    return x;
  }

  // y = N v, matrix-free. One assembly, as apply() is -- the projection and the
  // row are componentwise and cost nothing.
  void apply_symmetric(const std::vector<double>& v, std::vector<double>& y) const {
    if (v.size() != state_.size()) {
      throw std::invalid_argument("Simulation::apply_symmetric: size");
    }
    if (constraints_.empty()) {
      apply(v, y);
      return;
    }
    require_pins("Simulation::apply_symmetric");
    ensure_scales();
    projected_ = v;  // P v
    for (std::size_t d = 0; d < projected_.size(); ++d) {
      if (constraints_.pinned(d)) projected_[d] = 0.0;
    }
    y.assign(state_.size(), 0.0);
    exokal::forms::ActionSink sink(projected_, y);
    model_.assemble(epoch_, state_, sink, ws_, colors());
    // the row of a pinned unknown is its own, scaled: v and not projected_, so
    // the block is a scaled identity and not zero
    for (std::size_t d = 0; d < y.size(); ++d) {
      if (constraints_.pinned(d)) y[d] = constraints_.scale_at(d) * v[d];
    }
  }

  // b <- b - P A x_c: the pinned columns, carried to the right-hand side. The
  // pinned rows already hold their own datum and are left alone.
  //
  // ONE APPLY PER TANGENT, NOT PER ITERATION. This is the cost Constraints
  // declines to pay: an action against a fixed vector, once, before the Krylov
  // method starts. A linear solve pays it once; a Newton loop pays it once a
  // step, where a term dropped into every assembly would have been once an
  // iteration.
  void fold_pinned_columns(std::vector<double>& b) const {
    if (constraints_.empty()) return;
    if (b.size() != state_.size()) {
      throw std::invalid_argument("Simulation::fold_pinned_columns: size");
    }
    require_pins("Simulation::fold_pinned_columns");
    ensure_scales();
    const std::vector<double> lift = pinned_lift();
    std::vector<double> column(state_.size(), 0.0);
    exokal::forms::ActionSink sink(lift, column);
    model_.assemble(epoch_, state_, sink, ws_, colors());
    for (std::size_t i = 0; i < b.size(); ++i) {
      if (!constraints_.pinned(i)) b[i] -= column[i];
    }
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

  void require_pins(const char* who) const {
    for (const Constraints::Form& f : constraints_.forms()) {
      if (f.dofs.size() == 1) continue;
      throw std::invalid_argument(
          std::string(who) + ": a form spanning " + std::to_string(f.dofs.size()) +
          " unknowns cannot be eliminated symmetrically -- its row is s a^T and reaches "
          "columns the elimination leaves in place, so there is no single column to carry "
          "across. Take the row-eliminated tangent and a method that does not assume "
          "symmetry.");
    }
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
    set_scales_from(std::move(diagonal));
  }

  // The diagonal of a row is not local, even where the row is: an interior
  // facet is assembled from both its cells, and those can belong to two
  // processes. A constrained row scaled by half its diagonal on one process
  // and half on another is two different equations, so the scales are summed
  // across the processes before they are used. Only the constrained rows
  // need it, and that is what is sent.
  void set_scales_from(std::vector<double> diagonal) const {
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

  // The diagonal alone, which is all a scale is.
  //
  // NOT A TripletSink. ensure_scales() below runs whenever a residual or an
  // action is asked for before any tangent -- which for a matrix-free route is
  // always, and for mechanics is the ordinary case, a traction being the
  // essential condition there. A TripletSink probe would form the whole saddle
  // point to read one number per constrained row, handing back exactly the
  // memory build(false) exists to save. This keeps the n entries and drops the
  // n^2.
  //
  // Faithful to what measure_scales reads off a tangent, rather than merely
  // close to it: every entry the terms emit is visited and the ones whose row
  // and column are the same global unknown are summed, which is that filter
  // written out. The arithmetic is an assembly's; the storage is not.
  class DiagonalSink final : public exokal::forms::Sink {
   public:
    explicit DiagonalSink(std::size_t n) : diagonal(n, 0.0) {}
    std::vector<double> diagonal;

    void scatter(const exokal::forms::Stencil& st,
                 const exokal::ad::LocalSystem& sys) override {
      const auto& dofs = st.view.dofs;
      const exokal::ad::LocalSpace& sp = sys.space();
      for (std::size_t bi = 0; bi < sp.n_blocks(); ++bi) {
        for (std::size_t bj = 0; bj < sp.n_blocks(); ++bj) {
          if (!sys.has_block(bi, bj)) continue;
          const exokal::numerics::Dense& blk = sys.block(bi, bj);
          for (std::size_t i = 0; i < blk.rows(); ++i) {
            const auto r = dofs[sp.begin(bi) + i];
            for (std::size_t j = 0; j < blk.cols(); ++j) {
              if (dofs[sp.begin(bj) + j] != r) continue;
              diagonal[static_cast<std::size_t>(r)] += blk(i, j);
            }
          }
        }
      }
    }
  };

  // A residual or a tangent-action asked for before any tangent has nothing to
  // read the scale from, so it assembles one -- the diagonal of one, which is
  // the only part a scale is made of.
  void ensure_scales() const {
    if (constraints_.scaled()) return;
    DiagonalSink probe(state_.size());
    model_.assemble(epoch_, state_, probe, ws_, colors());
    set_scales_from(std::move(probe.diagonal));
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
  mutable std::vector<double> projected_;  // P v, kept so apply_symmetric allocates once
  // the partition, as this layer needs it: which sites to assemble, which
  // constrained rows to write, and how to sum a vector across the processes
  std::vector<char> owned_cells_, owned_facets_, owned_dofs_;
  std::function<void(std::vector<double>&)> reduce_;
  mutable std::map<std::pair<std::size_t, int>, exokal::spaces::Coloring> restricted_;
};

}  // namespace mimetika
