#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/forms/model.hpp"

// ESSENTIAL CONSTRAINTS: degrees of freedom whose value is imposed rather
// than solved for.
//
// Terzaghi's column is the smallest problem that needs all of them at once —
// rollers on the sides and base pin the normal displacement, sealed faces pin
// the normal flux, and get either wrong and the column stops being
// one-dimensional. They are not physics: pinning a degree of freedom is a
// statement about the discrete system, so it lives beside the model rather
// than inside a package.
//
// SUBSTITUTION, NOT PENALTY. A constrained row is replaced by x_i = g_i:
// the residual becomes the discrepancy, the Jacobian row becomes the
// identity, and the action of the tangent along a direction is the direction
// itself. That keeps the constraint EXACT — a penalty would leave an error
// scaling with the penalty parameter, and a benchmark checked against a
// closed form would then be measuring the penalty rather than the
// discretization.
//
// The column of a constrained unknown is left alone. That makes the system
// unsymmetric, which is deliberate: eliminating the column requires moving
// its contribution into the right-hand side, and doing that in a matrix-free
// setting means an extra operator apply per assembly. Keeping the column
// costs a solver that cannot assume symmetry and nothing else.
//
// THE CONSTRAINT ROW CARRIES THE SCALE OF THE EQUATION IT REPLACES.
//
// x_i = g_i and s x_i = s g_i are the same constraint for any s != 0, so the
// scale is free and the solution does not depend on it. It is not free for the
// FACTORIZATION. The row that gets replaced is a constitutive relation and it
// carries that relation's factors:
//
//     a traction moment    A_ii ~ 1/(2 mu)      1e-9 for rock
//     a normal flux moment A_ii ~ 1/(k dt)      1e14 for a small step
//
// Writing s = 1 into such a row puts it many orders of magnitude away from
// everything around it. The system stays exactly as well posed as before, but
// a direct factorization pivots on the wrong entries and reports a zero pivot
// somewhere else entirely -- MUMPS returns DIVERGED_PC_FAILED, on a problem
// with a perfectly good solution. Taking s from the diagonal the terms wrote
// on that row keeps the replaced equation in scale with the rest.
//
// It has to be the same s on all three paths. The residual becomes
// s (x_i - g_i) and the tangent's action s v_i, so
//
//     s dx_i = -s (x_i - g_i)   =>   x_i + dx_i = g_i
//
// exactly, and a Krylov method sees one consistent operator.

namespace mimetika {

using exokal::forms::Index;

class Constraints {
 public:
  // Pin one degree of freedom of the global numbering.
  void pin(Index dof, double value) {
    dofs_.push_back(dof);
    values_.push_back(value);
    sorted_ = false;
  }

  void pin_all(const std::vector<Index>& dofs, double value) {
    for (const Index d : dofs) pin(d, value);
  }

  std::size_t size() const { return dofs_.size(); }
  bool empty() const { return dofs_.empty(); }

  // A membership mask over the global numbering, built once. An assembly
  // touches every row, so asking "is this pinned" must be a lookup rather
  // than a search.
  void finalize(std::size_t n_dofs) {
    mask_.assign(n_dofs, 0);
    value_of_.assign(n_dofs, 0.0);
    for (std::size_t k = 0; k < dofs_.size(); ++k) {
      const auto d = static_cast<std::size_t>(dofs_[k]);
      if (d >= n_dofs) throw std::out_of_range("Constraints: dof outside the numbering");
      if (mask_[d] != 0 && value_of_[d] != values_[k]) {
        throw std::invalid_argument("Constraints: dof " + std::to_string(dofs_[k]) +
                                    " pinned to two different values");
      }
      mask_[d] = 1;
      value_of_[d] = values_[k];
    }
    sorted_ = true;
  }

  bool pinned(std::size_t d) const { return d < mask_.size() && mask_[d] != 0; }
  double value_at(std::size_t d) const { return value_of_[d]; }
  const std::vector<char>& mask() const { return mask_; }

  // The scale each constrained equation is written with, taken from the
  // diagonal of the row it replaces. Unity until it is measured, which is what
  // an unconstrained or unassembled system reduces to.
  double scale_at(std::size_t d) const { return scale_.empty() ? 1.0 : scale_[d]; }
  const std::vector<double>& scales() const { return scale_; }

  // Rows the terms left empty have no scale of their own and fall back to the
  // mean of the rest, so a degree of freedom no equation reached does not
  // become the pivot the factorization trips on.
  void set_scales(const std::vector<double>& diagonal) {
    require_final();
    if (diagonal.size() != mask_.size()) {
      throw std::invalid_argument("Constraints::set_scales: size");
    }
    double sum = 0.0;
    std::size_t count = 0;
    for (const double v : diagonal) {
      if (std::abs(v) > 0.0) {
        sum += std::abs(v);
        ++count;
      }
    }
    const double reference = count > 0 ? sum / static_cast<double>(count) : 1.0;
    scale_.assign(mask_.size(), 1.0);
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] == 0) continue;
      scale_[d] = std::abs(diagonal[d]) > 0.0 ? std::abs(diagonal[d]) : reference;
    }
  }

  // Put the constrained values into a state vector, so the very first
  // residual is already consistent with them.
  void apply_to_state(std::vector<double>& x) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) x[d] = value_of_[d];
    }
  }

  // r_i <- s_i (x_i - g_i) on constrained rows.
  void apply_to_residual(const std::vector<double>& x, std::vector<double>& r) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) r[d] = scale_at(d) * (x[d] - value_of_[d]);
    }
  }

  // The tangent's action: a constrained row sees only its own direction, with
  // the same scale its row of the assembled tangent carries.
  void apply_to_action(const std::vector<double>& v, std::vector<double>& y) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) y[d] = scale_at(d) * v[d];
    }
  }

 private:
  void require_final() const {
    if (!sorted_) throw std::logic_error("Constraints: finalize() before use");
  }

  std::vector<Index> dofs_;
  std::vector<double> values_;
  std::vector<char> mask_;
  std::vector<double> value_of_;
  std::vector<double> scale_;
  bool sorted_{false};
};

}  // namespace mimetika
