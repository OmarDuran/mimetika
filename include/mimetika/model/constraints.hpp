#pragma once

#include <algorithm>
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

  // Put the constrained values into a state vector, so the very first
  // residual is already consistent with them.
  void apply_to_state(std::vector<double>& x) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) x[d] = value_of_[d];
    }
  }

  // r_i <- x_i - g_i on constrained rows.
  void apply_to_residual(const std::vector<double>& x, std::vector<double>& r) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) r[d] = x[d] - value_of_[d];
    }
  }

  // The tangent's action: a constrained row sees only its own direction.
  void apply_to_action(const std::vector<double>& v, std::vector<double>& y) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] != 0) y[d] = v[d];
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
  bool sorted_{false};
};

}  // namespace mimetika
